# NOTE — GoL Cellular Automaton (Experimental — DNF on feature set C)
# ─────────────────────────────────────────────────────────────────────────────
# This model was evaluated in the v1 ablation study. It completed on feature
# sets A (ll=1.2054) and B (ll=1.0519) but did not terminate on feature set C
# within the time budget. This is consistent with exponential scaling of CA
# rule-table search with input dimensionality. Included for transparency.
# See docs/checkin_2.md §3 and data/processed/model_results_complete.csv.
# ─────────────────────────────────────────────────────────────────────────────

"""
Novel Model Family 3: Game of Life-Inspired Cellular Automaton
===============================================================
Maps each team's fingerprint onto a small grid of cells. A matchup is
modeled as two grids "interacting" via locally-evolved rules. The final
cell state distribution is read out as a feature vector for classification.

Motivation: standard models treat a team's performance as the *sum* of its
features. This model treats it as *emergent behavior* from local feature
interactions — a team's press resistance emerges from the interaction of
pressing intensity, squad fitness, and opponent pressing rate, not their
sum. By running a cellular automaton over the discretized feature grid, we
allow complex non-linear dynamics to emerge that a fixed MLP would need many
layers to approximate.

Architecture:
  1. Discretize: each feature → binary cell state (above/below median)
  2. Grid layout: 4×4 grid (16 cells per team), 2 teams → 4×8 combined grid
  3. Interaction rules: 3×3 neighborhood Conway-style, rules evolved by GA
  4. Readout: final cell state counts → logistic regression

Rule encoding: each cell's next state = f(3×3 neighborhood sum ∈ {0..9}).
That's 10 possible inputs per cell. With a binary output: 2^10 = 1024 possible
rule tables. The GA evolves the rule table that minimizes prediction log-loss.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

try:
    from deap import base, creator, tools, algorithms
    HAS_DEAP = True
except ImportError:
    HAS_DEAP = False

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

GRID_ROWS = 4
GRID_COLS = 4    # per team; combined grid is 4×8
N_STEPS   = 3   # CA evolution steps
# Rule table: maps neighborhood sum (0..8 for Moore neighborhood) → 0/1
# We use a simpler totalistic rule: 9 bits (sum 0..8 for neighbors only)
RULE_BITS = 9    # neighborhood sums 0..8


def features_to_grid(x: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """
    Convert a 1D feature vector to a 4×4 binary grid.
    Features are binarized by comparison to training medians.
    Remaining cells (if n_features < 16) are zero-padded.
    """
    binary = (x > medians[:len(x)]).astype(np.uint8)
    padded = np.zeros(GRID_ROWS * GRID_COLS, dtype=np.uint8)
    padded[:len(binary)] = binary[:GRID_ROWS * GRID_COLS]
    return padded.reshape(GRID_ROWS, GRID_COLS)


def make_matchup_grid(home_grid: np.ndarray, away_grid: np.ndarray) -> np.ndarray:
    """Stack home (4×4) and away (4×4) grids side by side → 4×8 combined grid."""
    return np.concatenate([home_grid, away_grid], axis=1)


def ca_step(grid: np.ndarray, rule: np.ndarray) -> np.ndarray:
    """
    One step of totalistic cellular automaton.
    rule: array of length 9 (indexed by Moore neighbor sum 0..8)
          maps sum → new state (0 or 1)
    """
    rows, cols = grid.shape
    new_grid = np.zeros_like(grid)
    for r in range(rows):
        for c in range(cols):
            # Moore neighborhood (8 neighbors)
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = (r + dr) % rows, (c + dc) % cols
                    neighbors.append(grid[nr, nc])
            s = int(sum(neighbors))
            new_grid[r, c] = rule[s]
    return new_grid


def run_ca(grid: np.ndarray, rule: np.ndarray, n_steps: int) -> np.ndarray:
    """Run cellular automaton for n_steps steps."""
    for _ in range(n_steps):
        grid = ca_step(grid, rule)
    return grid


def extract_features(grid: np.ndarray) -> np.ndarray:
    """
    Extract readout features from final CA state:
      - total alive cells (density)
      - row densities (4 features)
      - col densities (8 features)
      - quadrant densities (4 features)
      - alive in home half vs. away half
    Total: 18 features
    """
    rows, cols = grid.shape
    half = cols // 2
    feats = [
        grid.sum() / grid.size,                             # overall density
        *[grid[r, :].sum() / cols for r in range(rows)],   # row densities
        *[grid[:, c].sum() / rows for c in range(cols)],   # col densities
        grid[:2, :half].mean(), grid[:2, half:].mean(),    # top quadrants
        grid[2:, :half].mean(), grid[2:, half:].mean(),    # bot quadrants
        grid[:, :half].mean(),                              # home half density
        grid[:, half:].mean(),                              # away half density
    ]
    return np.array(feats)


class GoLClassifier(BaseEstimator, ClassifierMixin):
    """
    Game of Life-Inspired Cellular Automaton Classifier.

    1. Binarize features per team → 4×4 grid each
    2. Stack grids → 4×8 matchup grid
    3. Evolve CA for N_STEPS steps using GA-optimized rule table
    4. Read out final state features → logistic regression head
    """

    def __init__(self, n_generations: int = 40, pop_size: int = 50,
                 random_state: int = RANDOM_SEED):
        self.n_generations = n_generations
        self.pop_size      = pop_size
        self.random_state  = random_state
        self.best_rule     = None
        self.readout       = None
        self.medians_home  = None
        self.medians_away  = None
        self.classes_      = np.array([0, 1, 2])
        self.scaler_out    = StandardScaler()

    def _build_ca_features(self, X_home, X_away, rule):
        """Build CA readout feature matrix for all matches."""
        n = X_home.shape[0]
        out = []
        for i in range(n):
            hg = features_to_grid(X_home[i], self.medians_home)
            ag = features_to_grid(X_away[i], self.medians_away)
            grid = make_matchup_grid(hg, ag)
            final = run_ca(grid, rule, N_STEPS)
            out.append(extract_features(final))
        return np.array(out)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_home: np.ndarray = None, X_away: np.ndarray = None):
        """
        X_home, X_away: raw feature matrices for home/away teams.
        If not provided, split X in half (first half = home features).
        """
        rng = np.random.RandomState(self.random_state)

        half = X.shape[1] // 2
        if X_home is None:
            X_home = X[:, :half]
        if X_away is None:
            X_away = X[:, half:] if X.shape[1] > half else X[:, :half]

        # Ensure same width
        min_w = min(X_home.shape[1], GRID_ROWS * GRID_COLS)
        X_home = X_home[:, :min_w]
        X_away = X_away[:, :min_w]

        self.medians_home = np.median(X_home, axis=0)
        self.medians_away = np.median(X_away, axis=0)

        def evaluate(individual):
            rule = np.array(individual, dtype=np.uint8)
            try:
                ca_feats = self._build_ca_features(X_home, X_away, rule)
                ca_feats_s = self.scaler_out.fit_transform(ca_feats)
                head = LogisticRegression(C=1.0, max_iter=200,
                                          multi_class="multinomial",
                                          random_state=self.random_state)
                head.fit(ca_feats_s, y)
                probs = head.predict_proba(ca_feats_s)
                ll = log_loss(y, np.clip(probs, 1e-7, 1 - 1e-7))
            except Exception:
                ll = 10.0
            return (ll,)

        if not HAS_DEAP:
            # Random rule search fallback
            best_ll = np.inf
            best_rule = rng.randint(0, 2, RULE_BITS)
            for _ in range(self.pop_size * self.n_generations):
                rule = rng.randint(0, 2, RULE_BITS)
                ll = evaluate(rule)[0]
                if ll < best_ll:
                    best_ll = ll
                    best_rule = rule.copy()
            self.best_rule = best_rule
        else:
            for attr in ["FitnessMinGoL", "IndividualGoL"]:
                if hasattr(creator, attr):
                    delattr(creator, attr)

            creator.create("FitnessMinGoL", base.Fitness, weights=(-1.0,))
            creator.create("IndividualGoL", list, fitness=creator.FitnessMinGoL)

            toolbox = base.Toolbox()
            toolbox.register("bit", lambda: int(rng.randint(0, 2)))
            toolbox.register("individual", tools.initRepeat,
                             creator.IndividualGoL, toolbox.bit, n=RULE_BITS)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", evaluate)
            toolbox.register("mate",   tools.cxUniform, indpb=0.5)
            toolbox.register("mutate", tools.mutFlipBit, indpb=0.15)
            toolbox.register("select", tools.selTournament, tournsize=3)

            pop = toolbox.population(n=self.pop_size)
            algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3,
                                ngen=self.n_generations, verbose=False)

            best_ind = tools.selBest(pop, k=1)[0]
            self.best_rule = np.array(best_ind, dtype=np.uint8)

        # Fit final readout head with best rule
        ca_feats = self._build_ca_features(X_home, X_away, self.best_rule)
        ca_feats_s = self.scaler_out.fit_transform(ca_feats)
        self.readout = LogisticRegression(C=1.0, max_iter=300,
                                          multi_class="multinomial",
                                          random_state=self.random_state)
        self.readout.fit(ca_feats_s, y)
        self._X_home_cols = X_home.shape[1]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        half = self._X_home_cols
        X_home = X[:, :half]
        X_away = X[:, :half]   # symmetric at test time (no home/away split known)
        ca_feats = self._build_ca_features(X_home, X_away, self.best_rule)
        ca_feats_s = self.scaler_out.transform(ca_feats)
        return np.clip(self.readout.predict_proba(ca_feats_s), 1e-7, 1 - 1e-7)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
