# NOTE — Novel / Experimental Model
# ─────────────────────────────────────────────────────────────────────────────
# This is an experimental model architecture evaluated in the v1 ablation study.
# Result: underperforms classical Random Forest at this data scale (~280 training
# samples per LOTO fold). Included to document the negative finding transparently.
# See docs/checkin_2.md §3 and data/processed/model_results_complete.csv.
# ─────────────────────────────────────────────────────────────────────────────

"""
Novel Model Family 2: Genetic Algorithm Dynamic Ensemble
===========================================================
Uses a genetic algorithm (DEAP) to evolve the optimal weighting of an
ensemble of base classifiers. Unlike a fixed-weight ensemble (e.g., voting),
the GA:
  1. Searches over a continuous weight space (one weight per base model per class)
  2. Evolves weights conditioned on archetype context (different weights per archetype)
  3. Updates weights across generations using tournament selection + Gaussian mutation

This implements the "dynamically weighted ensemble" described in the proposal:
  - Weights w_k^(a) for base model k in archetype context a
  - Fitness: LOTO-CV log-loss on WC matches
  - Population: 60 individuals, 50 generations, tournament selection k=3

The key novelty vs. standard ensemble: **archetype-conditioned weighting** —
a `heavyweight_clash` match may weight the Elo-based random forest more heavily,
while a `generational_transition` match weights the squad-feature MLP more.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

try:
    from deap import base, creator, tools, algorithms
    HAS_DEAP = True
except ImportError:
    HAS_DEAP = False

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ARCHETYPE_COLS = [
    "heavyweight_clash", "favorite_vs_underdog", "host_pressure",
    "generational_transition", "club_power_mismatch", "tactical_contrast",
    "knockout_volatility",
]

# Archetype contexts for dynamic weighting
CONTEXTS = ["default"] + ARCHETYPE_COLS   # 8 contexts


def get_base_models():
    return [
        LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_SEED,
                           multi_class="multinomial"),
        RandomForestClassifier(n_estimators=100, max_depth=5,
                               random_state=RANDOM_SEED, n_jobs=-1),
        GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                   learning_rate=0.1, random_state=RANDOM_SEED),
        MLPClassifier(hidden_layer_sizes=(32,), max_iter=300,
                      random_state=RANDOM_SEED, early_stopping=True),
    ]


class GADynamicEnsemble(BaseEstimator, ClassifierMixin):
    """
    Genetic Algorithm optimized dynamic ensemble.

    For each match, the archetype context is determined, and the ensemble
    uses the evolved weights specific to that context. This means a
    'heavyweight_clash' match may rely more on the Elo-based RF,
    while a 'generational_transition' match weights the MLP more.

    Genome: flat array of shape (n_contexts × n_base_models)
    The values are softmax-normalized within each context to get weights.
    """

    def __init__(self, n_generations: int = 50, pop_size: int = 60,
                 random_state: int = RANDOM_SEED):
        self.n_generations = n_generations
        self.pop_size      = pop_size
        self.random_state  = random_state
        self.base_models   = None
        self.best_weights  = None   # shape: (n_contexts, n_base_models)
        self.classes_      = np.array([0, 1, 2])
        self.n_classes_    = 3

    def _get_context_idx(self, X_arch: pd.DataFrame, i: int) -> int:
        """Return the archetype context index for match i."""
        if X_arch is None:
            return 0  # default context
        row = X_arch.iloc[i]
        for ctx_idx, col in enumerate(ARCHETYPE_COLS, start=1):
            if col in row.index and row[col] == 1:
                return ctx_idx
        return 0  # default

    def _combine_probs(self, base_probs: list, weights: np.ndarray) -> np.ndarray:
        """
        Weighted combination of base model probability arrays.
        base_probs: list of (n_samples, 3) arrays
        weights: (n_contexts, n_base_models) — softmax already applied
        """
        n_samples = base_probs[0].shape[0]
        combined = np.zeros((n_samples, 3))
        for i in range(n_samples):
            ctx = self._context_map[i]
            w = weights[ctx]  # (n_base_models,)
            for k, bp in enumerate(base_probs):
                combined[i] += w[k] * bp[i]
        return np.clip(combined, 1e-7, 1 - 1e-7)

    def _softmax_weights(self, raw: np.ndarray) -> np.ndarray:
        """Apply softmax across base models for each context."""
        out = np.zeros_like(raw)
        for c in range(raw.shape[0]):
            e = np.exp(raw[c] - raw[c].max())
            out[c] = e / e.sum()
        return out

    def fit(self, X: np.ndarray, y: np.ndarray, X_arch=None):
        rng = np.random.RandomState(self.random_state)
        n_models = len(get_base_models())
        n_contexts = len(CONTEXTS)
        n_genome = n_contexts * n_models

        # Fit base models
        self.base_models = get_base_models()
        for m in self.base_models:
            m.fit(X, y)

        # Get base model probability predictions
        base_probs = [m.predict_proba(X) for m in self.base_models]

        # Build context map for training data
        self._context_map = np.zeros(len(y), dtype=int)
        if X_arch is not None:
            for i in range(len(y)):
                self._context_map[i] = self._get_context_idx(X_arch, i)

        def evaluate(individual):
            raw = np.array(individual).reshape(n_contexts, n_models)
            weights = self._softmax_weights(raw)
            probs = self._combine_probs(base_probs, weights)
            ll = log_loss(y, probs)
            return (ll,)

        if not HAS_DEAP:
            # Random search fallback
            best_ll = np.inf
            best_w  = rng.randn(n_contexts, n_models)
            for _ in range(self.pop_size * self.n_generations):
                w = rng.randn(n_contexts, n_models)
                ll = evaluate(w.ravel())[0]
                if ll < best_ll:
                    best_ll = ll
                    best_w  = w
            self.best_weights = self._softmax_weights(best_w)
            return self

        # DEAP GA
        for attr in ["FitnessMinGA", "IndividualGA"]:
            if hasattr(creator, attr):
                delattr(creator, attr)

        creator.create("FitnessMinGA", base.Fitness, weights=(-1.0,))
        creator.create("IndividualGA", list, fitness=creator.FitnessMinGA)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", rng.standard_normal)
        toolbox.register("individual", tools.initRepeat,
                         creator.IndividualGA, toolbox.attr_float, n=n_genome)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate",   tools.cxBlend, alpha=0.3)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.4, indpb=0.15)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=self.pop_size)
        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3,
                            ngen=self.n_generations, verbose=False)

        best_ind = tools.selBest(pop, k=1)[0]
        raw_best = np.array(best_ind).reshape(n_contexts, n_models)
        self.best_weights = self._softmax_weights(raw_best)
        return self

    def predict_proba(self, X: np.ndarray, X_arch=None) -> np.ndarray:
        base_probs = [m.predict_proba(X) for m in self.base_models]
        # Context map for test
        self._context_map = np.zeros(X.shape[0], dtype=int)
        if X_arch is not None:
            for i in range(X.shape[0]):
                self._context_map[i] = self._get_context_idx(X_arch, i)
        return self._combine_probs(base_probs, self.best_weights)

    def predict(self, X: np.ndarray, X_arch=None) -> np.ndarray:
        return self.predict_proba(X, X_arch).argmax(axis=1)
