const DATA_URL = "./data/matchups.json";

const els = {
  matchCount: document.querySelector("#match-count"),
  modelLoss: document.querySelector("#model-loss"),
  validationAcc: document.querySelector("#validation-acc"),
  stageFilter: document.querySelector("#stage-filter"),
  matchSelect: document.querySelector("#match-select"),
  countryFilter: document.querySelector("#country-filter"),
  resetCountry: document.querySelector("#reset-country"),
  upsetToggle: document.querySelector("#upset-toggle"),
  auditSummary: document.querySelector("#audit-summary"),
  auditStatGrid: document.querySelector("#audit-stat-grid"),
  auditStateList: document.querySelector("#audit-state-list"),
  auditTimestamp: document.querySelector("#audit-timestamp"),
  homeCode: document.querySelector("#home-code"),
  awayCode: document.querySelector("#away-code"),
  homeTeam: document.querySelector("#home-team"),
  awayTeam: document.querySelector("#away-team"),
  matchStage: document.querySelector("#match-stage"),
  matchDate: document.querySelector("#match-date"),
  matchScore: document.querySelector("#match-score"),
  badgeList: document.querySelector("#badge-list"),
  snapshotNote: document.querySelector("#snapshot-note"),
  legendHome: document.querySelector("#legend-home"),
  legendAway: document.querySelector("#legend-away"),
  radar: document.querySelector("#radar-canvas"),
  probBars: document.querySelector("#prob-bars"),
  probTable: document.querySelector("#prob-table"),
  upsetPill: document.querySelector("#upset-pill"),
  predictionCall: document.querySelector("#prediction-call"),
  reasoningList: document.querySelector("#reasoning-list"),
  lineupStatus: document.querySelector("#lineup-status"),
  lineupToggle: document.querySelector("#lineup-toggle"),
  lineupBody: document.querySelector("#lineup-body"),
  lineupBreakdown: document.querySelector("#lineup-breakdown"),
  timelineList: document.querySelector("#timeline-list"),
  lineupEmpty: document.querySelector("#lineup-empty"),
  similarGrid: document.querySelector("#similar-grid"),
  metricList: document.querySelector("#metric-list"),
  importanceList: document.querySelector("#importance-list"),
  dataCaveat: document.querySelector("#data-caveat"),
  sectionNavLinks: document.querySelectorAll(".section-nav-link"),
};

const state = {
  data: null,
  matches: [],
  filtered: [],
  selectedId: null,
  countries: [],
  countryCounts: new Map(),
  lineupExpanded: false,
};

const archetypeClass = new Map([
  ["Heavyweight Clash", "heavyweight_clash"],
  ["Favorite Vs Underdog", "favorite_vs_underdog"],
  ["Host Pressure", "host_pressure"],
  ["Generational Transition", "generational_transition"],
  ["Club Power Mismatch", "club_power_mismatch"],
  ["Tactical Contrast", "tactical_contrast"],
  ["Knockout Volatility", "knockout_volatility"],
  ["Upset Realized", "upset_realized"],
]);

const TEAM_FLAG_CODES = {
  algeria: "DZ",
  argentina: "AR",
  australia: "AU",
  austria: "AT",
  belgium: "BE",
  "bosnia and herzegovina": "BA",
  brazil: "BR",
  canada: "CA",
  "cape verde": "CV",
  colombia: "CO",
  croatia: "HR",
  curacao: "CW",
  "czech republic": "CZ",
  "dr congo": "CD",
  ecuador: "EC",
  egypt: "EG",
  france: "FR",
  germany: "DE",
  ghana: "GH",
  haiti: "HT",
  iran: "IR",
  iraq: "IQ",
  "ivory coast": "CI",
  japan: "JP",
  jordan: "JO",
  mexico: "MX",
  morocco: "MA",
  netherlands: "NL",
  "new zealand": "NZ",
  norway: "NO",
  panama: "PA",
  paraguay: "PY",
  portugal: "PT",
  qatar: "QA",
  "saudi arabia": "SA",
  senegal: "SN",
  "south africa": "ZA",
  "south korea": "KR",
  spain: "ES",
  sweden: "SE",
  switzerland: "CH",
  tunisia: "TN",
  turkey: "TR",
  "united states": "US",
  uruguay: "UY",
  uzbekistan: "UZ",
};

const TEAM_SUBDIVISION_FLAG_CODES = {
  england: "gbeng",
  scotland: "gbsct",
};

function normalizeTeamKey(team) {
  return String(team || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function regionalFlag(code) {
  if (!/^[A-Z]{2}$/.test(code || "")) return "";
  return [...code].map((char) => String.fromCodePoint(0x1f1e6 + char.charCodeAt(0) - 65)).join("");
}

function subdivisionFlag(code) {
  if (!/^[a-z]{5}$/.test(code || "")) return "";
  const tagPoints = [...code].map((char) => 0xe0061 + char.charCodeAt(0) - 97);
  return String.fromCodePoint(0x1f3f4, ...tagPoints, 0xe007f);
}

function teamInitials(team) {
  return String(team || "")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((word) => word[0]?.toUpperCase() || "")
    .join("");
}

function teamFlag(team) {
  const key = normalizeTeamKey(team);
  if (TEAM_SUBDIVISION_FLAG_CODES[key]) return subdivisionFlag(TEAM_SUBDIVISION_FLAG_CODES[key]);
  return regionalFlag(TEAM_FLAG_CODES[key]);
}

function setTeamFlag(element, team) {
  const flag = teamFlag(team);
  element.textContent = flag || teamInitials(team);
  element.classList.toggle("team-mark-fallback", !flag);
  element.setAttribute("aria-label", flag ? `${team} flag` : `${team} flag unavailable`);
  element.title = flag ? `${team} flag` : `${team}`;
}

function pct(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function number(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(digits);
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char];
  });
}

function cleanLabel(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function teamColor(team, fallback) {
  let hash = 0;
  for (const char of team || "") hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  const hue = hash % 360;
  return `hsl(${hue}, 70%, ${fallback === "home" ? "56%" : "52%"})`;
}

function teamNameSize(team) {
  const length = String(team || "").length;
  if (length >= 24) return "1.05rem";
  if (length >= 19) return "1.2rem";
  if (length >= 15) return "1.42rem";
  return "clamp(1.25rem, 2vw, 1.85rem)";
}

function setTeamName(element, team) {
  element.textContent = team;
  element.style.setProperty("--team-name-size", teamNameSize(team));
}

function resultWord(code) {
  if (code === "H") return "Home win";
  if (code === "A") return "Away win";
  if (code === "D") return "Draw";
  return "Unknown";
}

function localDateString(date = new Date()) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function dateToDayNumber(value) {
  const match = String(value || "").match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!match) return null;
  const [, year, month, day] = match;
  return Math.floor(Date.UTC(Number(year), Number(month) - 1, Number(day)) / 86400000);
}

function dayDifference(later, earlier) {
  const laterDay = dateToDayNumber(later);
  const earlierDay = dateToDayNumber(earlier);
  if (laterDay === null || earlierDay === null) return null;
  return laterDay - earlierDay;
}

function asOfLabel(summary) {
  if (summary.as_of_local) return summary.as_of_local.replace(/\.\d+/, "").replace("T", " ");
  if (summary.as_of_utc) return summary.as_of_utc.replace(/\.\d+/, "").replace("T", " ").replace("Z", " UTC");
  return summary.as_of_date || localDateString();
}

function shortDateTime(value) {
  if (!value) return "unknown";
  return String(value).replace(/\.\d+/, "").replace("T", " ").replace("+00:00", "Z");
}

function fixtureStatusText(match, today) {
  if (match.is_imputed_prediction) return "Prediction is available; actual result has not been ingested yet.";
  if (match.prediction_actual_status === "actualized_prediction_correct") return "Actual result is available and matches the model call.";
  if (match.prediction_actual_status === "actualized_prediction_missed") return "Actual result is available and differs from the model call.";
  if (match.model_status === "Modeled row; actual result pending.") return match.model_status;
  if (match.prediction_scope === "pre_match_forecast") return "Forecast generated before kickoff; this match has not happened yet.";
  if (match.prediction_scope === "post_kickoff_pending_result") return "Forecast row exists, but kickoff has passed and the result is not ingested yet.";
  if (match.prediction_scope === "result_available_retrospective") return "Result is already visible in the schedule/status layer; this forecast is retrospective.";
  if (match.status_label) return match.status_label;
  const diff = dayDifference(match.date, today);
  if (diff === null) return "Scheduled fixture from the raw source; not yet harmonized into the model-ready layer.";
  if (diff < 0) return "Past scheduled fixture; the local dataset has not ingested the result yet.";
  if (diff === 0) return "Scheduled for today; not yet harmonized into the model-ready prediction layer.";
  return "Upcoming scheduled fixture; not yet harmonized into the model-ready prediction layer.";
}

function fixtureTimingLine(match) {
  const parts = [stageDisplay(match.stage), match.date];
  if (match.kickoff_time_local) parts.push(`kickoff ${match.kickoff_time_local}`);
  if (match.venue) parts.push(match.venue);
  if (match.score) parts.push(`result ${match.score}`);
  return parts.join(" · ");
}

function sectionLetter(stage) {
  const match = String(stage || "").match(/^Group\s+([A-L])$/i);
  return match ? match[1].toUpperCase() : null;
}

function stageDisplay(stage) {
  const letter = sectionLetter(stage);
  if (letter) return `Opening round · Section ${letter}`;
  if (!stage || stage === "Unassigned") return "Opening round · Section label missing";
  const normalized = String(stage).toLowerCase();
  if (normalized === "group stage") return "Opening round";
  if (normalized === "second group stage") return "Second opening round";
  if (normalized.includes("third-place")) return "Third-place match";
  if (/round of 32/i.test(stage)) return "First elimination round · 32 countries left";
  if (/round of 16/i.test(stage)) return "Second elimination round · 16 countries left";
  if (/quarter/i.test(stage)) return "Quarterfinal · 8 countries left";
  if (/semi/i.test(stage)) return "Semifinal · 4 countries left";
  if (/final/i.test(stage)) return "Final";
  return String(stage);
}

function stageSortValue(stage) {
  const letter = sectionLetter(stage);
  if (letter) return letter.charCodeAt(0);
  if (!stage || stage === "Unassigned") return 900;
  return 500 + String(stage).localeCompare("zzzz");
}

function predictedTeam(match) {
  if (match.predicted_result === "H") return match.home_team;
  if (match.predicted_result === "A") return match.away_team;
  return "Draw";
}

function predictedSide(match) {
  if (match.predicted_result === "H") return "home";
  if (match.predicted_result === "A") return "away";
  if (match.predicted_result === "D") return "draw";
  return "unknown";
}

function predictionProbability(match) {
  if (match.predicted_result === "H") return match.probabilities?.home;
  if (match.predicted_result === "A") return match.probabilities?.away;
  if (match.predicted_result === "D") return match.probabilities?.draw;
  return null;
}

function isScheduledForecast(match) {
  return match.prediction_source === "scheduled_fixture_forecast";
}

function isActualized(match) {
  return match.actual_available === true || Boolean(match.score);
}

function modelGrade(match) {
  if (!isActualized(match)) return "pending_actual";
  return match.prediction_grade || (match.prediction_correct ? "exact_correct" : "incorrect");
}

function gradeLabel(match) {
  const grade = modelGrade(match);
  if (grade === "exact_correct") return "Model correct";
  if (grade === "draw_push") return "Actual draw · partial credit";
  if (grade === "incorrect") return "Model incorrect";
  return "Actual pending";
}

function timelineGradeClass(match) {
  const grade = modelGrade(match);
  if (grade === "exact_correct") return "timeline-card-correct";
  if (grade === "draw_push") return "timeline-card-draw-push";
  if (grade === "incorrect") return "timeline-card-incorrect";
  return "timeline-card-pending";
}

function chronologyValue(match) {
  const kickoff = Date.parse(match.kickoff_at_utc || "");
  if (!Number.isNaN(kickoff)) return kickoff;
  const date = Date.parse(`${match.date || "9999-12-31"}T00:00:00Z`);
  return Number.isNaN(date) ? Number.MAX_SAFE_INTEGER : date;
}

function chronologySort(a, b) {
  return (
    chronologyValue(a) - chronologyValue(b) ||
    String(a.home_team).localeCompare(String(b.home_team)) ||
    String(a.away_team).localeCompare(String(b.away_team))
  );
}

function matchStatusLabel(match) {
  if (isActualized(match)) {
    return gradeLabel(match);
  }
  if (match.prediction_timing === "post_kickoff_pending_result") return "Past kickoff · actual pending";
  if (match.prediction_timing === "pre_match") return "Upcoming · forecast pending actual";
  return "Forecast pending actual";
}

function timingLabel(match) {
  if (match.prediction_timing === "pre_match") return "Pre-match forecast";
  if (match.prediction_timing === "post_kickoff_pending_result") return "Post-kickoff pending result";
  if (match.prediction_timing === "retrospective") return "Retrospective validation";
  return cleanLabel(match.prediction_timing || "Unknown timing");
}

function localFeatureSummary(match) {
  const features = match.features || {};
  const parts = [];
  const elo = Number(features.elo_gap_abs);
  if (!Number.isNaN(elo)) parts.push(`Elo gap ${number(elo, 0)}`);
  if (features.win_prob_home !== null && features.win_prob_home !== undefined) {
    parts.push(`home prior ${pct(features.win_prob_home, 0)}`);
  }
  const age = Number(features.age_mean_gap);
  if (!Number.isNaN(age)) parts.push(`age gap ${number(age, 1)}`);
  const league = Number(features.league_div_gap);
  if (!Number.isNaN(league)) parts.push(`league diversity gap ${number(league, 2)}`);
  return parts.slice(0, 4).join(" · ") || "Feature values are limited for this row.";
}

function buildCountryFilter() {
  state.countries = [
    ...new Set(state.matches.flatMap((match) => [match.home_team, match.away_team]).filter(Boolean)),
  ].sort((a, b) => a.localeCompare(b));

  state.countryCounts = new Map();
  for (const match of state.matches) {
    for (const country of [match.home_team, match.away_team]) {
      if (!country) continue;
      state.countryCounts.set(country, (state.countryCounts.get(country) || 0) + 1);
    }
  }

  els.countryFilter.replaceChildren();
  els.countryFilter.append(new Option("All countries", "all"));
  for (const country of state.countries) {
    const flag = teamFlag(country);
    const count = state.countryCounts.get(country) || 0;
    const label = `${flag ? `${flag} ` : ""}${country} (${count})`;
    els.countryFilter.append(new Option(label, country));
  }
}

function auditRoleLabel(role) {
  return (
    {
      retrospective_2026_holdout_validation: "2026 holdout validation, retrospective rows",
      scheduled_actualized_validation: "Scheduled forecasts with actuals now available",
      live_pre_match_forecast_pending_actual: "Live pre-match forecasts awaiting actuals",
      post_kickoff_forecast_pending_actual: "Post-kickoff rows still awaiting actuals",
      announced_fixture_pending_actual: "Announced fixtures awaiting actuals",
    }[role] || cleanLabel(role)
  );
}

function selectedMatch() {
  return state.matches.find((match) => String(match.match_id) === String(state.selectedId)) || state.filtered[0] || state.matches[0];
}

function buildStageFilter() {
  const stages = [...new Set(state.matches.map((match) => match.stage || "Unassigned"))].sort((a, b) => {
    return stageSortValue(a) - stageSortValue(b) || String(a).localeCompare(String(b), undefined, { numeric: true });
  });

  els.stageFilter.replaceChildren();
  const all = new Option("All tournament sections", "all");
  els.stageFilter.append(all);
  for (const stage of stages) els.stageFilter.append(new Option(stageDisplay(stage), stage));
}

function applyFilters() {
  const stage = els.stageFilter.value;
  const country = els.countryFilter.value;
  const highUpsetOnly = els.upsetToggle.checked;

  state.filtered = state.matches.filter((match) => {
    const stageOk = stage === "all" || match.stage === stage;
    const countryOk =
      country === "all" ||
      normalizeTeamKey(match.home_team) === normalizeTeamKey(country) ||
      normalizeTeamKey(match.away_team) === normalizeTeamKey(country);
    const upsetOk = !highUpsetOnly || Number(match.upset_risk) >= 0.5;
    return stageOk && countryOk && upsetOk;
  });

  if (!state.filtered.length) {
    state.filtered = state.matches.slice(0, 1);
  }

  if (!state.filtered.some((match) => String(match.match_id) === String(state.selectedId))) {
    state.selectedId = state.filtered[0].match_id;
  }

  buildMatchSelect();
  renderLineup();
  renderLineupDisclosure();
  render();
}

function buildMatchSelect() {
  els.matchSelect.replaceChildren();
  for (const match of state.filtered) {
    const score = match.score ? `, ${match.score}` : "";
    const label = `${match.date} - ${match.home_team} vs ${match.away_team} (${stageDisplay(match.stage)}${score})`;
    els.matchSelect.append(new Option(label, String(match.match_id)));
  }
  els.matchSelect.value = String(state.selectedId);
}

function renderSummary() {
  const modelSummary = state.data.model_summary || {};
  const validation = state.data.validation_summary || {};
  const audit = state.data.prediction_audit || {};
  els.matchCount.textContent = String(state.matches.length);
  els.modelLoss.textContent = modelSummary.best_log_loss_B ? Number(modelSummary.best_log_loss_B).toFixed(4) : "--";
  els.validationAcc.textContent = audit.draw_adjusted_score ? pct(audit.draw_adjusted_score, 1) : audit.accuracy ? pct(audit.accuracy, 1) : validation.accuracy ? pct(validation.accuracy, 1) : "--";
  els.dataCaveat.textContent = state.data.caveat || "";
}

function renderAuditOverview() {
  const audit = state.data.prediction_audit || {};
  if (!audit.focus_rows) return;

  els.auditSummary.textContent =
    audit.summary_line ||
    "Train on historical World Cup matches, then audit 2026 predictions against actual results as they arrive.";

  const coverage = `${audit.prediction_available_rows || 0}/${audit.focus_rows || 0}`;
  const actualized = `${audit.actual_available_rows || 0}`;
  const pending = audit.pending_actual_rows || 0;
  const adjustedScore = audit.draw_adjusted_score === null || audit.draw_adjusted_score === undefined ? "--" : pct(audit.draw_adjusted_score, 1);
  const exactAccuracy = audit.strict_exact_accuracy === null || audit.strict_exact_accuracy === undefined ? "--" : pct(audit.strict_exact_accuracy, 1);
  const trainingDetail = `${audit.training_window || "historical"} · cutoff ${audit.training_cutoff_date || "unknown"}`;

  const cards = [
    {
      value: String(audit.training_match_count || "--"),
      label: `Training rows · ${trainingDetail}`,
    },
    {
      value: coverage,
      label: "Announced 2026 fixtures with a model forecast",
    },
    {
      value: actualized,
      label: `${pending} actual result${pending === 1 ? "" : "s"} still pending`,
    },
    {
      value: adjustedScore,
      label: `Draw-adjusted score on ${audit.evaluated_rows || 0} actualized rows`,
    },
    {
      value: exactAccuracy,
      label: `Strict exact H/D/A accuracy; ${audit.draw_push_rows || 0} draws are partial-credit rows`,
    },
  ];

  els.auditStatGrid.innerHTML = cards
    .map(
      (card) => `
        <div class="audit-stat">
          <strong>${card.value}</strong>
          <span>${card.label}</span>
        </div>
      `,
    )
    .join("");

  const roleOrder = [
    "retrospective_2026_holdout_validation",
    "scheduled_actualized_validation",
    "live_pre_match_forecast_pending_actual",
    "post_kickoff_forecast_pending_actual",
    "announced_fixture_pending_actual",
  ];
  const roleCounts = audit.audit_role_counts || {};
  const roleRows = roleOrder
    .filter((role) => roleCounts[role])
    .map((role) => [auditRoleLabel(role), roleCounts[role]]);

  const cv = audit.historical_cv || {};
  const cvLine =
    cv.evaluated_match_count && cv.holdout_year_count
      ? `${cv.evaluated_match_count} World Cup fold-evaluations across ${cv.holdout_year_count} held-out tournaments`
      : "Historical held-out tournament testing";

  els.auditStateList.innerHTML = [
    [cvLine, cv.log_loss ? `RF log-loss ${Number(cv.log_loss).toFixed(4)}` : ""],
    ...roleRows,
  ]
    .map(
      ([label, value]) => `
        <div class="audit-state-row">
          <span>${label}</span>
          <strong>${value}</strong>
        </div>
      `,
    )
    .join("");

  els.auditTimestamp.textContent = `Prediction timestamps: ${shortDateTime(
    audit.predicted_at_min_utc,
  )} to ${shortDateTime(audit.predicted_at_max_utc)}. Fixture and actual-status layer as of ${
    audit.as_of_local || audit.as_of_utc || "unknown"
  }.`;
}

function renderLineup() {
  const scheduleSummary = state.data.schedule_summary || {};
  const vizSummary = state.data.viz_summary || {};
  const audit = state.data.prediction_audit || {};
  const asOf = asOfLabel(scheduleSummary);
  const rows = state.filtered.slice().sort(chronologySort);
  const allActualized = state.matches.filter(isActualized).length;
  const allPending = state.matches.length - allActualized;
  const announcedSlots = vizSummary.rows || audit.focus_rows || state.matches.length;
  const forecastedSlots = vizSummary.prediction_available_rows || audit.prediction_available_rows || state.matches.length;
  const unresolvedSlots = Math.max(0, announcedSlots - forecastedSlots);
  const exactCorrectRows = audit.correct_rows || state.matches.filter((match) => modelGrade(match) === "exact_correct").length;
  const drawPushRows = audit.draw_push_rows || state.matches.filter((match) => modelGrade(match) === "draw_push").length;
  const incorrectRows = audit.incorrect_rows || state.matches.filter((match) => modelGrade(match) === "incorrect").length;
  const adjusted = audit.draw_adjusted_score === null || audit.draw_adjusted_score === undefined ? "--" : pct(audit.draw_adjusted_score, 1);
  const exact = audit.strict_exact_accuracy === null || audit.strict_exact_accuracy === undefined ? "--" : pct(audit.strict_exact_accuracy, 1);

  els.lineupStatus.textContent = announcedSlots
    ? `${forecastedSlots}/${announcedSlots} announced fixture slots forecasted. ${unresolvedSlots} unresolved bracket slots are waiting for winners. ${allActualized} actualized, ${allPending} forecasted pending. Adjusted score ${adjusted}; strict exact ${exact}. Snapshot ${asOf}.`
    : `${state.matches.length} forecast rows loaded. Snapshot ${asOf}.`;

  els.lineupBreakdown.innerHTML = `
    <div class="lineup-breakdown-card breakdown-correct">
      <strong>${exactCorrectRows}</strong>
      <span>Model correct</span>
    </div>
    <div class="lineup-breakdown-card breakdown-draw-push">
      <strong>${drawPushRows}</strong>
      <span>Actual draw / partial credit</span>
    </div>
    <div class="lineup-breakdown-card breakdown-incorrect">
      <strong>${incorrectRows}</strong>
      <span>Model incorrect</span>
    </div>
    <div class="lineup-breakdown-card breakdown-pending">
      <strong>${allPending}</strong>
      <span>Forecasted pending actual</span>
    </div>
    <div class="lineup-breakdown-card breakdown-unresolved">
      <strong>${unresolvedSlots}</strong>
      <span>Unresolved bracket slots not forecasted</span>
    </div>
    <div class="lineup-breakdown-card breakdown-accuracy">
      <strong>${adjusted}</strong>
      <span>Draw-adjusted score</span>
    </div>
  `;

  els.lineupEmpty.hidden = rows.length > 0;
  if (!rows.length) {
    els.timelineList.innerHTML = "";
    return;
  }

  const groups = new Map();
  for (const match of rows) {
    const date = match.date || "Date missing";
    if (!groups.has(date)) groups.set(date, []);
    groups.get(date).push(match);
  }

  els.timelineList.innerHTML = [...groups.entries()]
    .map(([date, matches]) => {
      const dateActualized = matches.filter(isActualized).length;
      const datePending = matches.length - dateActualized;
      const cards = matches
        .sort(chronologySort)
        .map((match) => {
          const actualized = isActualized(match);
          const grade = modelGrade(match);
          const selected = String(match.match_id) === String(state.selectedId);
          const status = matchStatusLabel(match);
          const prediction = `${predictedTeam(match)} ${pct(predictionProbability(match))}`;
          const result = match.score ? `Actual ${match.score}` : "Actual pending";
          const archetype = match.archetypes?.[0] || "No active archetype";
          return `
            <button
              class="timeline-card ${timelineGradeClass(match)} ${selected ? "timeline-card-selected" : ""}"
              type="button"
              data-match-id="${match.match_id}"
            >
              <span class="timeline-card-top">
                <span>${match.kickoff_time_local || "time TBD"}</span>
                <b>${actualized ? gradeLabel(match) : "Pending"}</b>
              </span>
              <strong>${match.home_team} vs ${match.away_team}</strong>
              <span class="timeline-stage">${stageDisplay(match.stage)}</span>
              <span class="timeline-forecast">Forecast: ${prediction}</span>
              <span class="timeline-result">${result} · ${status}</span>
              <span class="timeline-tags">${archetype} · Upset risk ${pct(match.upset_risk)}</span>
            </button>
          `;
        })
        .join("");
      return `
        <section class="timeline-day">
          <div class="timeline-day-head">
            <strong>${date}</strong>
            <span>${dateActualized} happened · ${datePending} pending</span>
          </div>
          <div class="timeline-day-grid">${cards}</div>
        </section>
      `;
    })
    .join("");
}

function renderLineupDisclosure() {
  els.lineupBody.hidden = !state.lineupExpanded;
  els.lineupToggle.setAttribute("aria-expanded", String(state.lineupExpanded));
  els.lineupToggle.textContent = state.lineupExpanded ? "Collapse lineup" : "Expand lineup";
  document.querySelector("#event-lineup")?.classList.toggle("lineup-panel-collapsed", !state.lineupExpanded);
}

function render() {
  const match = selectedMatch();
  if (!match) return;

  state.selectedId = match.match_id;
  els.matchSelect.value = String(match.match_id);

  const home = teamColor(match.home_team, "home");
  const away = teamColor(match.away_team, "away");
  document.documentElement.style.setProperty("--home", home);
  document.documentElement.style.setProperty("--away", away);

  setTeamFlag(els.homeCode, match.home_team);
  setTeamFlag(els.awayCode, match.away_team);
  setTeamName(els.homeTeam, match.home_team);
  setTeamName(els.awayTeam, match.away_team);
  els.legendHome.textContent = match.home_team;
  els.legendAway.textContent = match.away_team;
  els.legendHome.style.color = home;
  els.legendAway.style.color = away;
  els.matchStage.textContent = stageDisplay(match.stage);
  els.matchDate.textContent = match.kickoff_time_local ? `${match.date} · ${match.kickoff_time_local}` : match.date;
  els.matchScore.textContent = match.score ? `Actual: ${match.score}` : "Actual: pending; forecast shown";

  renderBadges(match);
  renderProbabilities(match);
  renderPrediction(match);
  renderReasoning(match);
  renderMetrics(match);
  renderImportance();
  renderSimilar(match);
  updateLineupSelection();
  drawRadar(match);
}

function updateLineupSelection() {
  if (!els.timelineList) return;
  for (const card of els.timelineList.querySelectorAll(".timeline-card")) {
    card.classList.toggle("timeline-card-selected", String(card.dataset.matchId) === String(state.selectedId));
  }
}

function renderBadges(match) {
  els.badgeList.replaceChildren();
  const labels = match.archetypes && match.archetypes.length ? match.archetypes : ["No Active Archetype"];
  for (const label of labels) {
    const badge = document.createElement("span");
    badge.className = `badge badge-${archetypeClass.get(label) || "tactical_contrast"}`;
    badge.textContent = label;
    els.badgeList.append(badge);
  }

  let snapshotType = match.pre_match_snapshot ? "true pre-match snapshot" : "retrospective validation row";
  if (isScheduledForecast(match)) {
    if (match.prediction_scope === "pre_match_forecast") snapshotType = "scheduled fixture forecast made before kickoff";
    else if (match.prediction_scope === "post_kickoff_pending_result") snapshotType = "scheduled fixture forecast made after kickoff while the result was not ingested";
    else if (match.prediction_scope === "result_available_retrospective") snapshotType = "retrospective scheduled-fixture forecast after a result was already visible";
    else snapshotType = "scheduled fixture forecast outside the main validation layer";
  }
  const actual = match.score ? `${match.result_label}, ${match.score}` : "actual result pending";
  const imputed = match.is_imputed_prediction ? " The displayed outcome is forecast-only until the actual result is ingested." : "";
  els.snapshotNote.textContent = `Prediction saved at ${match.predicted_at_utc}. This row is a ${snapshotType}; actual result: ${actual}.${imputed}`;
}

function renderProbabilities(match) {
  const items = [
    { key: "home", label: match.home_team, value: match.probabilities.home, color: "var(--win)" },
    { key: "draw", label: "Draw", value: match.probabilities.draw, color: "var(--draw)" },
    { key: "away", label: match.away_team, value: match.probabilities.away, color: "var(--loss)" },
  ];

  els.probBars.replaceChildren();
  for (const item of items) {
    const cell = document.createElement("div");
    cell.className = "prob-bar-cell";
    cell.innerHTML = `
      <div class="prob-value">${pct(item.value)}</div>
      <div class="prob-track"><div class="prob-fill" style="height:${Math.max(4, item.value * 100)}%; background:${item.color}"></div></div>
      <div class="prob-label">${item.label}</div>
    `;
    els.probBars.append(cell);
  }

  els.probTable.innerHTML = `
    <div><strong style="color:var(--win)">${pct(match.probabilities.home)}</strong><small>${match.home_team}</small></div>
    <div><strong style="color:var(--draw)">${pct(match.probabilities.draw)}</strong><small>Draw</small></div>
    <div><strong style="color:var(--loss)">${pct(match.probabilities.away)}</strong><small>${match.away_team}</small></div>
  `;
  els.upsetPill.textContent = `Upset risk: ${pct(match.upset_risk)}`;
}

function renderPrediction(match) {
  const predicted = predictedTeam(match);
  const probability = predictionProbability(match);
  const side = predictedSide(match);
  const actual = match.score ? `${match.result_label} (${match.score})` : "pending";
  const upset = match.actual_upset ? " Actual result is marked as an upset." : "";
  const grade = modelGrade(match);
  const correctness =
    grade === "exact_correct"
      ? " Correct call."
      : grade === "draw_push"
        ? " Draw partial credit, not a full miss."
        : grade === "incorrect"
          ? " Missed call."
          : "";
  const pending = match.is_imputed_prediction ? " Forecast is standing in until the actual arrives." : "";
  const drawPressure = match.probabilities?.draw;
  const favoriteLine =
    side === "draw"
      ? `Most likely result: draw`
      : `Expected winner: ${predicted}`;
  const confidenceLine =
    side === "unknown"
      ? "Model confidence unavailable"
      : `${pct(probability)} model confidence · draw pressure ${pct(drawPressure)} · upset risk ${pct(match.upset_risk)}`;
  const path = pathToWinInsight(match);

  els.predictionCall.innerHTML = `
    <div class="prediction-winner">${escapeHtml(favoriteLine)}</div>
    <div class="prediction-confidence">${escapeHtml(confidenceLine)}</div>
    <div class="prediction-path"><span>Path to win</span>${escapeHtml(path)}</div>
    <div class="prediction-audit-line">Actual: ${escapeHtml(actual)}.${escapeHtml(correctness)}${escapeHtml(pending)}${escapeHtml(upset)}</div>
  `;
}

function pathToWinInsight(match) {
  const side = predictedSide(match);
  const winner = predictedTeam(match);
  const features = match.features || {};
  const eloGap = Number(features.elo_gap_abs);
  const upsetRisk = Number(match.upset_risk);
  const archetypes = match.archetypes || [];
  const notes = [];

  if (side === "draw") {
    return "Keep the game low-event, avoid transition exposure, and force the opponent into low-quality chances.";
  }

  if (!Number.isNaN(eloGap) && eloGap >= 120) {
    notes.push("turn the rating edge into early chance volume");
  } else {
    notes.push("win the first-goal and game-state battle");
  }

  if (!Number.isNaN(upsetRisk) && upsetRisk >= 0.45) {
    notes.push("manage upset volatility instead of letting the match become chaotic");
  } else {
    notes.push("protect the forecast edge with controlled possession and shot quality");
  }

  if (archetypes.includes("Host Pressure")) {
    notes.push("handle crowd and host-pressure swings without chasing the game");
  }
  if (archetypes.includes("Knockout Volatility")) {
    notes.push("avoid penalty-box mistakes and late-game variance");
  }

  return `${winner} profile: ${notes.slice(0, 3).join("; ")}.`;
}

function renderReasoning(match) {
  const prediction = `${predictedTeam(match)} at ${pct(predictionProbability(match))}`;
  const timing = `${timingLabel(match)} · generated ${shortDateTime(match.predicted_at_utc)}`;
  const status = matchStatusLabel(match);
  const archetypes = match.archetypes?.length ? match.archetypes.join(", ") : "No active archetype";
  const localFeatures = localFeatureSummary(match);
  const source = match.prediction_layer === "scheduled_fixture_forecast" ? "Scheduled fixture forecast layer" : "Main 2026 validation layer";
  const rows = [
    ["Prediction", prediction],
    ["Grade", `${gradeLabel(match)}${match.prediction_credit !== null && match.prediction_credit !== undefined ? ` · credit ${number(match.prediction_credit, 1)}` : ""}`],
    ["Timing", timing],
    ["Status", status],
    ["Archetypes", archetypes],
    ["Readable factors", localFeatures],
    ["Layer", source],
  ];

  els.reasoningList.innerHTML = rows
    .map(
      ([label, value]) => `
        <div class="reasoning-row">
          <span>${label}</span>
          <strong>${value}</strong>
        </div>
      `,
    )
    .join("");
}

function renderMetrics(match) {
  const cluster = match.cluster || {};
  const metrics = [
    ["Prediction generated", shortDateTime(match.predicted_at_utc)],
    ["Prediction timing", timingLabel(match)],
    ["Model grade", gradeLabel(match)],
    ["Audit role", auditRoleLabel(match.audit_role)],
    ["Actual status", matchStatusLabel(match)],
    ["Venue", match.venue || "Venue unavailable"],
    ["Training cutoff", match.training_cutoff_date || "--"],
    ["Home Elo", number(match.features.elo_home_pre, 0)],
    ["Away Elo", number(match.features.elo_away_pre, 0)],
    ["Absolute Elo gap", number(match.features.elo_gap_abs, 1)],
    ["Model home win prior", pct(match.features.win_prob_home, 1)],
    ["Age mean gap", number(match.features.age_mean_gap, 2)],
    ["League diversity gap", number(match.features.league_div_gap, 2)],
    ["K-means cluster", cluster.kmeans_k5 === null ? "--" : `Cluster ${cluster.kmeans_k5}`],
  ];

  els.metricList.innerHTML = metrics
    .map(([label, value]) => `<div class="metric-row"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

function renderImportance() {
  const rows = state.data.feature_importance || [];
  const max = Math.max(...rows.map((row) => Number(row.importance) || 0), 0.001);
  els.importanceList.innerHTML = rows
    .map((row) => {
      const width = ((Number(row.importance) || 0) / max) * 100;
      return `
        <div class="importance-row">
          <span title="${row.feature}">${cleanLabel(row.feature)}</span>
          <div class="importance-track"><div class="importance-fill" style="width:${width}%"></div></div>
          <strong>${Number(row.importance).toFixed(3)}</strong>
        </div>
      `;
    })
    .join("");
}

function renderSimilar(match) {
  els.similarGrid.replaceChildren();
  for (const item of match.similar || []) {
    const card = document.createElement("article");
    card.className = "similar-card";
    const arch = item.archetypes && item.archetypes.length ? item.archetypes[0] : "World Cup";
    card.innerHTML = `
      <h3>${item.home_team} vs ${item.away_team}</h3>
      <div class="mini-score">
        <span class="mini-team">${item.home_team}</span>
        <span class="mini-result">${item.score || item.result}</span>
        <span class="mini-team">${item.away_team}</span>
      </div>
      <div class="similar-meta">${item.year} · ${stageDisplay(item.stage)}<br>${item.result}</div>
      <span class="similar-chip">${Math.round((item.similarity || 0) * 100)}% similar - ${arch}</span>
    `;
    els.similarGrid.append(card);
  }
}

function drawRadar(match) {
  const canvas = els.radar;
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(320, Math.round(rect.width * ratio));
  canvas.height = Math.max(280, Math.round(rect.height * ratio));
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);

  const axes = match.radar.axes;
  const home = match.radar.home;
  const away = match.radar.away;
  const center = { x: width / 2, y: height / 2 + 10 };
  const radius = Math.max(86, Math.min(width, height) * 0.32);
  const sides = axes.length;

  ctx.lineWidth = 1;
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  for (let ring = 1; ring <= 5; ring += 1) {
    const points = polygonPoints(sides, center, (radius * ring) / 5);
    drawPath(ctx, points, false);
    ctx.strokeStyle = "rgba(190, 206, 220, 0.22)";
    ctx.stroke();
  }

  for (let i = 0; i < sides; i += 1) {
    const angle = angleFor(i, sides);
    const end = pointAt(center, radius, angle);
    ctx.beginPath();
    ctx.moveTo(center.x, center.y);
    ctx.lineTo(end.x, end.y);
    ctx.strokeStyle = "rgba(190, 206, 220, 0.22)";
    ctx.stroke();

    const labelPoint = pointAt(center, radius + 34, angle);
    const label = axes[i];
    ctx.fillStyle = "rgba(242, 245, 247, 0.92)";
    wrapCanvasLabel(ctx, label, labelPoint.x, labelPoint.y, 88, 15);
  }

  const styles = getComputedStyle(document.documentElement);
  const homeColor = styles.getPropertyValue("--home").trim() || "#4f86e8";
  const awayColor = styles.getPropertyValue("--away").trim() || "#f0a33a";
  drawRadarShape(ctx, home, sides, center, radius, homeColor, 0.26);
  drawRadarShape(ctx, away, sides, center, radius, awayColor, 0.24);
}

function angleFor(index, sides) {
  return -Math.PI / 2 + (index * Math.PI * 2) / sides;
}

function pointAt(center, radius, angle) {
  return {
    x: center.x + Math.cos(angle) * radius,
    y: center.y + Math.sin(angle) * radius,
  };
}

function polygonPoints(sides, center, radius) {
  return Array.from({ length: sides }, (_, i) => pointAt(center, radius, angleFor(i, sides)));
}

function drawPath(ctx, points, close = true) {
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
  if (close) ctx.closePath();
}

function drawRadarShape(ctx, values, sides, center, radius, color, fillAlpha) {
  const points = values.map((value, index) => pointAt(center, radius * Number(value || 0), angleFor(index, sides)));
  drawPath(ctx, points);
  ctx.save();
  ctx.globalAlpha = fillAlpha;
  ctx.fillStyle = color;
  ctx.fill();
  ctx.restore();

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();

  for (const point of points) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }
}

function wrapCanvasLabel(ctx, text, x, y, maxWidth, lineHeight) {
  const words = text.split(" ");
  const lines = [];
  let line = "";
  for (const word of words) {
    const test = line ? `${line} ${word}` : word;
    if (ctx.measureText(test).width > maxWidth && line) {
      lines.push(line);
      line = word;
    } else {
      line = test;
    }
  }
  lines.push(line);
  const start = y - ((lines.length - 1) * lineHeight) / 2;
  lines.forEach((part, index) => ctx.fillText(part, x, start + index * lineHeight));
}

function setActiveSectionNav(sectionId) {
  for (const link of els.sectionNavLinks) {
    const active = link.dataset.sectionTarget === sectionId;
    link.classList.toggle("section-nav-active", active);
    if (active) link.setAttribute("aria-current", "true");
    else link.removeAttribute("aria-current");
  }
}

function setupSectionNav() {
  if (!els.sectionNavLinks.length) return;
  const nav = document.querySelector(".section-nav");
  const sections = [...els.sectionNavLinks]
    .map((link) => document.querySelector(`#${link.dataset.sectionTarget}`))
    .filter(Boolean);

  setActiveSectionNav(els.sectionNavLinks[0].dataset.sectionTarget);

  for (const link of els.sectionNavLinks) {
    link.addEventListener("click", () => setActiveSectionNav(link.dataset.sectionTarget));
  }

  if (!sections.length) return;
  let ticking = false;
  const updateFromScroll = () => {
    ticking = false;
    const navBottom = nav?.getBoundingClientRect().bottom || 0;
    const checkpoint = Math.min(window.innerHeight * 0.36, Math.max(180, navBottom + 32));
    let activeId = sections[0].id;
    for (const section of sections) {
      const rect = section.getBoundingClientRect();
      if (rect.top <= checkpoint && rect.bottom > checkpoint) {
        activeId = section.id;
        break;
      }
      if (rect.top <= checkpoint) activeId = section.id;
    }
    setActiveSectionNav(activeId);
  };
  const requestUpdate = () => {
    if (ticking) return;
    ticking = true;
    window.requestAnimationFrame(updateFromScroll);
  };

  window.addEventListener("scroll", requestUpdate, { passive: true });
  window.addEventListener("resize", requestUpdate);
  updateFromScroll();
}

async function init() {
  const response = await fetch(DATA_URL, { cache: "no-store" });
  if (!response.ok) throw new Error(`Unable to load ${DATA_URL}`);
  state.data = await response.json();
  state.matches = (state.data.matches || []).slice().sort(chronologySort);
  state.selectedId = state.matches[0]?.match_id ?? null;

  buildCountryFilter();
  renderSummary();
  renderAuditOverview();
  buildStageFilter();
  applyFilters();
  setupSectionNav();

  els.stageFilter.addEventListener("change", applyFilters);
  els.countryFilter.addEventListener("change", applyFilters);
  els.resetCountry.addEventListener("click", () => {
    els.countryFilter.value = "all";
    applyFilters();
    els.countryFilter.focus();
  });
  els.lineupToggle.addEventListener("click", () => {
    state.lineupExpanded = !state.lineupExpanded;
    renderLineupDisclosure();
  });
  els.upsetToggle.addEventListener("change", applyFilters);
  els.matchSelect.addEventListener("change", (event) => {
    state.selectedId = event.target.value;
    render();
  });
  els.timelineList.addEventListener("click", (event) => {
    const card = event.target.closest(".timeline-card");
    if (!card) return;
    state.selectedId = card.dataset.matchId;
    els.matchSelect.value = String(state.selectedId);
    render();
    document.querySelector("#match-specific-stats")?.scrollIntoView({ behavior: "smooth", block: "start" });
    setActiveSectionNav("match-specific-stats");
  });
  window.addEventListener("resize", () => drawRadar(selectedMatch()));

}

init().catch((error) => {
  console.error(error);
  document.body.innerHTML = `<main class="app-shell"><article class="panel match-panel"><h1>Unable to load matchup data</h1><p>${error.message}</p></article></main>`;
});
