const DATA_URL = "./data/matchups.json";

const els = {
  matchCount: document.querySelector("#match-count"),
  modelLoss: document.querySelector("#model-loss"),
  validationAcc: document.querySelector("#validation-acc"),
  filterSummary: document.querySelector("#filter-summary"),
  filterStatusLine: document.querySelector("#filter-status-line"),
  activeFilterRow: document.querySelector("#active-filter-row"),
  contextTitle: document.querySelector("#context-title"),
  contextCopy: document.querySelector("#context-copy"),
  contextPillRow: document.querySelector("#context-pill-row"),
  viewChipRow: document.querySelector("#view-chip-row"),
  countryChipRow: document.querySelector("#country-chip-row"),
  stageFilter: document.querySelector("#stage-filter"),
  matchSelect: document.querySelector("#match-select"),
  countryFilter: document.querySelector("#country-filter"),
  resetCountry: document.querySelector("#reset-country"),
  upsetToggle: document.querySelector("#upset-toggle"),
  auditSummary: document.querySelector("#audit-summary"),
  auditViewToggle: document.querySelector("#audit-view-toggle"),
  auditVisual: document.querySelector("#audit-visual"),
  auditGrid: document.querySelector("#audit-grid"),
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
  archetypePitch: document.querySelector("#archetype-pitch"),
  archetypeCommentary: document.querySelector("#archetype-commentary"),
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
  auditView: "visual",
  statusFilter: "all",
  lineupExpanded: false,
  activeArchetypeMode: "default",
  activeArchetypes: [],
  activeArchetypeMatchId: null,
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

async function loadMatchupData() {
  if (window.MATCHUP_DATA && window.location.protocol === "file:") {
    return window.MATCHUP_DATA;
  }

  try {
    const cacheBust = `?t=${Math.floor(Date.now() / 60000)}`; // new value every 60s
    const response = await fetch(DATA_URL + cacheBust, { cache: "no-store" });
    if (!response.ok) throw new Error(`Unable to load ${DATA_URL}`);

    return await response.json();
  } catch (error) {
    if (window.MATCHUP_DATA) return window.MATCHUP_DATA;
    throw error;
  }
}

function renderLoadError(error) {
  const shell = document.querySelector(".app-shell") || document.body;
  const panel = document.createElement("article");
  panel.className = "panel load-error-panel";
  panel.innerHTML = `
    <div class="panel-kicker">Data Load Issue</div>
    <h2>Unable to load matchup data</h2>
    <p>${escapeHtml(error.message || "The matchup dataset could not be loaded.")}</p>
    <p>For local development, run the dashboard from the project folder with <code>python -m http.server 8877</code>, then open <code>http://127.0.0.1:8877/web/matchup-card/</code>.</p>
  `;
  shell.prepend(panel);
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
  if (length >= 26) return "0.96rem";
  if (length >= 22) return "1.06rem";
  if (length >= 18) return "1.18rem";
  if (length >= 14) return "1.36rem";
  return "clamp(1.3rem, 2vw, 1.85rem)";
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
  if (letter) return `Group stage · Group ${letter}`;
  if (!stage || stage === "Unassigned") return "Group stage · group label missing";
  const normalized = String(stage).toLowerCase();
  if (normalized === "group stage") return "Group stage";
  if (normalized === "second group stage") return "Second group stage";
  if (normalized.includes("third-place")) return "Third-place match";
  if (/round of 32/i.test(stage)) return "Knockout match · Round of 32";
  if (/round of 16/i.test(stage)) return "Knockout match · Round of 16";
  if (/quarter/i.test(stage)) return "Knockout match · Quarterfinal";
  if (/semi/i.test(stage)) return "Knockout match · Semifinal";
  if (/final/i.test(stage)) return "Final";
  return String(stage);
}

function isOpeningStage(stage) {
  const normalized = String(stage || "").toLowerCase();
  return Boolean(sectionLetter(stage)) || normalized === "group stage" || normalized === "second group stage";
}

function stageFilterDisplay(value) {
  if (!value || value === "all") return "All phases";
  if (value === "stage_group:opening") return "Group-stage matches";
  if (value === "stage_group:knockout") return "Knockout matches";
  return stageDisplay(value);
}

function stageFilterMatches(match, value) {
  if (!value || value === "all") return true;
  if (value === "stage_group:opening") return isOpeningStage(match.stage);
  if (value === "stage_group:knockout") return !isOpeningStage(match.stage);
  return match.stage === value;
}

function stageSortValue(stage) {
  const letter = sectionLetter(stage);
  if (letter) return letter.charCodeAt(0);
  if (!stage || stage === "Unassigned") return 900;
  if (/round of 32/i.test(stage)) return 210;
  if (/round of 16/i.test(stage)) return 220;
  if (/quarter/i.test(stage)) return 230;
  if (/semi/i.test(stage)) return 240;
  if (/third-place/i.test(stage)) return 250;
  if (/final/i.test(stage)) return 260;
  return 500 + String(stage).localeCompare("zzzz");
}

function shortDate(match) {
  if (!match?.date) return "date TBD";
  const [year, month, day] = String(match.date).split("-").map(Number);
  if (!year || !month || !day) return match.date;
  return new Intl.DateTimeFormat("en", { month: "short", day: "numeric" }).format(new Date(year, month - 1, day));
}

function matchupLabel(match, country = null) {
  if (!match) return "No loaded matchup";
  if (country) return `vs ${opponentFor(match, country)} (${shortDate(match)})`;
  return `${match.home_team} vs ${match.away_team} (${shortDate(match)})`;
}

function opponentFor(match, country) {
  if (normalizeTeamKey(match.home_team) === normalizeTeamKey(country)) return match.away_team;
  if (normalizeTeamKey(match.away_team) === normalizeTeamKey(country)) return match.home_team;
  return `${match.home_team} / ${match.away_team}`;
}

function rowsForCountry(country) {
  return state.matches
    .filter((match) => normalizeTeamKey(match.home_team) === normalizeTeamKey(country) || normalizeTeamKey(match.away_team) === normalizeTeamKey(country))
    .sort(chronologySort);
}

function groupRowsForCountry(country) {
  return rowsForCountry(country).filter((match) => isOpeningStage(match.stage));
}

function groupLabelForCountry(country) {
  const groupStage = groupRowsForCountry(country)[0]?.stage;
  const letter = sectionLetter(groupStage);
  return letter ? `Group ${letter}` : "Group not loaded";
}

function groupOpponentsForCountry(country) {
  return [
    ...new Set(
      groupRowsForCountry(country)
        .map((match) => opponentFor(match, country))
        .filter(Boolean),
    ),
  ];
}

function nextMatchForCountry(country) {
  return rowsForCountry(country).find((match) => !isActualized(match)) || null;
}

function lastResultForCountry(country) {
  return rowsForCountry(country)
    .filter(isActualized)
    .at(-1) || null;
}

function countryPathDetail(country) {
  const group = groupLabelForCountry(country);
  const next = nextMatchForCountry(country);
  const last = lastResultForCountry(country);
  if (next) return `${group} · next ${shortDate(next)}`;
  if (last) return `${group} · last ${shortDate(last)}`;
  return group;
}

function contextPill(label, value) {
  return `
    <span class="context-pill">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
    </span>
  `;
}

function renderTournamentContext(match = selectedMatch()) {
  const selectedCountry = els.countryFilter.value || "all";
  if (!els.contextTitle || !els.contextCopy || !els.contextPillRow) return;

  if (selectedCountry !== "all") {
    const rows = rowsForCountry(selectedCountry);
    const group = groupLabelForCountry(selectedCountry);
    const opponents = groupOpponentsForCountry(selectedCountry);
    const next = nextMatchForCountry(selectedCountry);
    const last = lastResultForCountry(selectedCountry);
    const selectedForCountry =
      match &&
      (normalizeTeamKey(match.home_team) === normalizeTeamKey(selectedCountry) ||
        normalizeTeamKey(match.away_team) === normalizeTeamKey(selectedCountry))
        ? match
        : next || last || rows[0];

    els.contextTitle.textContent = `${teamFlag(selectedCountry) ? `${teamFlag(selectedCountry)} ` : ""}${selectedCountry} tournament path`;
    els.contextCopy.textContent = next
      ? `${selectedCountry} is loaded in ${group}; the next loaded matchup is against ${opponentFor(next, selectedCountry)} on ${shortDate(next)}.`
      : `${selectedCountry} is loaded in ${group}; no upcoming fixture is currently loaded for this team.`;
    els.contextPillRow.innerHTML = [
      contextPill("Initial group", group),
      contextPill("Group opponents", opponents.join(", ") || "not loaded"),
      contextPill("Selected stage", selectedForCountry ? stageDisplay(selectedForCountry.stage) : "no selected stage"),
      contextPill("Last result", last ? `${matchupLabel(last, selectedCountry)} · ${last.score || "actualized"}` : "none loaded"),
      contextPill("Next matchup", next ? matchupLabel(next, selectedCountry) : "none loaded"),
    ].join("");
    return;
  }

  const nextGlobal = state.matches.filter((candidate) => !isActualized(candidate)).sort(chronologySort)[0];
  els.contextTitle.textContent = match ? `${stageDisplay(match.stage)} selected` : "Tournament structure snapshot";
  els.contextCopy.textContent =
    "The app starts with forecast status because it is the clearest search mode. Use country search to reveal group placement, selected round, last result, and next loaded matchup.";
  els.contextPillRow.innerHTML = [
    contextPill("Group stage", `${filterCount({ stage: "stage_group:opening" })} loaded matches`),
    contextPill("Knockout rows", `${filterCount({ stage: "stage_group:knockout" })} loaded matchups`),
    contextPill("Selected match", match ? `${match.home_team} vs ${match.away_team}` : "none"),
    contextPill("Next loaded", nextGlobal ? matchupLabel(nextGlobal) : "none loaded"),
  ].join("");
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

function chipButton({ active = false, count, countLabel, detail, label, kind, value }) {
  return `
    <button
      class="filter-chip${active ? " filter-chip-active" : ""}${Number(count) === 0 ? " filter-chip-zero" : ""}"
      type="button"
      data-filter-kind="${escapeHtml(kind)}"
      data-filter-value="${escapeHtml(value)}"
      aria-pressed="${String(active)}"
    >
      <span>${escapeHtml(label)}</span>
      ${count === undefined && countLabel === undefined ? "" : `<strong>${escapeHtml(countLabel ?? count)}</strong>`}
      ${detail ? `<small>${escapeHtml(detail)}</small>` : ""}
    </button>
  `;
}

function statusFilterMatches(match, value) {
  if (value === "upcoming") return !isActualized(match);
  if (value === "actualized") return isActualized(match);
  return true;
}

function filterRows({ country = "all", stage = "all", status = "all", upset = false } = {}) {
  return state.matches.filter((match) => {
    const countryOk =
      country === "all" ||
      normalizeTeamKey(match.home_team) === normalizeTeamKey(country) ||
      normalizeTeamKey(match.away_team) === normalizeTeamKey(country);
    const upsetOk = !upset || Number(match.upset_risk) >= 0.5;
    return stageFilterMatches(match, stage) && statusFilterMatches(match, status) && countryOk && upsetOk;
  });
}

function filterCount(filters = {}) {
  return filterRows(filters).length;
}

function matchCountLabel(count) {
  return `${count} match${count === 1 ? "" : "es"}`;
}

function officialSlotCount(stage) {
  if (/round of 32/i.test(stage)) return 16;
  if (/round of 16/i.test(stage)) return 8;
  if (/quarter/i.test(stage)) return 4;
  if (/semi/i.test(stage)) return 2;
  if (/final/i.test(stage)) return 1;
  return null;
}

function countLabelForView(value, count) {
  const officialSlots = officialSlotCount(value);
  return officialSlots ? `${count}/${officialSlots}` : String(count);
}

function stageChipDetail(stage, count) {
  if (stage === "stage_group:opening") return "group-stage matches";
  if (stage === "stage_group:knockout") return "loaded knockout rows";
  const officialSlots = officialSlotCount(stage);
  if (officialSlots && officialSlots !== count) return "loaded / official slots";
  if (officialSlots) return "official slots loaded";
  if (/quarter/i.test(stage)) return "8-country phase";
  if (/semi/i.test(stage)) return "4-country phase";
  if (/final/i.test(stage)) return "title match";
  return matchCountLabel(count);
}

function contextualCountryCounts({ stage = "all", status = "all", upset = false } = {}) {
  const counts = new Map();
  for (const match of filterRows({ stage, status, upset })) {
    for (const country of [match.home_team, match.away_team]) {
      if (!country) continue;
      counts.set(country, (counts.get(country) || 0) + 1);
    }
  }
  return counts;
}

function preferredCountrySuggestions(contextCounts, activeCountry = "all") {
  const featured = ["United States", "Brazil", "France", "England", "Canada", "Mexico", "Argentina", "Spain"];
  const byFrequency = [...state.countryCounts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([country]) => country);
  const selected = selectedMatch();
  return [
    activeCountry === "all" ? null : activeCountry,
    selected?.home_team,
    selected?.away_team,
    ...featured,
    ...[...contextCounts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([country]) => country),
    ...byFrequency,
  ]
    .filter((country, index, list) => country && state.countryCounts.has(country) && list.indexOf(country) === index)
    .slice(0, 12);
}

function updateCountryFilterLabels(contextCounts, currentViewCount) {
  for (const option of els.countryFilter.options) {
    if (option.value === "all") {
      option.textContent = "All countries";
      continue;
    }
    const country = option.value;
    const flag = teamFlag(country);
    option.textContent = `${flag ? `${flag} ` : ""}${country} - ${countryPathDetail(country)}`;
  }
}

function activeViewName(stageValue, statusValue, upsetActive) {
  if (stageValue === "all" && statusValue === "all" && !upsetActive) return "All 2026 matches";
  if (stageValue === "all" && statusValue === "upcoming" && !upsetActive) return "Upcoming forecasts";
  if (stageValue === "all" && statusValue === "actualized" && !upsetActive) return "Past results";
  if (stageValue !== "all" && statusValue === "all" && !upsetActive) return stageFilterDisplay(stageValue);
  if (stageValue === "all" && statusValue === "all" && upsetActive) return "High upset risk";
  const parts = [];
  if (stageValue !== "all") parts.push(stageFilterDisplay(stageValue));
  if (statusValue === "upcoming") parts.push("Upcoming");
  if (statusValue === "actualized") parts.push("Past results");
  if (upsetActive) parts.push("High upset risk");
  return parts.join(" + ");
}

function activeFilterPill(label, clearFilter) {
  return `
    <button class="active-filter-pill" type="button" data-clear-filter="${escapeHtml(clearFilter)}">
      <span>${escapeHtml(label)}</span>
      <strong aria-hidden="true">x</strong>
    </button>
  `;
}

function renderFilterSuggestions() {
  const stageValue = els.stageFilter.value || "all";
  const countryValue = els.countryFilter.value || "all";
  const statusValue = state.statusFilter;
  const upsetActive = els.upsetToggle.checked;
  const contextCounts = contextualCountryCounts({ stage: stageValue, status: statusValue, upset: upsetActive });
  const currentViewCount = filterCount({ stage: stageValue, status: statusValue, upset: upsetActive });
  const upcomingCount = filterCount({ status: "upcoming" });
  const actualizedCount = filterCount({ status: "actualized" });

  const viewChips = [
    {
      kind: "clear",
      value: "all",
      label: "All 2026",
      detail: "reset all filters",
      count: state.matches.length,
      countLabel: String(state.matches.length),
      active: stageValue === "all" && statusValue === "all" && !upsetActive,
    },
    {
      kind: "status",
      value: "upcoming",
      label: "Upcoming",
      detail: "no actual yet",
      count: upcomingCount,
      countLabel: String(upcomingCount),
      active: stageValue === "all" && statusValue === "upcoming" && !upsetActive,
    },
    {
      kind: "status",
      value: "actualized",
      label: "Past results",
      detail: "actual available",
      count: actualizedCount,
      countLabel: String(actualizedCount),
      active: stageValue === "all" && statusValue === "actualized" && !upsetActive,
    },
  ];

  els.viewChipRow.innerHTML = viewChips.map(chipButton).join("");
  updateCountryFilterLabels(contextCounts, currentViewCount);

  const countryChips = [
    {
      kind: "country",
      value: "all",
      label: "All countries",
      detail: "clear country",
      active: countryValue === "all",
    },
    ...preferredCountrySuggestions(contextCounts, countryValue).map((country) => ({
      kind: "country",
      value: country,
      label: `${teamFlag(country) ? `${teamFlag(country)} ` : ""}${country}`,
      detail: countryPathDetail(country),
      active: normalizeTeamKey(countryValue) === normalizeTeamKey(country),
    })),
  ];
  els.countryChipRow.innerHTML = countryChips.map(chipButton).join("");
  requestAnimationFrame(() => {
    els.viewChipRow.querySelector(".filter-chip-active")?.scrollIntoView({ block: "nearest", inline: "nearest" });
    els.countryChipRow.querySelector(".filter-chip-active")?.scrollIntoView({ block: "nearest", inline: "nearest" });
  });

  const activePills = [];
  if (stageValue !== "all") activePills.push(activeFilterPill(`Phase: ${stageFilterDisplay(stageValue)}`, "stage"));
  if (statusValue === "upcoming") activePills.push(activeFilterPill("Status: upcoming", "status"));
  if (statusValue === "actualized") activePills.push(activeFilterPill("Status: past results", "status"));
  if (countryValue !== "all") activePills.push(activeFilterPill(`Country: ${countryValue}`, "country"));
  if (upsetActive) activePills.push(activeFilterPill("High upset risk", "upset"));
  els.activeFilterRow.innerHTML = activePills.length ? activePills.join("") : `<span class="active-filter-none">No filters applied</span>`;

  els.filterSummary.textContent = `${state.filtered.length} match${state.filtered.length === 1 ? "" : "es"} shown`;
  const countryPrefix = countryValue === "all" ? "all countries" : countryValue;
  els.filterStatusLine.textContent = `${activeViewName(stageValue, statusValue, upsetActive)} · ${countryPrefix} · ${matchCountLabel(state.filtered.length)} shown`;
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

function auditMetricValue(value, fallback = 0) {
  const metric = Number(value);
  return value === null || value === undefined || Number.isNaN(metric) ? fallback : metric;
}

function auditMetricLabel(value) {
  return value === null || value === undefined || Number.isNaN(Number(value)) ? "--" : pct(value, 1);
}

function auditCounts(audit) {
  const actualizedFallback = state.matches.filter(isActualized).length;
  const correctFallback = state.matches.filter((match) => modelGrade(match) === "exact_correct").length;
  const drawPushFallback = state.matches.filter((match) => modelGrade(match) === "draw_push").length;
  const incorrectFallback = state.matches.filter((match) => modelGrade(match) === "incorrect").length;
  const focus = audit.focus_rows ?? state.matches.length;
  const forecasted = audit.prediction_available_rows ?? state.matches.length;
  const actualized = audit.actual_available_rows ?? actualizedFallback;
  const pending = audit.pending_actual_rows ?? Math.max(0, state.matches.length - actualizedFallback);
  const correct = audit.correct_rows ?? correctFallback;
  const drawPush = audit.draw_push_rows ?? drawPushFallback;
  const incorrect = audit.incorrect_rows ?? incorrectFallback;
  const evaluated = audit.evaluated_rows ?? correct + drawPush + incorrect;
  const unresolved = Math.max(0, focus - forecasted);

  return { actualized, correct, drawPush, evaluated, focus, forecasted, incorrect, pending, unresolved };
}

function auditSegmentButton(segment, total) {
  const width = total ? (segment.value / total) * 100 : 0;
  return `
    <button
      class="audit-stack-segment audit-stack-${segment.key}${segment.value ? "" : " audit-stack-empty"}"
      type="button"
      style="--segment-width: ${width.toFixed(2)}%;"
      data-audit-detail="${escapeHtml(segment.detail)}"
      aria-label="${escapeHtml(`${segment.label}: ${segment.value}`)}"
      title="${escapeHtml(segment.detail)}"
    >
      <span>${segment.value}</span>
    </button>
  `;
}

function auditBarButton(segment, maxValue) {
  const height = maxValue ? Math.max(8, (segment.value / maxValue) * 100) : 8;
  return `
    <button
      class="audit-histogram-bar audit-histogram-${segment.key}"
      type="button"
      style="--bar-height: ${height.toFixed(2)}%;"
      data-audit-detail="${escapeHtml(segment.detail)}"
      title="${escapeHtml(segment.detail)}"
    >
      <span>${segment.value}</span>
      <small>${segment.shortLabel}</small>
    </button>
  `;
}

function renderAuditVisual(audit) {
  const counts = auditCounts(audit);
  const adjusted = auditMetricValue(audit.draw_adjusted_score);
  const adjustedLabel = auditMetricLabel(audit.draw_adjusted_score);
  const exactLabel = auditMetricLabel(audit.strict_exact_accuracy);
  const coverage = counts.focus ? counts.forecasted / counts.focus : 0;
  const actualizedShare = counts.focus ? counts.actualized / counts.focus : 0;
  const pendingShare = counts.focus ? counts.pending / counts.focus : 0;
  const outcomeTotal = counts.correct + counts.drawPush + counts.incorrect + counts.pending + counts.unresolved;
  const evaluatedMax = Math.max(1, counts.correct, counts.drawPush, counts.incorrect);

  const segments = [
    {
      key: "correct",
      label: "Model correct",
      shortLabel: "Correct",
      value: counts.correct,
      detail: `${counts.correct} actualized matches were exact model calls.`,
    },
    {
      key: "draw",
      label: "Actual draw / partial credit",
      shortLabel: "Draw credit",
      value: counts.drawPush,
      detail: `${counts.drawPush} actualized matches ended in draws and are treated as partial-credit rows.`,
    },
    {
      key: "incorrect",
      label: "Model incorrect",
      shortLabel: "Incorrect",
      value: counts.incorrect,
      detail: `${counts.incorrect} actualized matches disagreed with the model's called result.`,
    },
    {
      key: "pending",
      label: "Pending actual",
      shortLabel: "Pending",
      value: counts.pending,
      detail: `${counts.pending} announced forecasts are waiting for final scores.`,
    },
    {
      key: "unresolved",
      label: "Unresolved bracket slots",
      shortLabel: "Unresolved",
      value: counts.unresolved,
      detail: `${counts.unresolved} bracket slots are not forecasted yet because the teams are unresolved.`,
    },
  ];
  const defaultDetail = `${adjustedLabel} draw-adjusted score across ${counts.evaluated} actualized audit rows; exact strict accuracy is ${exactLabel}.`;

  // Build stage breakdown live from match data
  const stageOrder = ["Group stage", "R32", "QF", "SF", "3rd", "Final"];
  const stageMap = new Map();
  for (const m of (state.matches || [])) {
    if (!m.actual_available) continue;
    const rawStage = m.stage || "";
    const label = rawStage.includes("Group") ? "Group stage"
      : rawStage.includes("Round of 32") || rawStage.includes("R32") ? "R32"
      : rawStage.includes("Quarter") ? "QF"
      : rawStage.includes("Semi") ? "SF"
      : rawStage.includes("Third") || rawStage.includes("3rd") ? "3rd"
      : rawStage.includes("Final") ? "Final"
      : rawStage;
    if (!stageMap.has(label)) stageMap.set(label, { correct: 0, total: 0, credit: 0 });
    const s = stageMap.get(label);
    s.total++;
    s.correct += m.prediction_correct ? 1 : 0;
    s.credit += Number(m.prediction_credit || 0);
  }
  const stageRows = stageOrder
    .filter(l => stageMap.has(l))
    .map(l => ({ label: l, ...stageMap.get(l) }));

  // Compute overall totals for the summary row
  const allActualized = (state.matches || []).filter(m => m.actual_available);
  const overallCorrect = allActualized.filter(m => m.prediction_correct).length;
  const overallTotal = allActualized.length;
  const overallCredit = allActualized.reduce((s, m) => s + Number(m.prediction_credit || 0), 0);

  function fmtPct(n, d) { return d ? (n / d * 100).toFixed(1) + "%" : "—"; }

  const overallRowHtml = `
    <tr class="audit-stage-overall">
      <td class="audit-stage-label">Overall</td>
      <td class="audit-stage-counts">${overallCorrect}/${overallTotal}</td>
      <td class="audit-stage-pct">${fmtPct(overallCorrect, overallTotal)}</td>
      <td class="audit-stage-dadj">${fmtPct(overallCredit, overallTotal)}</td>
    </tr>`;

  const stageRowsHtml = stageRows.map(s => `
    <tr class="audit-stage-row">
      <td class="audit-stage-label">${escapeHtml(s.label)}</td>
      <td class="audit-stage-counts">${s.correct}/${s.total}</td>
      <td class="audit-stage-pct">${fmtPct(s.correct, s.total)}</td>
      <td class="audit-stage-dadj">${fmtPct(s.credit, s.total)}</td>
    </tr>`).join("");

  els.auditVisual.innerHTML = `
    <div class="audit-visual-grid">
      <article class="audit-viz-card audit-gauge-card">
        <div class="audit-viz-kicker">Draw-adjusted score</div>
        <button
          class="audit-gauge audit-viz-active"
          type="button"
          style="--score-angle: ${(adjusted * 360).toFixed(1)}deg;"
          data-audit-detail="${escapeHtml(defaultDetail)}"
          aria-label="${escapeHtml(`Draw-adjusted score ${adjustedLabel}`)}"
        >
          <span>${adjustedLabel}</span>
        </button>
        <p>Strict exact result accuracy: <strong>${exactLabel}</strong></p>
      </article>

      <article class="audit-viz-card audit-stage-card">
        <div class="audit-viz-kicker">Accuracy by stage</div>
        <table class="audit-stage-table">
          <thead>
            <tr>
              <th></th>
              <th>Correct</th>
              <th>Exact %</th>
              <th>Draw-adj</th>
            </tr>
          </thead>
          <tbody>
            ${overallRowHtml}
            ${stageRowsHtml}
          </tbody>
        </table>
      </article>

      <article class="audit-viz-card audit-stack-card">
        <div class="audit-viz-kicker">Forecast outcome mix</div>
        <div class="audit-stacked-bar" aria-label="Forecast outcome mix">
          ${segments.map((segment) => auditSegmentButton(segment, outcomeTotal)).join("")}
        </div>
        <div class="audit-viz-legend">
          ${segments
            .map(
              (segment) => `
                <span><i class="audit-legend-dot audit-dot-${segment.key}"></i>${segment.label}</span>
              `,
            )
            .join("")}
        </div>
      </article>

      <article class="audit-viz-card audit-histogram-card">
        <div class="audit-viz-kicker">Actualized match audit</div>
        <div class="audit-histogram" aria-label="Actualized model audit histogram">
          ${segments.slice(0, 3).map((segment) => auditBarButton(segment, evaluatedMax)).join("")}
        </div>
      </article>


      <article class="audit-viz-card audit-readiness-card">
        <div class="audit-viz-kicker">Data readiness</div>
        <div class="audit-readiness-row">
          <span>Forecast coverage</span>
          <strong>${counts.forecasted}/${counts.focus}</strong>
          <i><b style="--progress: ${(coverage * 100).toFixed(1)}%;"></b></i>
        </div>
        <div class="audit-readiness-row">
          <span>Actualized</span>
          <strong>${counts.actualized}</strong>
          <i><b style="--progress: ${(actualizedShare * 100).toFixed(1)}%;"></b></i>
        </div>
        <div class="audit-readiness-row">
          <span>Pending</span>
          <strong>${counts.pending}</strong>
          <i><b style="--progress: ${(pendingShare * 100).toFixed(1)}%;"></b></i>
        </div>
      </article>
    </div>
    <p class="audit-viz-detail" id="audit-viz-detail">${escapeHtml(defaultDetail)}</p>
  `;
}

function setAuditView(view) {
  state.auditView = view;
  const showingNumbers = view === "numbers";
  els.auditVisual.hidden = showingNumbers;
  els.auditGrid.hidden = !showingNumbers;
  els.auditViewToggle.textContent = showingNumbers ? "Show visuals" : "Show numbers";
  els.auditViewToggle.setAttribute("aria-pressed", String(showingNumbers));
}

function selectedMatch() {
  return state.filtered.find((match) => String(match.match_id) === String(state.selectedId)) || state.filtered[0] || null;
}

function activeFilterLabel() {
  const parts = [];
  const stage = els.stageFilter.value;
  const country = els.countryFilter.value;
  if (stage && stage !== "all") parts.push(stageFilterDisplay(stage));
  if (state.statusFilter === "upcoming") parts.push("upcoming forecasts");
  if (state.statusFilter === "actualized") parts.push("past results");
  if (country && country !== "all") parts.push(country);
  if (els.upsetToggle.checked) parts.push("high upset risk");
  return parts.length ? parts.join(" + ") : "current filters";
}

function clearRadarCanvas() {
  const context = els.radar.getContext("2d");
  context.clearRect(0, 0, els.radar.width, els.radar.height);
  context.fillStyle = "rgba(216, 234, 255, 0.62)";
  context.font = "700 22px system-ui";
  context.textAlign = "center";
  context.fillText("No matching matchup", els.radar.width / 2, els.radar.height / 2 - 6);
  context.font = "600 15px system-ui";
  context.fillText("Relax filters to restore the radar profile.", els.radar.width / 2, els.radar.height / 2 + 24);
}

function buildStageFilter() {
  const openingCount = filterCount({ stage: "stage_group:opening" });
  const knockoutCount = filterCount({ stage: "stage_group:knockout" });

  els.stageFilter.replaceChildren();
  els.stageFilter.append(new Option(`All phases (${state.matches.length})`, "all"));
  if (openingCount) {
    els.stageFilter.append(new Option(`Group-stage matches (${openingCount})`, "stage_group:opening"));
  }
  if (knockoutCount) {
    els.stageFilter.append(new Option(`Knockout matches (${knockoutCount})`, "stage_group:knockout"));
  }
}

function applyFilters() {
  const stage = els.stageFilter.value;
  const country = els.countryFilter.value;
  const status = state.statusFilter;
  const highUpsetOnly = els.upsetToggle.checked;

  state.filtered = state.matches.filter((match) => {
    const stageOk = stageFilterMatches(match, stage);
    const countryOk =
      country === "all" ||
      normalizeTeamKey(match.home_team) === normalizeTeamKey(country) ||
      normalizeTeamKey(match.away_team) === normalizeTeamKey(country);
    const statusOk = statusFilterMatches(match, status);
    const upsetOk = !highUpsetOnly || Number(match.upset_risk) >= 0.5;
    return stageOk && countryOk && statusOk && upsetOk;
  });

  if (!state.filtered.some((match) => String(match.match_id) === String(state.selectedId))) {
    state.selectedId = state.filtered[0]?.match_id ?? null;
  }

  buildMatchSelect();
  renderLineup();
  renderLineupDisclosure();
  renderFilterSuggestions();
  render();
}

function buildMatchSelect() {
  els.matchSelect.replaceChildren();
  if (!state.filtered.length) {
    els.matchSelect.disabled = true;
    els.matchSelect.append(new Option(`No matches for ${activeFilterLabel()}`, ""));
    return;
  }

  els.matchSelect.disabled = false;
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
  renderAuditVisual(audit);

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
  setAuditView(state.auditView);
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
  if (!match) {
    renderNoMatch();
    return;
  }

  state.selectedId = match.match_id;
  els.matchSelect.value = String(match.match_id);
  syncActiveArchetypes(match);
  renderTournamentContext(match);

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
  renderArchetypePlaybook(match);
  renderProbabilities(match);
  renderPrediction(match);
  renderReasoning(match);
  renderMetrics(match);
  renderImportance();
  renderSimilar(match);
  updateLineupSelection();
  drawRadar(match);
}

function renderNoMatch() {
  state.selectedId = null;
  state.activeArchetypeMode = "default";
  state.activeArchetypes = [];
  state.activeArchetypeMatchId = null;
  els.matchSelect.value = "";
  renderTournamentContext(null);
  document.documentElement.style.setProperty("--home", "#7fb7ff");
  document.documentElement.style.setProperty("--away", "#a8bad0");

  els.homeCode.textContent = "0";
  els.awayCode.textContent = "0";
  els.homeCode.classList.add("team-mark-fallback");
  els.awayCode.classList.add("team-mark-fallback");
  els.homeCode.setAttribute("aria-label", "No home team selected");
  els.awayCode.setAttribute("aria-label", "No away team selected");
  setTeamName(els.homeTeam, "No matching");
  setTeamName(els.awayTeam, "matchup");
  els.legendHome.textContent = "No matching team";
  els.legendAway.textContent = "Adjust filters";
  els.legendHome.style.color = "var(--pending)";
  els.legendAway.style.color = "var(--muted)";
  els.matchStage.textContent = "No matches for this filter combination";
  els.matchDate.textContent = activeFilterLabel();
  els.matchScore.textContent = "Relax one filter to restore match details";

  els.badgeList.innerHTML = `<span class="badge badge-tactical_contrast">No Matching Matchup</span>`;
  els.archetypePitch.innerHTML = `<div class="pitch-empty">No animated read available.</div>`;
  els.archetypeCommentary.innerHTML = `
    <strong>No matchup selected</strong>
    <p>Relax one filter to restore the archetype animation and commentary.</p>
  `;
  els.snapshotNote.textContent = `The current filters use AND logic, so a match must satisfy every selected filter. ${activeFilterLabel()} returns 0 rows.`;
  els.probBars.innerHTML = `<div class="panel-empty-note">No probability bars available for this filter combination.</div>`;
  els.probTable.innerHTML = "";
  els.upsetPill.textContent = "Upset risk: --";
  els.predictionCall.innerHTML = `
    <div class="prediction-winner">No selected match</div>
    <div class="prediction-confidence">Try clearing the country, tournament view, status, or upset-risk filter.</div>
  `;
  els.reasoningList.innerHTML = `
    <div class="reasoning-row">
      <span>Filter logic</span>
      <strong>All selected filters must match the same fixture.</strong>
    </div>
    <div class="reasoning-row">
      <span>Current result</span>
      <strong>0 rows</strong>
    </div>
  `;
  els.metricList.innerHTML = `
    <div class="metric-row">
      <span>Selected filters</span>
      <strong>${escapeHtml(activeFilterLabel())}</strong>
    </div>
  `;
  els.similarGrid.innerHTML = `<div class="panel-empty-note">No historical similarity card is shown until a matchup is selected.</div>`;
  renderImportance();
  updateLineupSelection();
  clearRadarCanvas();
}

function updateLineupSelection() {
  if (!els.timelineList) return;
  for (const card of els.timelineList.querySelectorAll(".timeline-card")) {
    card.classList.toggle("timeline-card-selected", String(card.dataset.matchId) === String(state.selectedId));
  }
}

function renderBadges(match) {
  els.badgeList.replaceChildren();
  const { labels, selected, isDefault } = syncActiveArchetypes(match);
  const defaultBadge = document.createElement("button");
  defaultBadge.type = "button";
  defaultBadge.className = "badge archetype-badge-button archetype-default-button";
  defaultBadge.dataset.archetypeDefault = "true";
  defaultBadge.setAttribute("aria-pressed", String(isDefault));
  defaultBadge.title = "Show all model-assigned archetype overlays";
  defaultBadge.textContent = "Model badges";
  els.badgeList.append(defaultBadge);

  const noOverlayBadge = document.createElement("button");
  noOverlayBadge.type = "button";
  noOverlayBadge.className = "badge archetype-badge-button archetype-none-button";
  noOverlayBadge.dataset.archetypeNone = "true";
  noOverlayBadge.setAttribute("aria-pressed", String(!isDefault && selected.length === 0));
  noOverlayBadge.title = "Hide archetype overlays without changing the prediction";
  noOverlayBadge.textContent = "No overlay";
  els.badgeList.append(noOverlayBadge);

  for (const label of labels) {
    const badge = document.createElement("button");
    badge.type = "button";
    badge.className = `badge archetype-badge-button badge-${archetypeClass.get(label) || "tactical_contrast"}`;
    badge.dataset.archetype = label;
    badge.setAttribute("aria-pressed", String(selected.includes(label)));
    badge.title = `${selected.includes(label) ? "Remove" : "Add"} ${label} lens`;
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

function matchArchetypeLabels(match) {
  return match.archetypes && match.archetypes.length ? match.archetypes : ["No Active Archetype"];
}

function syncActiveArchetypes(match) {
  const labels = matchArchetypeLabels(match);
  const matchId = String(match.match_id);
  if (state.activeArchetypeMatchId !== matchId) {
    state.activeArchetypeMode = "default";
    state.activeArchetypes = labels.slice();
    state.activeArchetypeMatchId = matchId;
  }

  if (state.activeArchetypeMode === "default") {
    state.activeArchetypes = labels.slice();
  } else {
    state.activeArchetypes = state.activeArchetypes.filter((label) => labels.includes(label));
  }

  return {
    labels,
    selected: state.activeArchetypeMode === "default" ? labels.slice() : state.activeArchetypes.slice(),
    isDefault: state.activeArchetypeMode === "default",
  };
}

function renderArchetypePlaybook(match) {
  const lens = archetypeLens(match);
  const pattern = lens.pattern;
  const intensity = archetypeIntensity(match);
  const predicted = predictedSide(match);
  const motion = matchAnimationMotion(match, lens, intensity, predicted);
  const homeColor = teamColor(match.home_team, "home");
  const awayColor = teamColor(match.away_team, "away");
  const homeInitials = match.home_code || teamInitials(match.home_team) || "H";
  const awayInitials = match.away_code || teamInitials(match.away_team) || "A";

  els.archetypePitch.innerHTML = `
    <div class="match-animation-shell" data-pattern="${escapeHtml(pattern)}" data-intensity="${escapeHtml(intensity)}" data-predicted="${escapeHtml(predicted)}" style="--anim-home:${homeColor}; --anim-away:${awayColor};">
      <div class="match-pitch match-pitch-${escapeHtml(motion)}" aria-hidden="true">
        <svg viewBox="0 0 240 156" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Animated soccer pitch summary">
          <rect width="240" height="156" fill="#183d24"></rect>
          <rect x="0" y="0" width="30" height="156" fill="#1f4a2d" opacity=".72"></rect>
          <rect x="60" y="0" width="30" height="156" fill="#1f4a2d" opacity=".72"></rect>
          <rect x="120" y="0" width="30" height="156" fill="#1f4a2d" opacity=".72"></rect>
          <rect x="180" y="0" width="30" height="156" fill="#1f4a2d" opacity=".72"></rect>
          <rect x="6" y="6" width="228" height="144" fill="none" stroke="rgba(255,255,255,.68)" stroke-width="1"></rect>
          <line x1="120" y1="6" x2="120" y2="150" stroke="rgba(255,255,255,.68)" stroke-width="1"></line>
          <circle cx="120" cy="78" r="21" fill="none" stroke="rgba(255,255,255,.68)" stroke-width="1"></circle>
          <circle cx="120" cy="78" r="1.8" fill="rgba(255,255,255,.72)"></circle>
          <rect x="6" y="44" width="34" height="68" fill="none" stroke="rgba(255,255,255,.52)" stroke-width=".8"></rect>
          <rect x="200" y="44" width="34" height="68" fill="none" stroke="rgba(255,255,255,.52)" stroke-width=".8"></rect>
          <rect x="1" y="64" width="5" height="28" fill="rgba(255,255,255,.08)" stroke="rgba(255,255,255,.82)" stroke-width="1"></rect>
          <rect x="234" y="64" width="5" height="28" fill="rgba(255,255,255,.08)" stroke="rgba(255,255,255,.82)" stroke-width="1"></rect>
          <circle cx="28" cy="78" r="1.5" fill="rgba(255,255,255,.45)"></circle>
          <circle cx="212" cy="78" r="1.5" fill="rgba(255,255,255,.45)"></circle>
          <circle class="anim-dot anim-dot-home" cx="22" cy="78" r="5.5" fill="var(--anim-home)" opacity=".95"></circle>
          <circle class="anim-dot anim-dot-home" cx="68" cy="50" r="5.5" fill="var(--anim-home)" opacity=".95"></circle>
          <circle class="anim-dot anim-dot-home" cx="68" cy="106" r="5.5" fill="var(--anim-home)" opacity=".95"></circle>
          <circle class="anim-dot anim-dot-home" cx="94" cy="78" r="5.5" fill="var(--anim-home)" opacity=".95"></circle>
          <circle class="anim-dot anim-dot-away" cx="218" cy="78" r="5.5" fill="var(--anim-away)" opacity=".95"></circle>
          <circle class="anim-dot anim-dot-away" cx="172" cy="50" r="5.5" fill="var(--anim-away)" opacity=".95"></circle>
          <circle class="anim-dot anim-dot-away" cx="172" cy="106" r="5.5" fill="var(--anim-away)" opacity=".95"></circle>
          <circle class="anim-dot anim-dot-away" cx="146" cy="78" r="5.5" fill="var(--anim-away)" opacity=".95"></circle>
          <defs><filter id="match-animation-drop"><feDropShadow dx="0" dy="1" stdDeviation="1.5" flood-opacity="0.42"></feDropShadow></filter></defs>
          <text class="match-ball" x="120" y="78" text-anchor="middle" dominant-baseline="central" font-size="14" filter="url(#match-animation-drop)">⚽</text>
        </svg>
      </div>
      <div class="match-animation-labels">
        <span style="color:${homeColor}">${escapeHtml(homeInitials)}</span>
        <span>${escapeHtml(animationNote(match, motion, lens))}</span>
        <span style="color:${awayColor}">${escapeHtml(awayInitials)}</span>
      </div>
    </div>
  `;

  els.archetypeCommentary.innerHTML = `
    <strong>${escapeHtml(lens.title)} read</strong>
    <p>${escapeHtml(archetypeNarrative(match, lens, intensity, motion))}</p>
  `;
}

function primaryArchetype(match) {
  return match.archetypes?.length ? match.archetypes[0] : "Baseline matchup";
}

function archetypeLens(match) {
  const { selected, isDefault } = syncActiveArchetypes(match);
  const patterns = selected.map((label) => archetypePattern(label));
  const pattern = selected.length === 0 ? "none" : selected.length === 1 ? patterns[0] : "mix";
  const title =
    selected.length === 0
      ? "No overlay"
      : isDefault
        ? "Model badges"
        : selected.join(" + ");
  return { isDefault, labels: selected, patterns, pattern, title };
}

function archetypePattern(label) {
  if (label === "Host Pressure") return "host";
  if (label === "Knockout Volatility") return "knockout";
  if (label === "Favorite Vs Underdog") return "favorite";
  if (label === "Heavyweight Clash") return "heavyweight";
  if (label === "Generational Transition") return "transition";
  if (label === "Club Power Mismatch") return "mismatch";
  if (label === "Tactical Contrast") return "tactical";
  if (label === "Upset Realized") return "upset";
  return "balanced";
}

function archetypeIntensity(match) {
  const risk = Number(match.upset_risk);
  const confidence = Number(predictionProbability(match));
  if (Number.isFinite(risk) && risk >= 0.45) return "high";
  if (Number.isFinite(confidence) && confidence >= 0.62) return "high";
  if (Number.isFinite(risk) && risk >= 0.28) return "medium";
  return "calm";
}

function matchAnimationMotion(match, lens, intensity, predicted) {
  const pattern = lens.pattern;
  const patterns = lens.patterns || [];
  if (!lens.labels.length) return "none";
  if (pattern === "mix") {
    if (patterns.includes("knockout") || patterns.includes("upset")) return "upset";
    if (patterns.includes("host") && patterns.includes("favorite")) return predicted === "away" ? "favorite-away" : "favorite-home";
    if (patterns.includes("host")) return "host";
    if (patterns.includes("tactical")) return "mix";
    return "mix";
  }
  if (pattern === "host") return "host";
  if (pattern === "favorite") return predicted === "away" ? "favorite-away" : "favorite-home";
  if (pattern === "tactical") return "tactical";
  if (pattern === "heavyweight") return "heavyweight";
  if (pattern === "transition") return "transition";
  if (pattern === "mismatch") return "mismatch";
  if (pattern === "upset") return "upset";
  if (pattern === "knockout" || intensity === "high") return "chaotic";
  if (Number(match.upset_risk) >= 0.34) return "chaotic";
  return "stable";
}

function animationNote(match, motion, lens) {
  const pattern = lens.pattern;
  if (lens.labels.length === 0) return "No archetype overlay · prediction unchanged";
  if (lens.isDefault) return "Model-assigned badges · overlay on";
  if (lens.labels.length > 1) return `${lens.labels.length} lenses · blended read`;
  if (motion === "host") return "Host pressure · home-side surge";
  if (motion === "favorite-home" || motion === "favorite-away") return "Forecast edge · favorite tilt";
  if (motion === "tactical") return "Shape vs shape · midfield probing";
  if (motion === "heavyweight") return "Midfield duel · pressure tradeoff";
  if (motion === "transition") return "Tempo shift · fresh legs";
  if (motion === "mismatch") return "Quality gap · overload pressure";
  if (motion === "upset") return "Upset lens · broken rhythm";
  if (motion === "chaotic") {
    if (pattern === "knockout") return "Knockout pressure · transition swings";
    return "High pressure · transitions everywhere";
  }
  if (pattern === "host") return "Host pressure · controlled build-up";
  if (pattern === "favorite") return "Forecast edge · stable tempo";
  if (pattern === "heavyweight") return "Midfield exchange · measured tempo";
  return "Possession build-up · stable tempo";
}

function archetypeNarrative(match, lens, intensity, motion) {
  const predicted = predictedTeam(match);
  const probability = pct(predictionProbability(match));
  const drawPressure = pct(match.probabilities?.draw);
  const upsetRisk = pct(match.upset_risk);
  const pattern = lens.pattern;
  const actual =
    isActualized(match)
      ? `Actual loaded: ${match.score}, so this forecast can be audited against the result.`
      : "Actual pending, so this remains a forecast-only read.";
  const patternLine = {
    none: "No archetype overlay is active; the field intentionally removes the archetype animation layer.",
    mix: `The route blends ${lens.labels.join(", ")} rather than isolating a single archetype.`,
    host: "The route starts from host-side pressure and pushes into the attacking half.",
    knockout: "The end-to-end route signals volatility around late swings, mistakes, and penalties.",
    favorite: "The tilted route follows the forecast edge toward the expected winner while leaving room for upset pressure.",
    heavyweight: "The central exchange shows two high-profile profiles trading pressure through midfield.",
    transition: "The diagonal route shows tempo changing as squad profile and lineup age become part of the read.",
    mismatch: "The overload route shows one side trying to turn a quality gap into territorial pressure.",
    tactical: "The central probing route shows shape-vs-shape tension before either side commits forward.",
    upset: "The broken route shows the match from an upset lens, where rhythm and game state become unstable.",
    balanced: "The balanced route stays closer to midfield because no single archetype dominates the read.",
  }[pattern];
  const intensityLine =
    lens.labels.length === 0
      ? "This is not an ablated model baseline; the prediction is unchanged."
      : lens.isDefault
        ? "Model badges: all loaded archetype overlays are included."
      : motion === "chaotic" || motion === "upset"
      ? "Chaotic motion: the model read is more exposed to transition or upset pressure."
      : motion === "host"
        ? "Host motion: crowd and venue context are being treated as part of the matchup pressure."
      : motion === "tactical"
        ? "Tactical motion: the ball stays central because the archetype is more about structure than raw chaos."
      : intensity === "high"
        ? "Fast model signal: high confidence even though the field motion stays structured."
      : intensity === "medium"
        ? "Moderate pulse: readable edge, not settled outcome."
        : "Calm pulse: lower-volatility model read.";

  return `${stageDisplay(match.stage)}: ${predicted} carries the headline forecast at ${probability}, with draw pressure ${drawPressure} and upset risk ${upsetRisk}. ${patternLine} ${intensityLine} ${actual} The animation is a model-read summary, not live play-by-play tracking.`;
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
  state.data = await loadMatchupData();
  state.matches = (state.data.matches || []).slice().sort(chronologySort);
  state.selectedId = state.matches[0]?.match_id ?? null;

  buildCountryFilter();
  renderSummary();
  renderAuditOverview();
  buildStageFilter();
  applyFilters();
  setupSectionNav();

  els.auditViewToggle.addEventListener("click", () => {
    setAuditView(state.auditView === "visual" ? "numbers" : "visual");
  });
  els.auditVisual.addEventListener("click", (event) => {
    const node = event.target.closest("[data-audit-detail]");
    if (!node) return;
    for (const item of els.auditVisual.querySelectorAll("[data-audit-detail]")) {
      item.classList.toggle("audit-viz-active", item === node);
    }
    const detail = els.auditVisual.querySelector("#audit-viz-detail");
    if (detail) detail.textContent = node.dataset.auditDetail || "";
  });
  els.badgeList.addEventListener("click", (event) => {
    const reset = event.target.closest("[data-archetype-default]");
    const noOverlay = event.target.closest("[data-archetype-none]");
    const badge = event.target.closest("[data-archetype]");
    if (!reset && !noOverlay && !badge) return;
    const match = selectedMatch();
    if (!match) return;
    const labels = matchArchetypeLabels(match);
    if (reset) {
      state.activeArchetypeMode = "default";
      state.activeArchetypes = labels.slice();
    } else if (noOverlay) {
      state.activeArchetypeMode = "custom";
      state.activeArchetypes = [];
    } else {
      const label = badge.dataset.archetype;
      const current = state.activeArchetypeMode === "default" ? labels.slice() : state.activeArchetypes.slice();
      const next = current.includes(label) ? current.filter((item) => item !== label) : [...current, label];
      state.activeArchetypeMode = next.length === labels.length ? "default" : "custom";
      state.activeArchetypes = state.activeArchetypeMode === "default" ? labels.slice() : next;
    }
    state.activeArchetypeMatchId = String(match.match_id);
    renderBadges(match);
    renderArchetypePlaybook(match);
  });
  els.viewChipRow.addEventListener("click", (event) => {
    const chip = event.target.closest("[data-filter-kind]");
    if (!chip) return;
    const kind = chip.dataset.filterKind;
    const value = chip.dataset.filterValue;
    const isActive = chip.getAttribute("aria-pressed") === "true";
    if (kind === "clear") {
      els.stageFilter.value = "all";
      els.countryFilter.value = "all";
      els.upsetToggle.checked = false;
      state.statusFilter = "all";
    } else if (kind === "status") {
      els.stageFilter.value = "all";
      els.upsetToggle.checked = false;
      state.statusFilter = isActive ? "all" : value;
    } else if (kind === "stage") {
      state.statusFilter = "all";
      els.upsetToggle.checked = false;
      els.stageFilter.value = isActive ? "all" : value;
    } else if (kind === "upset") {
      els.stageFilter.value = "all";
      state.statusFilter = "all";
      els.upsetToggle.checked = !isActive;
    }
    applyFilters();
  });
  els.activeFilterRow.addEventListener("click", (event) => {
    const pill = event.target.closest("[data-clear-filter]");
    if (!pill) return;
    const filter = pill.dataset.clearFilter;
    if (filter === "stage") els.stageFilter.value = "all";
    if (filter === "status") state.statusFilter = "all";
    if (filter === "country") els.countryFilter.value = "all";
    if (filter === "upset") els.upsetToggle.checked = false;
    applyFilters();
  });
  els.countryChipRow.addEventListener("click", (event) => {
    const chip = event.target.closest("[data-filter-kind='country']");
    if (!chip) return;
    const value = chip.dataset.filterValue;
    els.countryFilter.value = normalizeTeamKey(els.countryFilter.value) === normalizeTeamKey(value) ? "all" : value;
    applyFilters();
  });
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
  window.addEventListener("resize", () => {
    const match = selectedMatch();
    if (match) drawRadar(match);
    else clearRadarCanvas();
  });

}

init().catch((error) => {
  console.error(error);
  renderLoadError(error);
});
