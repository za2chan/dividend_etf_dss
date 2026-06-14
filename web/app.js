const state = {
  data: null,
  budget: 50000,
  scenario: "max_weight_35",
  planName: "Balanced",
  bucket: "all",
  category: "all",
  search: "",
};

const palette = [
  "#16737c",
  "#1f7a4d",
  "#b7791f",
  "#3566a8",
  "#8a5d3b",
  "#6f5a9a",
  "#b33a3a",
  "#4b7f52",
  "#9b6b1d",
  "#477b8e",
  "#7b6a43",
  "#8f4f68",
];

const bucketColors = {
  Safe: "#1f7a4d",
  Watch: "#b7791f",
  Risky: "#b33a3a",
};

const elements = {
  dataMeta: document.querySelector("#dataMeta"),
  statusStrip: document.querySelector("#statusStrip"),
  budgetInput: document.querySelector("#budgetInput"),
  scenarioSelect: document.querySelector("#scenarioSelect"),
  planTabs: document.querySelector("#planTabs"),
  bucketFilter: document.querySelector("#bucketFilter"),
  categoryFilter: document.querySelector("#categoryFilter"),
  searchInput: document.querySelector("#searchInput"),
  kpiGrid: document.querySelector("#kpiGrid"),
  allocationSubtitle: document.querySelector("#allocationSubtitle"),
  donutChart: document.querySelector("#donutChart"),
  weightBars: document.querySelector("#weightBars"),
  scatterPlot: document.querySelector("#scatterPlot"),
  modelSubtitle: document.querySelector("#modelSubtitle"),
  modelMetrics: document.querySelector("#modelMetrics"),
  holdingsSubtitle: document.querySelector("#holdingsSubtitle"),
  holdingsBody: document.querySelector("#holdingsBody"),
  scoresSubtitle: document.querySelector("#scoresSubtitle"),
  scoresBody: document.querySelector("#scoresBody"),
  screeningSubtitle: document.querySelector("#screeningSubtitle"),
  droppedList: document.querySelector("#droppedList"),
  tooltip: document.querySelector("#tooltip"),
};

async function init() {
  try {
    const response = await fetch("data/latest.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.data = await response.json();
    state.budget = Number(state.data.metadata.budget || 50000);
    elements.budgetInput.value = String(Math.round(state.budget));
    bindEvents();
    renderStaticControls();
    render();
  } catch (error) {
    document.body.innerHTML = `<main class="app-shell"><section class="panel"><p class="empty">Could not load data/latest.json: ${escapeHtml(error.message)}</p></section></main>`;
  }
}

function bindEvents() {
  elements.budgetInput.addEventListener("input", () => {
    state.budget = Math.max(0, Number(elements.budgetInput.value || 0));
    render();
  });
  elements.scenarioSelect.addEventListener("change", () => {
    state.scenario = elements.scenarioSelect.value;
    render();
  });
  elements.bucketFilter.addEventListener("change", () => {
    state.bucket = elements.bucketFilter.value;
    render();
  });
  elements.categoryFilter.addEventListener("change", () => {
    state.category = elements.categoryFilter.value;
    render();
  });
  elements.searchInput.addEventListener("input", () => {
    state.search = elements.searchInput.value.trim().toUpperCase();
    render();
  });
}

function renderStaticControls() {
  const plans = currentPlans();
  state.planName = plans.some((plan) => plan.name === state.planName) ? state.planName : plans[0]?.name;
  elements.planTabs.innerHTML = plans
    .map((plan) => `<button type="button" data-plan="${escapeHtml(plan.name)}">${escapeHtml(plan.name)}</button>`)
    .join("");
  elements.planTabs.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.planName = button.dataset.plan;
      render();
    });
  });

  const categories = [...new Set(state.data.scores.map((score) => score.category))].sort();
  elements.categoryFilter.innerHTML =
    `<option value="all">All categories</option>` +
    categories.map((category) => `<option value="${escapeHtml(category)}">${escapeHtml(labelize(category))}</option>`).join("");
}

function render() {
  if (!state.data) return;
  const plan = currentPlan();
  if (!plan) return;

  elements.planTabs.querySelectorAll("button").forEach((button) => {
    button.classList.toggle("active", button.dataset.plan === state.planName);
  });

  renderHeader();
  renderKpis(plan);
  renderAllocation(plan);
  renderScatter();
  renderModel();
  renderHoldings(plan);
  renderScores();
  renderScreening();
}

function renderHeader() {
  const meta = state.data.metadata;
  elements.dataMeta.textContent = `Data as of ${meta.source_data_as_of || "n/a"} · generated ${formatDateTime(meta.generated_at)}`;
  elements.statusStrip.innerHTML = [
    `${meta.screened_count}/${meta.candidate_count} screened`,
    `${formatPct(meta.cut_label_rate)} cut-label rate`,
    `${meta.source_mode}`,
  ]
    .map((text) => `<span class="pill">${escapeHtml(text)}</span>`)
    .join("");
}

function renderKpis(plan) {
  const holdings = recalcHoldings(plan);
  const invested = holdings.reduce((sum, holding) => sum + holding.shares * holding.price, 0);
  const watchWeight = holdings
    .filter((holding) => holding.bucket === "Watch")
    .reduce((sum, holding) => sum + holding.weight, 0);
  const monthlyIncome = (state.budget * plan.expected_yield) / 12;
  elements.kpiGrid.innerHTML = [
    ["Monthly Income", formatMoney(monthlyIncome), `${formatMoney(state.budget)} budget`],
    ["Expected Yield", formatPct(plan.expected_yield), plan.risk_level],
    ["Annual Vol", formatPct(plan.annual_volatility), "Close-return covariance"],
    ["Holdings", String(holdings.length), `${formatPct(watchWeight)} Watch weight`],
    ["Invested", formatMoney(invested), `${formatMoney(Math.max(state.budget - invested, 0))} cash after rounding`],
  ]
    .map(([label, value, sub]) => `<div class="kpi"><span>${label}</span><strong>${value}</strong><small>${sub}</small></div>`)
    .join("");
}

function renderAllocation(plan) {
  const holdings = recalcHoldings(plan);
  elements.allocationSubtitle.textContent = `${plan.name} · ${state.scenario === "no_max_weight" ? "no max weight" : "35% max weight"}`;
  elements.donutChart.innerHTML = donutSvg(holdings);
  elements.weightBars.innerHTML = holdings
    .map((holding, index) => {
      const color = palette[index % palette.length];
      return `
        <div class="bar-row">
          <span class="ticker-cell">${escapeHtml(holding.ticker)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${holding.weight * 100}%; background:${color}"></div></div>
          <span>${formatPct(holding.weight)}</span>
        </div>
      `;
    })
    .join("");
}

function renderScatter() {
  const scores = filteredScores();
  elements.scatterPlot.innerHTML = scatterSvg(scores);
  elements.scatterPlot.querySelectorAll("[data-point]").forEach((point) => {
    point.addEventListener("mousemove", (event) => {
      elements.tooltip.hidden = false;
      elements.tooltip.innerHTML = point.dataset.tooltip;
      elements.tooltip.style.left = `${event.clientX + 12}px`;
      elements.tooltip.style.top = `${event.clientY + 12}px`;
    });
    point.addEventListener("mouseleave", () => {
      elements.tooltip.hidden = true;
    });
    point.addEventListener("click", () => {
      elements.searchInput.value = point.dataset.ticker;
      state.search = point.dataset.ticker;
      render();
    });
  });
}

function renderModel() {
  const model = state.data.classifier;
  const metrics = model.metrics || {};
  elements.modelSubtitle.textContent = `${model.training_rows.toLocaleString()} rows · ${formatPct(model.positive_rate)} positive`;
  elements.modelMetrics.innerHTML = [
    ["Validation", metrics.status || "n/a"],
    ["Folds", metrics.folds ?? "n/a"],
    ["PR-AUC", formatNumber(metrics.pr_auc, 3)],
    ["Risky precision", formatNumber(metrics.risky_precision, 3)],
    ["Baseline PR-AUC", formatNumber(metrics.baseline_pr_auc, 3)],
    ["Baseline precision", formatNumber(metrics.baseline_precision, 3)],
    ["Random PR-AUC", formatNumber(metrics.random_pr_auc, 3)],
  ]
    .map(([label, value]) => `<div class="metric-item"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

function renderHoldings(plan) {
  const holdings = recalcHoldings(plan);
  elements.holdingsSubtitle.textContent = `${holdings.length} positions · ${formatMoney((state.budget * plan.expected_yield) / 12)} projected monthly income`;
  elements.holdingsBody.innerHTML = holdings
    .map(
      (holding) => `
        <tr>
          <td class="ticker-cell">${escapeHtml(holding.ticker)}</td>
          <td><span class="bucket ${holding.bucket}">${holding.bucket}</span></td>
          <td>${formatPct(holding.weight)}</td>
          <td>${holding.shares.toLocaleString()}</td>
          <td>${formatMoney(holding.price)}</td>
          <td>${formatPct(holding.dist_yield)}</td>
          <td>${formatPct(holding.p_cut)}</td>
        </tr>
      `,
    )
    .join("");
}

function renderScores() {
  const scores = filteredScores();
  elements.scoresSubtitle.textContent = `${scores.length} ETFs shown`;
  elements.scoresBody.innerHTML = scores
    .map(
      (score) => `
        <tr>
          <td class="ticker-cell">${escapeHtml(score.ticker)}</td>
          <td>${escapeHtml(labelize(score.category))}</td>
          <td><span class="bucket ${score.bucket}">${score.bucket}</span></td>
          <td>${formatPct(score.p_cut)}</td>
          <td>${formatPct(score.dist_yield)}</td>
          <td>${formatMoney(score.price)}</td>
        </tr>
      `,
    )
    .join("");
}

function renderScreening() {
  const dropped = state.data.screening.dropped || [];
  elements.screeningSubtitle.textContent = `${state.data.metadata.screened_count} kept · ${dropped.length} dropped`;
  elements.droppedList.innerHTML = dropped.length
    ? dropped
        .map((item) => `<span class="dropped-item"><strong>${escapeHtml(item.ticker)}</strong>${escapeHtml(item.reason)}</span>`)
        .join("")
    : `<span class="empty">No dropped tickers.</span>`;
}

function currentPlans() {
  return state.data?.scenarios?.[state.scenario] || [];
}

function currentPlan() {
  return currentPlans().find((plan) => plan.name === state.planName) || currentPlans()[0];
}

function recalcHoldings(plan) {
  return [...plan.holdings]
    .map((holding) => ({
      ...holding,
      shares: Math.floor((state.budget * holding.weight) / holding.price),
    }))
    .sort((a, b) => b.weight - a.weight);
}

function filteredScores() {
  return state.data.scores
    .filter((score) => state.bucket === "all" || score.bucket === state.bucket)
    .filter((score) => state.category === "all" || score.category === state.category)
    .filter((score) => !state.search || score.ticker.includes(state.search))
    .sort((a, b) => a.p_cut - b.p_cut);
}

function donutSvg(holdings) {
  const size = 220;
  const cx = 110;
  const cy = 110;
  const radius = 78;
  const stroke = 30;
  let angle = -90;
  const segments = holdings
    .map((holding, index) => {
      const sweep = holding.weight * 360;
      const path = arcPath(cx, cy, radius, angle, angle + sweep);
      angle += sweep;
      return `<path d="${path}" fill="none" stroke="${palette[index % palette.length]}" stroke-width="${stroke}" stroke-linecap="butt"></path>`;
    })
    .join("");
  const top = holdings[0];
  return `
    <svg viewBox="0 0 ${size} ${size}" role="img" aria-label="Allocation donut">
      <circle cx="${cx}" cy="${cy}" r="${radius}" fill="none" stroke="#edf0ec" stroke-width="${stroke}"></circle>
      ${segments}
      <text x="${cx}" y="${cy - 5}" text-anchor="middle" font-size="24" font-weight="760" fill="#1f2523">${escapeHtml(top?.ticker || "")}</text>
      <text x="${cx}" y="${cy + 18}" text-anchor="middle" font-size="13" fill="#65706c">${top ? formatPct(top.weight) : ""}</text>
    </svg>
  `;
}

function scatterSvg(scores) {
  const width = 520;
  const height = 320;
  const margin = { top: 14, right: 18, bottom: 42, left: 52 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const maxYield = Math.max(0.12, ...scores.map((score) => score.dist_yield || 0)) * 1.08;
  const points = scores
    .map((score) => {
      const x = margin.left + clamp(score.p_cut || 0, 0, 1) * innerW;
      const y = margin.top + (1 - clamp((score.dist_yield || 0) / maxYield, 0, 1)) * innerH;
      const color = bucketColors[score.bucket] || "#65706c";
      const tooltip = `${escapeHtml(score.ticker)}<br>${escapeHtml(labelize(score.category))}<br>P(cut) ${formatPct(score.p_cut)} · Yield ${formatPct(score.dist_yield)}`;
      return `<circle data-point data-ticker="${escapeHtml(score.ticker)}" data-tooltip="${tooltip}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="5.5" fill="${color}" fill-opacity="0.82"></circle>`;
    })
    .join("");
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="ETF score scatter plot">
      ${[0, 0.25, 0.5, 0.75, 1]
        .map((tick) => {
          const x = margin.left + tick * innerW;
          return `<line class="grid-line" x1="${x}" y1="${margin.top}" x2="${x}" y2="${margin.top + innerH}"></line><text class="axis-label" x="${x}" y="${height - 16}" text-anchor="middle">${Math.round(tick * 100)}%</text>`;
        })
        .join("")}
      ${[0, 0.25, 0.5, 0.75, 1]
        .map((tick) => {
          const y = margin.top + (1 - tick) * innerH;
          return `<line class="grid-line" x1="${margin.left}" y1="${y}" x2="${margin.left + innerW}" y2="${y}"></line><text class="axis-label" x="${margin.left - 10}" y="${y + 4}" text-anchor="end">${Math.round(tick * maxYield * 100)}%</text>`;
        })
        .join("")}
      <line x1="${margin.left}" y1="${margin.top + innerH}" x2="${margin.left + innerW}" y2="${margin.top + innerH}" stroke="#9aa39d"></line>
      <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + innerH}" stroke="#9aa39d"></line>
      <text class="axis-label" x="${margin.left + innerW / 2}" y="${height - 2}" text-anchor="middle">P(cut)</text>
      <text class="axis-label" transform="translate(14 ${margin.top + innerH / 2}) rotate(-90)" text-anchor="middle">Distribution yield</text>
      ${points}
    </svg>
  `;
}

function arcPath(cx, cy, radius, startAngle, endAngle) {
  const start = polar(cx, cy, radius, endAngle);
  const end = polar(cx, cy, radius, startAngle);
  const largeArc = endAngle - startAngle <= 180 ? 0 : 1;
  return `M ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArc} 0 ${end.x} ${end.y}`;
}

function polar(cx, cy, radius, angle) {
  const radians = ((angle - 90) * Math.PI) / 180;
  return {
    x: cx + radius * Math.cos(radians),
    y: cy + radius * Math.sin(radians),
  };
}

function formatMoney(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: value >= 1000 ? 0 : 2,
  }).format(Number(value || 0));
}

function formatPct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toFixed(digits);
}

function formatDateTime(value) {
  if (!value) return "n/a";
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function labelize(value) {
  return String(value || "").replaceAll("_", " ");
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

init();
