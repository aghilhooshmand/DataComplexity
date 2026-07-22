/* Data Complexity Explorer — browse, compare, distributions */

const CHEAP_METRICS = [
  "F1", "F1v", "F2", "F3", "F4", "input_noise", "R_value", "deg_overlap",
  "N3", "SI", "N4", "kDN", "D3_value", "CM", "N1", "Clust", "LSC", "N2",
  "MRCA", "C1", "C2", "purity", "neighbourhood_separability", "borderline",
];

const STANDARD_EXTRA = ["ONB", "DBC"];
const STANDARD_METRICS = [...CHEAP_METRICS, ...STANDARD_EXTRA];

const SIZE_VARS = [
  { key: "n_rows_original", label: "Rows (original)" },
  { key: "n_rows_used", label: "Rows (used)" },
  { key: "n_columns_original", label: "Columns (original)" },
  { key: "n_features_after_encoding", label: "Features (after encoding)" },
  { key: "n_features_raw", label: "Features (raw)" },
  { key: "n_classes", label: "Classes" },
  { key: "majority_class_fraction", label: "Majority class fraction" },
];

const VIEW_META = {
  overview: { kicker: "Overview", title: "Snapshot of computed complexity" },
  browse: { kicker: "All datasets", title: "Searchable complexity table" },
  compare: { kicker: "Compare", title: "Side-by-side dataset metrics" },
  glance: { kicker: "Complexity glance", title: "Hardness bars, heatmap, and radar" },
  distributions: { kicker: "Distributions", title: "Shape of metrics and sizes" },
  coverage: { kicker: "Coverage", title: "What is complete vs still missing" },
};

const state = {
  rows: [],
  charts: {},
};

function present(val) {
  if (val === null || val === undefined) return false;
  if (typeof val === "number" && Number.isNaN(val)) return false;
  const s = String(val).trim();
  return s !== "" && s.toLowerCase() !== "nan" && s.toLowerCase() !== "none";
}

function num(val) {
  if (!present(val)) return null;
  const n = Number(val);
  return Number.isFinite(n) ? n : null;
}

function metricCol(m) {
  return `pycol_${m}`;
}

function fillCount(row, metrics) {
  let ok = 0;
  for (const m of metrics) {
    if (present(row[metricCol(m)])) ok += 1;
  }
  return ok;
}

function missingMetrics(row, metrics) {
  return metrics.filter((m) => !present(row[metricCol(m)]));
}

function datasetLabel(row) {
  const name = row.dataset_name || row.dataset_file || "unknown";
  return String(name).replace(/\.csv$/i, "");
}

function median(values) {
  if (!values.length) return null;
  const a = [...values].sort((x, y) => x - y);
  const mid = Math.floor(a.length / 2);
  return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
}

function destroyChart(key) {
  if (state.charts[key]) {
    state.charts[key].destroy();
    delete state.charts[key];
  }
}

function setStatus(text) {
  const el = document.getElementById("loadStatus");
  if (el) el.textContent = text;
}

function showView(name) {
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  document.querySelectorAll(".menu-item").forEach((b) => b.classList.remove("active"));
  const view = document.getElementById(`view-${name}`);
  const btn = document.querySelector(`.menu-item[data-view="${name}"]`);
  if (view) view.classList.add("active");
  if (btn) btn.classList.add("active");
  const meta = VIEW_META[name];
  if (meta) {
    document.getElementById("pageKicker").textContent = meta.kicker;
    document.getElementById("pageTitle").textContent = meta.title;
  }
  closeNav();
  if (name === "overview") renderOverview();
  if (name === "browse") renderBrowse();
  if (name === "compare") renderCompare();
  if (name === "glance") renderGlance();
  if (name === "distributions") renderDistributions();
  if (name === "coverage") renderCoverage();
}

function closeNav() {
  document.getElementById("sidebar")?.classList.remove("open");
  const backdrop = document.getElementById("backdrop");
  if (backdrop) backdrop.hidden = true;
}

function openNav() {
  document.getElementById("sidebar")?.classList.add("open");
  const backdrop = document.getElementById("backdrop");
  if (backdrop) backdrop.hidden = false;
}

function enrich(rows) {
  return rows
    .filter((r) => present(r.dataset_file) && r.dataset_file !== "datasets_complexity_summary.csv")
    .map((r) => {
      const cheap = fillCount(r, CHEAP_METRICS);
      const std = fillCount(r, STANDARD_METRICS);
      return {
        ...r,
        _label: datasetLabel(r),
        cheap_fill: cheap,
        cheap_pct: (100 * cheap) / CHEAP_METRICS.length,
        std_fill: std,
        std_pct: (100 * std) / STANDARD_METRICS.length,
        missing_cheap: missingMetrics(r, CHEAP_METRICS),
        missing_std: missingMetrics(r, STANDARD_METRICS),
      };
    });
}

function renderOverview() {
  const rows = state.rows;
  const cheapDone = rows.filter((r) => r.cheap_fill === CHEAP_METRICS.length).length;
  const stdDone = rows.filter((r) => r.std_fill === STANDARD_METRICS.length).length;
  const nRows = rows.map((r) => num(r.n_rows_original)).filter((v) => v !== null);
  const nFeat = rows.map((r) => num(r.n_features_after_encoding)).filter((v) => v !== null);

  document.getElementById("kpiRow").innerHTML = [
    kpi("Datasets", rows.length, "In summary CSV"),
    kpi("Cheap complete", `${cheapDone}/${rows.length}`, `${((100 * cheapDone) / rows.length).toFixed(0)}% with all 24`),
    kpi("Standard complete", `${stdDone}/${rows.length}`, "Cheap + ONB + DBC"),
    kpi("Median rows", formatNum(median(nRows)), `Median features ${formatNum(median(nFeat))}`),
  ].join("");

  destroyChart("scatter");
  const scatterCtx = document.getElementById("scatterSize");
  state.charts.scatter = new Chart(scatterCtx, {
    type: "scatter",
    data: {
      datasets: [{
        label: "Datasets",
        data: rows.map((r) => ({
          x: num(r.n_rows_original),
          y: num(r.n_features_after_encoding),
          label: r._label,
        })).filter((d) => d.x !== null && d.y !== null),
        backgroundColor: "rgba(15, 118, 110, 0.55)",
        borderColor: "rgba(15, 118, 110, 0.9)",
        pointRadius: 5,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label(ctx) {
              const raw = ctx.raw;
              return `${raw.label}: rows=${raw.x}, features=${raw.y}`;
            },
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: "Rows" },
          type: "logarithmic",
        },
        y: {
          title: { display: true, text: "Features (encoded)" },
          type: "logarithmic",
        },
      },
    },
  });

  destroyChart("classes");
  const classCounts = {};
  for (const r of rows) {
    const c = num(r.n_classes);
    if (c === null) continue;
    const key = String(Math.round(c));
    classCounts[key] = (classCounts[key] || 0) + 1;
  }
  const labels = Object.keys(classCounts).sort((a, b) => Number(a) - Number(b));
  state.charts.classes = new Chart(document.getElementById("barClasses"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: labels.map((k) => classCounts[k]),
        backgroundColor: "#0f766e",
        borderRadius: 8,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: "n_classes" } },
        y: { title: { display: true, text: "Datasets" }, ticks: { precision: 0 } },
      },
    },
  });
}

function kpi(label, value, sub) {
  return `<div class="kpi"><p class="label">${escapeHtml(label)}</p><p class="value">${escapeHtml(String(value))}</p><p class="sub">${escapeHtml(sub || "")}</p></div>`;
}

function formatNum(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (Math.abs(v) >= 100) return Math.round(v).toLocaleString();
  return Number(v).toPrecision(4).replace(/\.?0+$/, "");
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function pillFor(fill, total) {
  const cls = fill === total ? "" : fill >= total - 2 ? "warn" : "bad";
  return `<span class="pill ${cls}">${fill}/${total}</span>`;
}

function renderBrowse() {
  const q = (document.getElementById("browseSearch").value || "").trim().toLowerCase();
  const sortKey = document.getElementById("browseSort").value;
  const order = document.getElementById("browseOrder").value;

  let rows = [...state.rows];
  if (q) {
    rows = rows.filter((r) => {
      const blob = [r.dataset_file, r.dataset_name, r._label, r.download_url, r.profile_url]
        .map((x) => String(x || "").toLowerCase())
        .join(" ");
      return blob.includes(q);
    });
  }

  rows.sort((a, b) => {
    const av = num(a[sortKey]) ?? String(a[sortKey] || "").toLowerCase();
    const bv = num(b[sortKey]) ?? String(b[sortKey] || "").toLowerCase();
    if (typeof av === "number" && typeof bv === "number") {
      return order === "asc" ? av - bv : bv - av;
    }
    const cmp = String(av).localeCompare(String(bv));
    return order === "asc" ? cmp : -cmp;
  });

  const tbody = document.querySelector("#browseTable tbody");
  tbody.innerHTML = rows.map((r) => {
    const links = [];
    if (present(r.download_url)) links.push(`<a class="link" href="${escapeHtml(r.download_url)}" target="_blank" rel="noopener">CSV.gz</a>`);
    if (present(r.profile_url)) links.push(`<a class="link" href="${escapeHtml(r.profile_url)}" target="_blank" rel="noopener">Profile</a>`);
    return `<tr>
      <td><strong>${escapeHtml(r._label)}</strong></td>
      <td>${escapeHtml(formatNum(num(r.n_rows_original)))}</td>
      <td>${escapeHtml(formatNum(num(r.n_columns_original)))}</td>
      <td>${escapeHtml(formatNum(num(r.n_features_after_encoding)))}</td>
      <td>${escapeHtml(formatNum(num(r.n_classes)))}</td>
      <td>${pillFor(r.cheap_fill, CHEAP_METRICS.length)}</td>
      <td>${pillFor(r.std_fill, STANDARD_METRICS.length)}</td>
      <td>${links.join(" · ") || "—"}</td>
    </tr>`;
  }).join("");
}

function metricListLabel(metric) {
  const hints = (typeof METRIC_HINTS !== "undefined") ? METRIC_HINTS : {};
  const h = hints[metric];
  if (!h) {
    return `${metric} → higher: more complex [Likely]`;
  }
  if (h.certainty === "context" || /context/i.test(h.hard || "")) {
    return `${metric} → context-dependent [Context]`;
  }
  const lessComplex = /less complex/i.test(h.hard || "");
  const tag = h.certainty === "certain" ? "" : " [Likely]";
  if (lessComplex) {
    return `${metric} → higher: less complex${tag}`;
  }
  return `${metric} → higher: more complex${tag}`;
}

function fillCompareSelectors() {
  const dsSel = document.getElementById("compareSelect");
  const mSel = document.getElementById("compareMetrics");
  if (!dsSel.options.length) {
    const sorted = [...state.rows].sort((a, b) => a._label.localeCompare(b._label));
    dsSel.innerHTML = sorted.map((r) => `<option value="${escapeHtml(r.dataset_file)}">${escapeHtml(r._label)}</option>`).join("");
    const defaults = ["iris.csv", "wine_quality_red.csv", "ecoli.csv", "yeast.csv"].filter((d) =>
      sorted.some((r) => r.dataset_file === d)
    );
    [...dsSel.options].forEach((opt) => {
      opt.selected = defaults.includes(opt.value);
    });
  }
  if (!mSel.options.length) {
    const defaults = ["F1", "N1", "N2", "N3", "LSC", "borderline", "C1", "C2"];
    mSel.innerHTML = STANDARD_METRICS.map((m) =>
      `<option value="${escapeHtml(m)}" ${defaults.includes(m) ? "selected" : ""}>${escapeHtml(metricListLabel(m))}</option>`
    ).join("");
  }
}

function selectedOptions(selectEl) {
  return [...selectEl.selectedOptions].map((o) => o.value);
}

function syncCompareSortOptions(metrics) {
  const sel = document.getElementById("compareSortBy");
  if (!sel) return;
  const prev = sel.value;
  const opts = [
    ["name", "Dataset name"],
    ["n_rows_original", "Rows"],
    ["n_columns_original", "Columns"],
    ["n_features_after_encoding", "Features"],
    ["n_classes", "Classes"],
    ...metrics.map((m) => [`metric:${m}`, `Metric · ${m}`]),
  ];
  sel.innerHTML = opts.map(([v, label]) =>
    `<option value="${escapeHtml(v)}">${escapeHtml(label)}</option>`
  ).join("");
  const values = opts.map(([v]) => v);
  sel.value = values.includes(prev) ? prev : (metrics[0] ? `metric:${metrics[0]}` : "name");
}

function sortCompareRows(rows, metrics, sortBy, order) {
  const dir = order === "asc" ? 1 : -1;
  const sorted = [...rows];
  sorted.sort((a, b) => {
    if (sortBy === "name") {
      return dir * a._label.localeCompare(b._label);
    }
    let key = sortBy;
    if (sortBy.startsWith("metric:")) {
      key = metricCol(sortBy.slice("metric:".length));
    }
    const av = num(a[key]);
    const bv = num(b[key]);
    if (av === null && bv === null) return dir * a._label.localeCompare(b._label);
    if (av === null) return 1;
    if (bv === null) return -1;
    if (av === bv) return dir * a._label.localeCompare(b._label);
    return dir * (av - bv);
  });
  return sorted;
}

function renderMetricHints(metrics) {
  const box = document.getElementById("metricHintsList");
  if (!box) return;
  const hints = (typeof METRIC_HINTS !== "undefined") ? METRIC_HINTS : {};
  if (!metrics.length) {
    box.innerHTML = "<p class='hint-blurb'>Select one or more metrics to see how to read them.</p>";
    return;
  }
  box.innerHTML = metrics.map((m) => {
    const h = hints[m];
    if (!h) {
      return `<div class="hint-card"><div class="hint-top"><span class="hint-name">${escapeHtml(m)}</span><span class="hint-cert likely">[Likely]</span><span class="hint-hard">Higher → more complex (default reading)</span></div><p class="hint-blurb">No project note for this metric; treat higher as harder unless you know otherwise.</p></div>`;
    }
    return `<div class="hint-card">
      <div class="hint-top">
        <span class="hint-name">${escapeHtml(m)}</span>
        <span class="hint-cert ${escapeHtml(h.certainty)}">${escapeHtml(h.certaintyLabel)}</span>
        <span class="hint-hard">${escapeHtml(h.hard)}</span>
      </div>
      <p class="hint-blurb">${escapeHtml(h.blurb)}</p>
    </div>`;
  }).join("");
}

function renderCompare() {
  fillCompareSelectors();
  const dsIds = selectedOptions(document.getElementById("compareSelect"));
  const metrics = selectedOptions(document.getElementById("compareMetrics"));
  renderMetricHints(metrics);
  syncCompareSortOptions(metrics);
  const sortBy = document.getElementById("compareSortBy")?.value || "name";
  const sortOrder = document.getElementById("compareSortOrder")?.value || "desc";
  let picked = state.rows.filter((r) => dsIds.includes(r.dataset_file));
  picked = sortCompareRows(picked, metrics, sortBy, sortOrder);

  destroyChart("compare");
  const palette = [
    "#0f766e", "#b45309", "#1d4ed8", "#9f1239", "#4d7c0f", "#7c3aed",
    "#0e7490", "#a16207", "#be123c", "#0369a1", "#15803d", "#c2410c",
    "#4338ca", "#a21caf", "#0f766e", "#854d0e",
  ];

  const status = document.getElementById("compareStatus");
  const chartBox = document.getElementById("compareChartBox");

  if (!picked.length || !metrics.length) {
    if (status) status.textContent = "Select at least one dataset and one metric.";
    document.getElementById("compareTableWrap").innerHTML = "<p class='hint'>Select datasets and metrics.</p>";
    return;
  }

  // Many datasets → put datasets on X so every selection is a visible category.
  const datasetsOnX = picked.length >= metrics.length || picked.length > 6;
  const sortLabel = document.getElementById("compareSortBy")?.selectedOptions?.[0]?.textContent || sortBy;
  if (status) {
    status.textContent = datasetsOnX
      ? `Plotting all ${picked.length} datasets × ${metrics.length} metrics · sorted by ${sortLabel} (${sortOrder}).`
      : `Plotting ${picked.length} datasets × ${metrics.length} metrics · sorted by ${sortLabel} (${sortOrder}).`;
  }
  if (chartBox) chartBox.classList.toggle("extra-tall", picked.length > 15);

  let labels;
  let chartDatasets;
  if (datasetsOnX) {
    labels = picked.map((r) => r._label);
    chartDatasets = metrics.map((m, i) => ({
      label: m,
      data: picked.map((r) => num(r[metricCol(m)])),
      backgroundColor: palette[i % palette.length],
      borderRadius: 4,
    }));
  } else {
    labels = metrics;
    chartDatasets = picked.map((r, i) => ({
      label: r._label,
      data: metrics.map((m) => num(r[metricCol(m)])),
      backgroundColor: palette[i % palette.length],
      borderRadius: 6,
    }));
  }

  state.charts.compare = new Chart(document.getElementById("compareBars"), {
    type: "bar",
    data: { labels, datasets: chartDatasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { boxWidth: 12, font: { size: chartDatasets.length > 10 ? 9 : 12 } },
        },
      },
      scales: {
        x: {
          ticks: {
            maxRotation: 75,
            minRotation: datasetsOnX ? 45 : 0,
            autoSkip: false,
            font: { size: picked.length > 40 ? 8 : 11 },
          },
        },
        y: { beginAtZero: true },
      },
    },
  });

  const head = ["Metric", ...picked.map((r) => r._label)];
  const body = metrics.map((m) => {
    const cells = picked.map((r) => formatNum(num(r[metricCol(m)])));
    return `<tr><td>${escapeHtml(m)}</td>${cells.map((c) => `<td>${escapeHtml(c)}</td>`).join("")}</tr>`;
  }).join("");

  const sizeRows = [
    ["Rows", ...picked.map((r) => formatNum(num(r.n_rows_original)))],
    ["Columns", ...picked.map((r) => formatNum(num(r.n_columns_original)))],
    ["Features", ...picked.map((r) => formatNum(num(r.n_features_after_encoding)))],
    ["Classes", ...picked.map((r) => formatNum(num(r.n_classes)))],
  ].map((row) => `<tr>${row.map((c) => `<td>${escapeHtml(c)}</td>`).join("")}</tr>`).join("");

  document.getElementById("compareTableWrap").innerHTML = `
    <table>
      <thead><tr>${head.map((h) => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead>
      <tbody>${sizeRows}${body}</tbody>
    </table>`;
}

function fillDistVariable() {
  const sel = document.getElementById("distVariable");
  if (sel.options.length) return;
  sel.innerHTML = [
    ...SIZE_VARS.map((v) => `<option value="${v.key}">${v.label}</option>`),
    ...STANDARD_METRICS.map((m) => `<option value="pycol_${m}">pycol_${m}</option>`),
  ].join("");
  sel.value = "n_rows_original";
}

function histogram(values, bins) {
  if (!values.length) return { labels: [], counts: [] };
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) return { labels: [formatNum(min)], counts: [values.length] };
  const width = (max - min) / bins;
  const counts = Array(bins).fill(0);
  const labels = [];
  for (let i = 0; i < bins; i += 1) {
    const a = min + i * width;
    const b = a + width;
    labels.push(`${formatNum(a)}–${formatNum(b)}`);
  }
  for (const v of values) {
    let idx = Math.floor((v - min) / width);
    if (idx >= bins) idx = bins - 1;
    if (idx < 0) idx = 0;
    counts[idx] += 1;
  }
  return { labels, counts };
}

function renderDistributions() {
  fillDistVariable();
  const key = document.getElementById("distVariable").value;
  const bins = Math.max(5, Math.min(40, Number(document.getElementById("distBins").value) || 16));
  const values = state.rows.map((r) => num(r[key])).filter((v) => v !== null);
  const { labels, counts } = histogram(values, bins);

  document.getElementById("distStats").innerHTML = [
    kpi("N values", values.length, `of ${state.rows.length} datasets`),
    kpi("Min", formatNum(values.length ? Math.min(...values) : null), key),
    kpi("Median", formatNum(median(values)), ""),
    kpi("Max", formatNum(values.length ? Math.max(...values) : null), ""),
  ].join("");

  document.getElementById("distLead").textContent =
    `Distribution of ${key} across datasets with a finite value.`;

  destroyChart("dist");
  state.charts.dist = new Chart(document.getElementById("distHist"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: counts,
        backgroundColor: "#0f766e",
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { maxRotation: 60, autoSkip: true, maxTicksLimit: 12 } },
        y: { title: { display: true, text: "Datasets" }, ticks: { precision: 0 } },
      },
    },
  });
}

function metricHardnessDirection(metric) {
  const hints = (typeof METRIC_HINTS !== "undefined") ? METRIC_HINTS : {};
  const h = hints[metric];
  if (!h) return "higher";
  if (h.certainty === "context" || /context/i.test(h.hard || "")) return "context";
  if (/less complex/i.test(h.hard || "")) return "lower";
  return "higher";
}

function percentileMap(valuesById) {
  const entries = Object.entries(valuesById).filter(([, v]) => v !== null);
  if (!entries.length) return {};
  const sorted = [...entries].sort((a, b) => a[1] - b[1]);
  const n = sorted.length;
  const out = {};
  if (n === 1) {
    out[sorted[0][0]] = 0.5;
    return out;
  }
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && sorted[j + 1][1] === sorted[i][1]) j += 1;
    const avgRank = ((i + j) / 2) / (n - 1);
    for (let k = i; k <= j; k += 1) out[sorted[k][0]] = avgRank;
    i = j + 1;
  }
  return out;
}

function computeHardnessTable(rows, metrics) {
  const perMetric = {};
  for (const m of metrics) {
    const raw = {};
    for (const r of rows) {
      raw[r.dataset_file] = num(r[metricCol(m)]);
    }
    const ranks = percentileMap(raw);
    const dir = metricHardnessDirection(m);
    const hard = {};
    for (const id of Object.keys(ranks)) {
      let h = ranks[id];
      if (dir === "lower") h = 1 - h;
      else if (dir === "context") h = Math.abs(h - 0.5) * 2;
      hard[id] = h;
    }
    perMetric[m] = hard;
  }

  return rows.map((r) => {
    const scores = {};
    const used = [];
    for (const m of metrics) {
      const h = perMetric[m][r.dataset_file];
      if (h === undefined || Number.isNaN(h)) continue;
      scores[m] = h;
      used.push(h);
    }
    const composite = used.length ? used.reduce((a, b) => a + b, 0) / used.length : null;
    return {
      row: r,
      scores,
      composite,
      nMetrics: used.length,
    };
  }).filter((x) => x.composite !== null);
}

function heatColor(t) {
  // 0 easy (teal) → 0.5 mid (amber) → 1 hard (red)
  const x = Math.max(0, Math.min(1, t));
  let r;
  let g;
  let b;
  if (x < 0.5) {
    const u = x / 0.5;
    r = Math.round(15 + u * (180 - 15));
    g = Math.round(118 + u * (120 - 118));
    b = Math.round(110 + u * (9 - 110));
  } else {
    const u = (x - 0.5) / 0.5;
    r = Math.round(180 + u * (185 - 180));
    g = Math.round(120 + u * (28 - 120));
    b = Math.round(9 + u * (28 - 9));
  }
  return `rgb(${r},${g},${b})`;
}

function fillGlanceSelectors() {
  const mSel = document.getElementById("glanceMetrics");
  const dSel = document.getElementById("glanceRadarDs");
  if (mSel && !mSel.options.length) {
    const defaults = ["F1", "N1", "N2", "N3", "borderline", "LSC", "C1", "purity"];
    mSel.innerHTML = STANDARD_METRICS.map((m) =>
      `<option value="${escapeHtml(m)}" ${defaults.includes(m) ? "selected" : ""}>${escapeHtml(metricListLabel(m))}</option>`
    ).join("");
  }
  if (dSel && !dSel.options.length) {
    const sorted = [...state.rows].sort((a, b) => a._label.localeCompare(b._label));
    dSel.innerHTML = sorted.map((r) =>
      `<option value="${escapeHtml(r.dataset_file)}">${escapeHtml(r._label)}</option>`
    ).join("");
    const radarDefaults = ["iris.csv", "wine_quality_red.csv", "ecoli.csv", "yeast.csv"].filter((d) =>
      sorted.some((r) => r.dataset_file === d)
    );
    [...dSel.options].forEach((o) => { o.selected = radarDefaults.includes(o.value); });
  }
}

function renderGlance() {
  fillGlanceSelectors();
  const metrics = selectedOptions(document.getElementById("glanceMetrics"));
  const topN = Number(document.getElementById("glanceTopN")?.value ?? 40);
  const order = document.getElementById("glanceOrder")?.value || "desc";
  const status = document.getElementById("glanceStatus");

  if (!metrics.length) {
    if (status) status.textContent = "Select at least one metric.";
    return;
  }

  let scored = computeHardnessTable(state.rows, metrics);
  scored.sort((a, b) => (order === "asc" ? a.composite - b.composite : b.composite - a.composite));
  const shown = topN > 0 ? scored.slice(0, topN) : scored;

  if (status) {
    status.textContent =
      `Composite hardness from ${metrics.length} metrics · showing ${shown.length} of ${scored.length} datasets ` +
      `(${order === "desc" ? "hardest first" : "easiest first"}). Context metrics use distance-from-median.`;
  }

  // 1) Bars
  destroyChart("glanceBars");
  const barLabels = shown.map((s) => s.row._label);
  const barData = shown.map((s) => s.composite);
  state.charts.glanceBars = new Chart(document.getElementById("glanceBars"), {
    type: "bar",
    data: {
      labels: barLabels,
      datasets: [{
        label: "Composite hardness",
        data: barData,
        backgroundColor: barData.map((v) => heatColor(v)),
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          min: 0,
          max: 1,
          title: { display: true, text: "Hardness (0 easy → 1 hard)" },
        },
        y: {
          ticks: {
            autoSkip: false,
            font: { size: shown.length > 50 ? 8 : 11 },
          },
        },
      },
    },
  });

  // 2) Heatmap
  const heat = document.getElementById("glanceHeatmap");
  if (heat) {
    const head = ["Dataset", "Composite", ...metrics];
    const body = shown.map((s) => {
      const cells = metrics.map((m) => {
        const v = s.scores[m];
        if (v === undefined) return "<td class='cell'>—</td>";
        return `<td class="cell" style="background:${heatColor(v)}" title="${escapeHtml(m)}=${v.toFixed(3)}">${v.toFixed(2)}</td>`;
      }).join("");
      return `<tr>
        <td class="ds">${escapeHtml(s.row._label)}</td>
        <td class="cell" style="background:${heatColor(s.composite)}">${s.composite.toFixed(2)}</td>
        ${cells}
      </tr>`;
    }).join("");
    heat.innerHTML = `<table class="heatmap">
      <thead><tr>${head.map((h, i) => `<th class="${i >= 2 ? "metric" : ""}">${escapeHtml(h)}</th>`).join("")}</tr></thead>
      <tbody>${body}</tbody>
    </table>`;
  }

  // 3) Radar — raw values for picked datasets
  destroyChart("glanceRadar");
  const radarIds = selectedOptions(document.getElementById("glanceRadarDs")).slice(0, 8);
  const radarRows = state.rows.filter((r) => radarIds.includes(r.dataset_file));
  const palette = [
    "#0f766e", "#b45309", "#1d4ed8", "#9f1239", "#4d7c0f", "#7c3aed", "#0e7490", "#a16207",
  ];

  if (!radarRows.length || !metrics.length) {
    return;
  }

  // Normalize each metric to [0,1] across ALL datasets for fair radar spokes
  const norm = {};
  for (const m of metrics) {
    const vals = state.rows.map((r) => num(r[metricCol(m)])).filter((v) => v !== null);
    const lo = vals.length ? Math.min(...vals) : 0;
    const hi = vals.length ? Math.max(...vals) : 1;
    norm[m] = { lo, hi: hi === lo ? lo + 1 : hi };
  }

  state.charts.glanceRadar = new Chart(document.getElementById("glanceRadar"), {
    type: "radar",
    data: {
      labels: metrics,
      datasets: radarRows.map((r, i) => ({
        label: r._label,
        data: metrics.map((m) => {
          const v = num(r[metricCol(m)]);
          if (v === null) return null;
          const { lo, hi } = norm[m];
          return (v - lo) / (hi - lo);
        }),
        borderColor: palette[i % palette.length],
        backgroundColor: palette[i % palette.length] + "33",
        pointBackgroundColor: palette[i % palette.length],
        borderWidth: 2,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom" },
        tooltip: {
          callbacks: {
            label(ctx) {
              const m = metrics[ctx.dataIndex];
              const row = radarRows[ctx.datasetIndex];
              const raw = num(row[metricCol(m)]);
              return `${ctx.dataset.label}: raw=${formatNum(raw)} · scaled=${ctx.parsed.r?.toFixed?.(2) ?? ctx.raw}`;
            },
          },
        },
      },
      scales: {
        r: {
          min: 0,
          max: 1,
          ticks: { display: false },
          pointLabels: { font: { size: 10 } },
        },
      },
    },
  });
}

function renderCoverage() {
  const rows = state.rows;
  const cheapDone = rows.filter((r) => r.cheap_fill === CHEAP_METRICS.length);
  const stdDone = rows.filter((r) => r.std_fill === STANDARD_METRICS.length);
  const onlyMrca = rows.filter((r) => r.missing_cheap.length === 1 && r.missing_cheap[0] === "MRCA");
  const otherCheap = rows.filter((r) => r.missing_cheap.length && !(r.missing_cheap.length === 1 && r.missing_cheap[0] === "MRCA"));
  const onlyOnbDbc = rows.filter((r) =>
    r.cheap_fill === CHEAP_METRICS.length &&
    r.missing_std.length &&
    r.missing_std.every((m) => STANDARD_EXTRA.includes(m))
  );

  document.getElementById("coverageKpis").innerHTML = [
    kpi("Cheap done", `${cheapDone.length}/${rows.length}`, `${onlyMrca.length} missing only MRCA`),
    kpi("Standard done", `${stdDone.length}/${rows.length}`, `${onlyOnbDbc.length} missing only ONB/DBC`),
    kpi("Other cheap gaps", otherCheap.length, "e.g. F1v + MRCA"),
    kpi("Failed marker", rows.filter((r) => present(r.pycol_metrics_failed)).length, "pycol_metrics_failed set"),
  ].join("");

  const cheapMiss = rows.filter((r) => r.missing_cheap.length)
    .sort((a, b) => a._label.localeCompare(b._label));
  document.getElementById("missingCheapWrap").innerHTML = `
    <table>
      <thead><tr><th>Dataset</th><th>Fill</th><th>Missing</th><th>Failed</th></tr></thead>
      <tbody>
        ${cheapMiss.map((r) => `<tr>
          <td>${escapeHtml(r._label)}</td>
          <td>${pillFor(r.cheap_fill, CHEAP_METRICS.length)}</td>
          <td>${escapeHtml(r.missing_cheap.join(", "))}</td>
          <td>${escapeHtml(present(r.pycol_metrics_failed) ? r.pycol_metrics_failed : "—")}</td>
        </tr>`).join("") || "<tr><td colspan='4'>None</td></tr>"}
      </tbody>
    </table>`;

  document.getElementById("missingStdWrap").innerHTML = `
    <table>
      <thead><tr><th>Dataset</th><th>Fill</th><th>Missing</th></tr></thead>
      <tbody>
        ${onlyOnbDbc.map((r) => `<tr>
          <td>${escapeHtml(r._label)}</td>
          <td>${pillFor(r.std_fill, STANDARD_METRICS.length)}</td>
          <td>${escapeHtml(r.missing_std.join(", "))}</td>
        </tr>`).join("") || "<tr><td colspan='3'>None</td></tr>"}
      </tbody>
    </table>`;
}

function wireUi() {
  document.querySelectorAll(".menu-item").forEach((btn) => {
    btn.addEventListener("click", () => showView(btn.dataset.view));
  });
  document.getElementById("navToggle")?.addEventListener("click", openNav);
  document.getElementById("backdrop")?.addEventListener("click", closeNav);

  ["browseSearch", "browseSort", "browseOrder"].forEach((id) => {
    document.getElementById(id)?.addEventListener("input", renderBrowse);
    document.getElementById(id)?.addEventListener("change", renderBrowse);
  });
  ["compareSelect", "compareMetrics", "compareSortBy", "compareSortOrder"].forEach((id) => {
    document.getElementById(id)?.addEventListener("change", renderCompare);
  });
  document.getElementById("compareSelectAll")?.addEventListener("click", () => {
    const sel = document.getElementById("compareSelect");
    [...sel.options].forEach((o) => { o.selected = true; });
    renderCompare();
  });
  document.getElementById("compareClearDs")?.addEventListener("click", () => {
    const sel = document.getElementById("compareSelect");
    [...sel.options].forEach((o) => { o.selected = false; });
    renderCompare();
  });
  document.getElementById("compareSelectAllMetrics")?.addEventListener("click", () => {
    const sel = document.getElementById("compareMetrics");
    [...sel.options].forEach((o) => { o.selected = true; });
    renderCompare();
  });
  document.getElementById("compareClearMetrics")?.addEventListener("click", () => {
    const sel = document.getElementById("compareMetrics");
    [...sel.options].forEach((o) => { o.selected = false; });
    renderCompare();
  });
  ["glanceMetrics", "glanceRadarDs", "glanceTopN", "glanceOrder"].forEach((id) => {
    document.getElementById(id)?.addEventListener("change", renderGlance);
  });
  document.getElementById("glanceMetricsDefault")?.addEventListener("click", () => {
    const sel = document.getElementById("glanceMetrics");
    const defaults = new Set(["F1", "N1", "N2", "N3", "borderline", "LSC", "C1", "purity"]);
    [...sel.options].forEach((o) => { o.selected = defaults.has(o.value); });
    renderGlance();
  });
  document.getElementById("glanceMetricsAll")?.addEventListener("click", () => {
    const sel = document.getElementById("glanceMetrics");
    [...sel.options].forEach((o) => { o.selected = true; });
    renderGlance();
  });
  ["distVariable", "distBins"].forEach((id) => {
    document.getElementById(id)?.addEventListener("change", renderDistributions);
    document.getElementById(id)?.addEventListener("input", renderDistributions);
  });
}

function loadCsv() {
  const url = "data/datasets_complexity_summary.csv";
  setStatus("Loading CSV…");
  Papa.parse(url, {
    download: true,
    header: true,
    dynamicTyping: false,
    skipEmptyLines: true,
    complete(results) {
      if (results.errors?.length) {
        console.warn(results.errors);
      }
      state.rows = enrich(results.data || []);
      setStatus(`${state.rows.length} datasets loaded`);
      showView("overview");
    },
    error(err) {
      console.error(err);
      setStatus("Failed to load CSV. Serve this folder over HTTP (see README).");
    },
  });
}

document.addEventListener("DOMContentLoaded", () => {
  wireUi();
  document.getElementById("gotoGlance")?.addEventListener("click", () => showView("glance"));
  loadCsv();
});
