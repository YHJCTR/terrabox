const $ = (id) => document.getElementById(id);

const state = {
  experiments: [],
  lastPayload: null,
};

const metricRows = [
  ["success_rate", "success_rate", "%"],
  ["set-F1", "set_f1", ""],
  ["multiset-F1", "multiset_f1", ""],
  ["exact_match", "exact_match", "%"],
  ["ordered_exact", "ordered_exact", "%"],
  ["AnyOrder", "any_order", "%"],
  ["SameOrder", "same_order", "%"],
  ["Unique", "unique", "%"],
  ["F1 perception", "f1_perception", ""],
  ["F1 operation", "f1_operation", ""],
  ["F1 logic", "f1_logic", ""],
  ["F1 gis", "f1_gis", ""],
  ["empty_rate", "empty_rate", "%"],
  ["cap_rate", "cap_rate", "%"],
  ["same-tool>=4 tasks", "repeat4_tasks", ""],
  ["errors/task", "errors_per_task", ""],
  ["tools/task", "tools_per_task", ""],
  ["llm/task", "llm_per_task", ""],
  ["tokens/task", "tokens_per_task", ""],
  ["time/task", "time_per_task", "s"],
];

function formatValue(key, value) {
  if (value === undefined || value === null || Number.isNaN(value)) return "-";
  if (["set_f1", "multiset_f1", "set_precision", "set_recall", "multiset_precision", "multiset_recall", "lcs_ratio"].includes(key)) {
    return Number(value).toFixed(3);
  }
  if (["repeat4_tasks", "tokens_per_task"].includes(key)) return Number(value).toFixed(0);
  if (key === "time_per_task") return Number(value).toFixed(1);
  return Number(value).toFixed(2);
}

function setNotice(text, isError = false) {
  const notice = $("notice");
  if (!text) {
    notice.classList.add("hidden");
    notice.textContent = "";
    return;
  }
  notice.classList.remove("hidden");
  notice.textContent = text;
  notice.style.borderColor = isError ? "#f0b4ae" : "#f3d19e";
  notice.style.background = isError ? "#fff1f0" : "#fff7e6";
  notice.style.color = isError ? "#b42318" : "#9a6700";
}

async function getJson(url) {
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

function optionLabel(exp) {
  return `${exp.name}  [${exp.kind}]`;
}

async function loadExperiments() {
  setNotice("正在扫描实验目录...");
  const data = await getJson("/api/experiments");
  state.experiments = data.experiments;
  for (const select of [$("current"), $("baseline")]) {
    select.innerHTML = "";
    for (const exp of state.experiments) {
      const opt = document.createElement("option");
      opt.value = exp.results_dir;
      opt.textContent = optionLabel(exp);
      select.appendChild(opt);
    }
  }
  if (state.experiments.length > 1) $("baseline").selectedIndex = 1;
  setNotice(`已发现 ${state.experiments.length} 个可读取 results/*.json 的实验。`);
}

function renderCards(summary, meta = {}) {
  const cards = [
    ["任务数", meta.n || summary.n, ""],
    ["完成进度", meta.progress || "", ""],
    ["success_rate", formatValue("success_rate", summary.success_rate), "%"],
    ["set-F1", formatValue("set_f1", summary.set_f1), ""],
    ["multiset-F1", formatValue("multiset_f1", summary.multiset_f1), ""],
    ["AnyOrder", formatValue("any_order", summary.any_order), "%"],
    ["F1 gis", formatValue("f1_gis", summary.f1_gis), ""],
    ["errors/task", formatValue("errors_per_task", summary.errors_per_task), ""],
  ];
  $("cards").innerHTML = cards.map(([label, value, suffix]) => `
    <div class="card">
      <div class="label">${label}</div>
      <div class="value">${value}${suffix}</div>
    </div>
  `).join("");
}

function renderSingleTable(summary) {
  $("table").innerHTML = `
    <table>
      <thead><tr><th>metric</th><th>value</th></tr></thead>
      <tbody>
        ${metricRows.map(([label, key, suffix]) => `
          <tr><td>${label}</td><td>${formatValue(key, summary[key])}${suffix}</td></tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderCompareTable(report) {
  $("table").innerHTML = `
    <table>
      <thead><tr><th>metric</th><th>baseline</th><th>current</th><th>delta</th></tr></thead>
      <tbody>
        ${metricRows.map(([label, key, suffix]) => {
          const delta = report.delta[key];
          const cls = delta > 0 ? "delta-up" : delta < 0 ? "delta-down" : "delta-neutral";
          return `<tr>
            <td>${label}</td>
            <td>${formatValue(key, report.base[key])}${suffix}</td>
            <td>${formatValue(key, report.cur[key])}${suffix}</td>
            <td class="${cls}">${delta >= 0 ? "+" : ""}${formatValue(key, delta)}${suffix}</td>
          </tr>`;
        }).join("")}
        <tr><td>success flips wrong→right</td><td></td><td>${report.success_flips_up}</td><td></td></tr>
        <tr><td>success flips right→wrong</td><td></td><td>${report.success_flips_down}</td><td></td></tr>
        <tr><td>success flips net</td><td></td><td>${report.success_flips_net}</td><td></td></tr>
      </tbody>
    </table>
  `;
}

function drawF1Chart(summaryA, summaryB = null) {
  const canvas = $("f1Chart");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || canvas.parentElement.clientWidth;
  const height = 180;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  const labels = ["perception", "operation", "logic", "gis"];
  const keys = ["f1_perception", "f1_operation", "f1_logic", "f1_gis"];
  const left = 62;
  const bottom = height - 28;
  const chartW = width - left - 18;
  const chartH = height - 46;
  ctx.strokeStyle = "#d9dee7";
  ctx.beginPath();
  ctx.moveTo(left, 12);
  ctx.lineTo(left, bottom);
  ctx.lineTo(width - 10, bottom);
  ctx.stroke();
  ctx.fillStyle = "#5f6b7a";
  ctx.font = "12px sans-serif";
  [0, 25, 50, 75, 100].forEach(v => {
    const y = bottom - (v / 100) * chartH;
    ctx.fillText(String(v), 12, y + 4);
    ctx.strokeStyle = "#eef1f5";
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(width - 10, y);
    ctx.stroke();
  });
  const groupW = chartW / labels.length;
  keys.forEach((key, i) => {
    const x = left + i * groupW + 12;
    const v = Number(summaryA[key] || 0);
    const h = (v / 100) * chartH;
    ctx.fillStyle = "#1d4ed8";
    ctx.fillRect(x, bottom - h, summaryB ? 18 : 30, h);
    if (summaryB) {
      const vb = Number(summaryB[key] || 0);
      const hb = (vb / 100) * chartH;
      ctx.fillStyle = "#087f5b";
      ctx.fillRect(x + 22, bottom - hb, 18, hb);
    }
    ctx.fillStyle = "#17202a";
    ctx.save();
    ctx.translate(left + i * groupW + groupW / 2, height - 6);
    ctx.rotate(-0.25);
    ctx.fillText(labels[i], -28, 0);
    ctx.restore();
  });
}

async function loadMetrics() {
  setNotice("");
  $("promptPanel").classList.add("hidden");
  const mode = $("mode").value;
  if (mode === "single") {
    const params = new URLSearchParams({
      experiment: $("current").value,
      scope: $("scope").value,
      total: "1162",
    });
    const data = await getJson(`/api/status?${params}`);
    const summary = data.summary;
    state.lastPayload = data;
    renderCards(summary, {
      n: `${summary.n}`,
      progress: `${data.done}/${data.total} (${data.progress_pct.toFixed(1)}%)`,
    });
    renderSingleTable(summary);
    drawF1Chart(summary);
    setNotice(`单实验：${data.experiment}，结果目录：${data.results_dir}`);
    return;
  }
  const params = new URLSearchParams({
    current: $("current").value,
    baseline: $("baseline").value,
  });
  const data = await getJson(`/api/compare?${params}`);
  state.lastPayload = data;
  renderCards(data.cur, {
    n: `${data.n_common}`,
    progress: `交集 ${data.n_common}`,
  });
  renderCompareTable(data);
  drawF1Chart(data.cur, data.base);
  setNotice(`双实验交集对比：n=${data.n_common}，current=${data.current.name}，baseline=${data.baseline.name}`);
}

async function showPrompt() {
  const params = new URLSearchParams({ experiment: $("current").value });
  const data = await getJson(`/api/prompt?${params}`);
  $("promptPanel").classList.remove("hidden");
  $("promptMeta").textContent = data.found ? data.prompt_path : "未找到静态提示词文件";
  $("promptText").textContent = data.prompt || "(empty)";
}

$("mode").addEventListener("change", () => {
  $("baselineBox").classList.toggle("hidden", $("mode").value !== "compare");
});
$("refresh").addEventListener("click", () => loadExperiments().catch(e => setNotice(e.message, true)));
$("load").addEventListener("click", () => loadMetrics().catch(e => setNotice(e.message, true)));
$("showPrompt").addEventListener("click", () => showPrompt().catch(e => setNotice(e.message, true)));

loadExperiments().then(loadMetrics).catch(e => setNotice(e.message, true));

