/* AI Hub Model Explorer: efficiency map of one snapshot (data from assets/data.js). */
(function () {
  "use strict";

  var D = window.AIHUB_DATA;
  var $ = function (id) { return document.getElementById(id); };
  var PARAMS = "__params";
  var WORLD_ORDER = ["Mobile", "Compute", "Automotive", "IoT", "XR"];
  var RUNTIME_LABEL = {
    tflite: "TFLite", qnn_dlc: "QNN DLC", qnn_context_binary: "QNN context binary", onnx: "ONNX Runtime",
    precompiled_qnn_onnx: "Precompiled QNN ONNX", voice_ai: "Voice AI", genie: "Genie",
    geniex_qairt: "GenieX (QAIRT)", geniex_llamacpp: "GenieX (llama.cpp)"
  };
  var DEFAULTS = {
    view: "models", world: "Mobile", device: "Samsung Galaxy S25", task: "Image Classification",
    metric: "", runtime: "auto", precision: "quant", context: "max", xaxis: "fps", pin: ""
  };

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function fmt(v, digits) {
    if (v == null || isNaN(v)) return "—";
    var d = digits == null ? (Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : 2) : digits;
    return Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  }
  function fmtParams(p) {
    if (!p) return "—";
    if (p >= 1e9) return fmt(p / 1e9, 2) + " B";
    if (p >= 1e6) return fmt(p / 1e6, p >= 1e8 ? 0 : 1) + " M";
    return fmt(p / 1e3, 0) + " K";
  }
  function runtimeLabel(r) { return RUNTIME_LABEL[r] || r; }
  function uniq(list) { return Array.from(new Set(list)); }

  /* ---------- indexes ---------- */

  var models = D.models, devices = D.devices;
  var PERF = {}, LLM = {}, COMPS = {};
  D.perf.rows.forEach(function (r) {
    var k = r[0] + "|" + r[1];
    (PERF[k] = PERF[k] || []).push(r);
    var c = r[0] + "|" + r[2];
    (COMPS[c] = COMPS[c] || new Set()).add(r[3]);
  });
  D.llm_perf.rows.forEach(function (r) {
    var k = r[0] + "|" + r[1];
    (LLM[k] = LLM[k] || []).push(r);
  });
  var deviceIndex = {};
  devices.forEach(function (d, i) { deviceIndex[d.name] = i; });

  function inView(m, view) { return view === "llm" ? m.is_llm : !m.is_llm; }
  function hasData(mi, di, view) { return !!(view === "llm" ? LLM : PERF)[mi + "|" + di]; }

  function metricKey(m) { return m.accuracy ? m.accuracy.metric + " [" + m.accuracy.unit + "]" : null; }

  /* ---------- picking one result per model ---------- */

  function aggregate(g) {
    var lat = 0, memMin = 0, memMax = 0, npu = 0, gpu = 0, cpu = 0, total = 0, main = null;
    g.rows.forEach(function (r) {
      lat += r[5];
      memMin = Math.max(memMin, r[6] || 0);
      memMax = Math.max(memMax, r[7] || 0);
      npu += r[8]; gpu += r[9]; cpu += r[10]; total += r[11];
      if (!main || r[5] > main[5]) main = r;
    });
    return {
      precision: g.precision, runtime: g.runtime, latency: lat, fps: 1000 / lat, mem: [memMin, memMax],
      layers: { npu: npu, gpu: gpu, cpu: cpu, total: total }, cu: main[12], qairt: main[13], ort: main[14],
      components: g.rows.map(function (r) { return r[3]; })
    };
  }

  function perfGroups(mi, di) {
    var out = {};
    (PERF[mi + "|" + di] || []).forEach(function (r) {
      var k = r[2] + "|" + r[4];
      (out[k] = out[k] || { precision: r[2], runtime: r[4], rows: [] }).rows.push(r);
    });
    return Object.keys(out).map(function (k) { return out[k]; }).filter(function (g) {
      return g.rows.length >= COMPS[mi + "|" + g.precision].size;
    }).map(aggregate);
  }

  function accuracyFor(m, precision, runtime) {
    var a = m.accuracy;
    if (!a) return null;
    var v = a.values.filter(function (x) { return x.precision === precision && x.runtime === runtime; })[0];
    if (v) return { value: v.value, device: v.device, runtime: v.runtime, exact: true };
    v = a.values.filter(function (x) { return x.precision === precision; })[0];
    if (v) return { value: v.value, device: v.device, runtime: v.runtime, exact: false };
    if (precision === "float" && a.torch != null) {
      return { value: a.torch, device: null, runtime: "PyTorch reference", exact: false, torch: true };
    }
    return null;
  }

  function choose(cands, better) {
    // Prefer results with an accuracy measured on the same runtime, then any accuracy, then fastest.
    var exact = cands.filter(function (c) { return c.acc && c.acc.exact; });
    var withAcc = cands.filter(function (c) { return c.acc; });
    var pool = exact.length ? exact : withAcc.length ? withAcc : cands;
    pool.sort(better);
    return pool[0] || null;
  }

  function pickModel(m, mi, di, precision, runtime) {
    var gs = perfGroups(mi, di).filter(function (g) {
      if (runtime !== "auto" && g.runtime !== runtime) return false;
      return precision === "quant" ? g.precision !== "float" : g.precision === precision;
    });
    gs.forEach(function (g) { g.acc = accuracyFor(m, g.precision, g.runtime); });
    return choose(gs, function (a, b) { return a.latency - b.latency; });
  }

  function pickLLM(m, mi, di, runtime, context) {
    var rows = (LLM[mi + "|" + di] || []).filter(function (r) { return runtime === "auto" || r[4] === runtime; });
    if (context === "max") {
      var best = {};
      rows.forEach(function (r) {
        var k = r[2] + "|" + r[4];
        if (!best[k] || r[5] > best[k][5]) best[k] = r;
      });
      rows = Object.keys(best).map(function (k) { return best[k]; });
    } else {
      rows = rows.filter(function (r) { return String(r[5]) === String(context); });
    }
    var cands = rows.map(function (r) {
      return {
        precision: r[2], runtime: r[4], context: r[5], tps: r[6], prefill: r[7], ttft: [r[8], r[9]], cu: r[10],
        acc: accuracyFor(m, r[2], r[4])
      };
    });
    return choose(cands, function (a, b) { return b.tps - a.tps; });
  }

  /* ---------- state ---------- */

  var state = Object.assign({}, DEFAULTS, { labels: true, excluded: new Set(), search: "" });
  var sort = { key: null, asc: true };
  var pinned = null;

  function readHash() {
    var h = decodeURIComponent(location.hash.replace(/^#/, ""));
    h.split("&").forEach(function (kv) {
      var i = kv.indexOf("=");
      if (i > 0 && kv.slice(0, i) in DEFAULTS) state[kv.slice(0, i)] = kv.slice(i + 1);
    });
  }
  function writeHash() {
    var parts = Object.keys(DEFAULTS).map(function (k) { return k + "=" + encodeURIComponent(state[k]); });
    history.replaceState(null, "", "#" + parts.join("&"));
  }

  function setOptions(select, options, value) {
    select.innerHTML = options.map(function (o) {
      return '<option value="' + esc(o.value) + '">' + esc(o.label) + "</option>";
    }).join("");
    var values = options.map(function (o) { return String(o.value); });
    select.value = values.indexOf(String(value)) >= 0 ? value : (values[0] || "");
    return select.value;
  }

  /* ---------- controls ---------- */

  function viewModels() {
    return models.map(function (m, i) { return { m: m, i: i }; }).filter(function (x) { return inView(x.m, state.view); });
  }

  function refreshControls() {
    var vm = viewModels();
    document.querySelectorAll(".tab").forEach(function (t) { t.classList.toggle("active", t.dataset.view === state.view); });
    document.querySelectorAll("[data-only]").forEach(function (el) { el.hidden = el.dataset.only !== state.view; });

    // Platform and device: only devices that have results for this view.
    var devOk = devices.map(function (d, di) {
      return vm.some(function (x) { return hasData(x.i, di, state.view); });
    });
    var worlds = WORLD_ORDER.filter(function (w) {
      return devices.some(function (d, di) { return devOk[di] && d.world === w; });
    });
    state.world = setOptions($("world"), worlds.map(function (w) { return { value: w, label: w }; }), state.world);
    var devOpts = devices.map(function (d, di) { return { d: d, di: di }; }).filter(function (x) {
      return devOk[x.di] && x.d.world === state.world;
    }).map(function (x) {
      return { value: x.d.name, label: x.d.name + (x.d.borrowed_from ? " *" : "") };
    });
    state.device = setOptions($("device"), devOpts, state.device);
    var di = deviceIndex[state.device];

    // Task (models view only).
    var scope = vm.filter(function (x) { return hasData(x.i, di, state.view); });
    if (state.view === "models") {
      var counts = {};
      scope.forEach(function (x) { counts[x.m.use_case] = (counts[x.m.use_case] || 0) + 1; });
      var tasks = Object.keys(counts).sort();
      state.task = setOptions($("task"), tasks.map(function (t) {
        return { value: t, label: t + " (" + counts[t] + ")" };
      }), state.task);
      scope = scope.filter(function (x) { return x.m.use_case === state.task; });
    }

    // Metric: accuracy metrics present in scope (most common first), then parameter count.
    var mcount = {};
    scope.forEach(function (x) { var k = metricKey(x.m); if (k) mcount[k] = (mcount[k] || 0) + 1; });
    var metrics = Object.keys(mcount).sort(function (a, b) { return mcount[b] - mcount[a]; });
    var metricOpts = metrics.map(function (k) { return { value: k, label: k + " (" + mcount[k] + ")" }; });
    metricOpts.push({ value: PARAMS, label: "Parameters (all models, no accuracy)" });
    state.metric = setOptions($("metric"), metricOpts, state.metric || metrics[0] || PARAMS);

    // Runtime / precision / context available in scope on this device.
    var rows = [];
    scope.forEach(function (x) { rows = rows.concat((state.view === "llm" ? LLM : PERF)[x.i + "|" + di] || []); });
    var runtimes = uniq(rows.map(function (r) { return r[4]; })).sort();
    state.runtime = setOptions($("runtime"), [{ value: "auto", label: "Fastest available" }].concat(
      runtimes.map(function (r) { return { value: r, label: runtimeLabel(r) }; })), state.runtime);
    if (state.view === "models") {
      var precs = uniq(rows.map(function (r) { return r[2]; })).sort();
      state.precision = setOptions($("precision"), [
        { value: "quant", label: "Quantized (fastest with accuracy)" },
        { value: "float", label: "Float" },
        { value: "both", label: "Float and quantized (linked)" }
      ].concat(precs.filter(function (p) { return p !== "float"; }).map(function (p) {
        return { value: p, label: "Only " + p };
      })), state.precision);
      $("xaxis").value = state.xaxis;
    } else {
      var ctxs = uniq(rows.map(function (r) { return r[5]; })).sort(function (a, b) { return a - b; });
      state.context = setOptions($("context"), [{ value: "max", label: "Largest available" }].concat(
        ctxs.map(function (c) { return { value: String(c), label: c + " tokens" }; })), state.context);
    }

    // Model list (models in task scope, any metric).
    var q = state.search.toLowerCase();
    var list = scope.slice().sort(function (a, b) { return a.m.name.localeCompare(b.m.name); });
    $("model-list").innerHTML = list.filter(function (x) {
      return !q || x.m.name.toLowerCase().indexOf(q) >= 0 || x.m.id.indexOf(q) >= 0;
    }).map(function (x) {
      var dim = state.metric !== PARAMS && metricKey(x.m) !== state.metric;
      return '<label class="' + (dim ? "unavailable" : "") + '" title="' + esc(dim ? "No " + state.metric + " data" : x.m.headline) + '">' +
        '<input type="checkbox" data-id="' + esc(x.m.id) + '"' + (state.excluded.has(x.m.id) ? "" : " checked") + "> " +
        esc(x.m.name) + "</label>";
    }).join("");
    $("model-count").textContent = "(" + list.filter(function (x) { return !state.excluded.has(x.m.id); }).length + "/" + list.length + ")";
    return { scope: scope, di: di };
  }

  /* ---------- chart ---------- */

  function cuColor(cu) {
    var u = String(cu || "").toUpperCase();
    return css(u === "NPU" ? "--npu" : u === "GPU" ? "--gpu" : u === "CPU" ? "--cpu" : "--other");
  }

  function buildPoints(ctx) {
    var di = ctx.di, points = [], floats = [], links = [], table = [];
    var metric = state.metric;
    ctx.scope.forEach(function (x) {
      var m = x.m, mi = x.i;
      var res, fl = null;
      if (state.view === "llm") {
        res = pickLLM(m, mi, di, state.runtime, state.context);
      } else {
        res = pickModel(m, mi, di, state.precision === "both" ? "quant" : state.precision, state.runtime);
        if (state.precision === "both") fl = pickModel(m, mi, di, "float", state.runtime);
      }
      table.push({ m: m, res: res || fl, excluded: state.excluded.has(m.id) });
      if (state.excluded.has(m.id)) return;
      if (metric !== PARAMS && metricKey(m) !== metric) return;
      function point(r, kind) {
        if (!r) return null;
        var y = metric === PARAMS ? (m.params ? m.params / 1e6 : null) : (r.acc ? r.acc.value : null);
        if (y == null) return null;
        var xv = state.view === "llm" ? r.tps : (state.xaxis === "fps" ? r.fps : r.latency);
        return { x: xv, y: y, r: 6, info: { m: m, res: r, kind: kind } };
      }
      var p = point(res, "quant"), f = point(fl, "float");
      if (p) points.push(p);
      if (f) floats.push(f);
      if (p && f) links.push([f, p]);
    });
    // Bubble radius from parameter count (log scale), 5-20 px.
    var all = points.concat(floats);
    var ps = all.map(function (p) { return p.info.m.params; }).filter(Boolean).map(Math.log);
    var lo = Math.min.apply(null, ps), hi = Math.max.apply(null, ps);
    all.forEach(function (p) {
      var v = p.info.m.params;
      p.r = !v ? 6 : hi > lo ? 5 + 15 * (Math.log(v) - lo) / (hi - lo) : 10;
    });
    return { points: points, floats: floats, links: links, table: table };
  }

  function logTick(value) {
    // Label only 1, 2 and 5 times a power of ten; keep the other grid lines unlabelled.
    var mantissa = value / Math.pow(10, Math.floor(Math.log10(value) + 1e-9));
    var shown = [1, 2, 5].some(function (k) { return Math.abs(mantissa - k) < 1e-6; });
    return shown ? fmt(value, value < 1 ? 1 : 0) : "";
  }

  var labelPlugin = {
    id: "modelLabels",
    afterDatasetsDraw: function (chart) {
      if (!state.labels) return;
      var c = chart.ctx, area = chart.chartArea;
      c.save();
      c.font = "11px system-ui, -apple-system, sans-serif";
      c.fillStyle = css("--text");
      c.textBaseline = "middle";
      chart.data.datasets.forEach(function (ds, i) {
        if (!ds.labelled) return;
        chart.getDatasetMeta(i).data.forEach(function (el, j) {
          var raw = ds.data[j];
          if (el.x < area.left || el.x > area.right || el.y < area.top || el.y > area.bottom) return;
          c.fillText(raw.info.m.name, el.x + raw.r + 3, el.y);
        });
      });
      c.restore();
    }
  };

  var chart = new Chart($("chart"), {
    type: "bubble",
    data: { datasets: [] },
    plugins: [labelPlugin],
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      layout: { padding: { right: 90 } },
      scales: {
        x: { type: "logarithmic", title: { display: true }, grid: {}, ticks: { callback: logTick } },
        y: { type: "linear", title: { display: true }, grid: {}, ticks: {}, grace: "5%" }
      },
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false, external: externalTooltip },
        zoom: {
          zoom: { wheel: { enabled: true, modifierKey: "ctrl" }, pinch: { enabled: true }, mode: "xy" },
          pan: { enabled: true, mode: "xy" }
        }
      },
      onClick: function (evt, elements) {
        if (!elements.length) { unpin(); return; }
        var el = elements[0], raw = chart.data.datasets[el.datasetIndex].data[el.index];
        pin(raw.info, el.element);
      },
      onHover: function (evt, elements) {
        evt.native.target.style.cursor = elements.length ? "pointer" : "default";
      }
    }
  });

  function axisTitle() {
    if (state.view === "llm") return "Tokens per second (log)";
    return state.xaxis === "fps" ? "Throughput, FPS (1000 / latency, log)" : "Latency, ms (log)";
  }

  function metricInfo(scope) {
    if (state.metric === PARAMS) return { title: "Parameters (millions, log)", hib: true, desc: "" };
    var m = scope.filter(function (x) { return metricKey(x.m) === state.metric; })[0];
    var a = m ? m.m.accuracy : null;
    var hib = a ? a.higher_is_better !== false : true;
    return {
      title: state.metric + (hib ? "" : " (lower is better, axis inverted)"),
      hib: hib, desc: a ? a.description : ""
    };
  }

  function draw() {
    var keepPin = state.pin;
    unpin();
    var ctx = refreshControls();
    writeHash();
    var built = buildPoints(ctx);
    var mi = metricInfo(ctx.scope);
    var grid = css("--grid"), muted = css("--muted");

    var datasets = built.links.map(function (pair) {
      return {
        type: "line", data: pair.map(function (p) { return { x: p.x, y: p.y }; }), borderColor: muted,
        borderWidth: 1, borderDash: [4, 3], pointRadius: 0, pointHitRadius: 0, order: 3
      };
    });
    if (built.floats.length) {
      datasets.push({
        data: built.floats, labelled: false, order: 2, borderWidth: 2,
        backgroundColor: css("--card"),
        borderColor: function (c) { return c.raw ? cuColor(c.raw.info.res.cu) : muted; }
      });
    }
    datasets.push({
      data: built.points, labelled: true, order: 1, borderWidth: 1,
      backgroundColor: function (c) { return c.raw ? cuColor(c.raw.info.res.cu) + "b3" : muted; },
      borderColor: function (c) { return c.raw ? cuColor(c.raw.info.res.cu) : muted; }
    });
    chart.data.datasets = datasets;

    var o = chart.options.scales;
    o.x.title.text = axisTitle();
    o.y.title.text = mi.title;
    o.y.type = state.metric === PARAMS ? "logarithmic" : "linear";
    o.y.ticks.callback = state.metric === PARAMS ? logTick : undefined;
    o.y.reverse = !mi.hib;
    [o.x, o.y].forEach(function (s) {
      s.grid.color = grid; s.ticks.color = muted; s.title.color = muted;
    });
    chart.resetZoom && chart.resetZoom("none");
    chart.update("none");

    var dev = devices[ctx.di];
    var view = state.view === "llm" ? "LLMs" : state.task;
    $("chart-title").textContent = view + " on " + state.device;
    $("chart-subtitle").textContent = (dev.marketing_name || dev.chipset || "") + (dev.os ? " · " + dev.os : "") +
      " · " + built.points.length + " models plotted";
    var empty = !built.points.length;
    $("empty").hidden = !empty;
    $("empty").textContent = empty ? "No model in this selection has both a result on this device/runtime and a value for the chosen metric. Try another metric, runtime or precision." : "";

    var legend = ["NPU", "GPU", "CPU"].map(function (u) {
      return '<span><i style="background:' + cuColor(u) + '"></i>' + u + "</span>";
    });
    if (built.floats.length) legend.push('<span><i class="hollow"></i>float</span>');
    $("legend").innerHTML = legend.join("");

    var notes = [];
    if (dev.borrowed_from) {
      notes.push("* " + esc(state.device) + " has no measurements of its own: AI Hub reports the numbers of " +
        esc(dev.borrowed_from.device || dev.borrowed_from.chipset) + ".");
    }
    if (mi.desc) notes.push(esc(mi.desc));
    var accDevs = uniq(built.points.map(function (p) { return p.info.res.acc && p.info.res.acc.device; }).filter(Boolean));
    if (accDevs.length) notes.push("Accuracy measured on " + accDevs.map(esc).join(", ") + ".");
    $("notes-inline").innerHTML = notes.join(" ");

    drawTable(built.table, ctx);
    if (keepPin) pinModel(keepPin);
  }

  /* ---------- popup ---------- */

  function popupHtml(info, pinnedView) {
    var m = info.m, r = info.res, a = m.accuracy, rows = [];
    function row(k, v) { rows.push("<tr><td>" + esc(k) + "</td><td>" + v + "</td></tr>"); }
    row("Task", esc(m.use_case));
    row("Precision", esc(r.precision) + (info.kind === "float" ? " (float point)" : ""));
    row("Runtime", esc(runtimeLabel(r.runtime)));
    if (state.view === "llm") {
      row("Tokens/s", fmt(r.tps));
      row("Prefill", fmt(r.prefill, 0) + " tokens/s");
      row("Time to first token", fmt(r.ttft[0], 0) + " – " + fmt(r.ttft[1], 0) + " ms");
      row("Context length", esc(r.context));
    } else {
      row("Latency", fmt(r.latency, 3) + " ms" + (r.components.length > 1 ? ' <span class="flag">(sum of ' + r.components.length + " components)</span>" : ""));
      row("Throughput", fmt(r.fps, 1) + " FPS");
      row("Peak memory", fmt(r.mem[0], 0) + " – " + fmt(r.mem[1], 0) + " MB");
      var L = r.layers;
      row("Compute", esc(r.cu) + " · layers NPU " + L.npu + (L.gpu ? " / GPU " + L.gpu : "") + (L.cpu ? " / CPU " + L.cpu : "") + " of " + L.total);
    }
    if (a) {
      var acc = r.acc;
      row(a.metric, acc ? fmt(acc.value) + " " + esc(a.unit) + (acc.exact ? "" : ' <span class="flag">(' + esc(acc.runtime) + ")</span>") : "—");
      if (a.torch != null) row("Float (PyTorch)", fmt(a.torch) + " " + esc(a.unit));
      if (acc && acc.device) row("Accuracy device", esc(acc.device));
      if (a.dataset) row("Dataset", esc(a.dataset) + (a.num_samples ? " (" + a.num_samples + " samples)" : ""));
    }
    row("Parameters", fmtParams(m.params));
    if (m.resolution) row("Input", esc(m.resolution));
    if (state.view !== "llm" && (r.qairt || r.ort)) {
      row("Tools", (r.qairt ? "QAIRT " + esc(String(r.qairt).split(".").slice(0, 2).join(".")) : "") + (r.ort ? " · ORT " + esc(r.ort) : ""));
    }
    if (m.license_type) row("License", m.license ? '<a href="' + esc(m.license) + '" target="_blank" rel="noopener">' + esc(m.license_type) + "</a>" : esc(m.license_type));
    var links = ['<a href="' + esc(m.links.aihub) + '" target="_blank" rel="noopener">AI Hub page</a>',
      '<a href="' + esc(m.links.github) + '" target="_blank" rel="noopener">GitHub</a>'];
    if (m.research_paper) links.push('<a href="' + esc(m.research_paper) + '" target="_blank" rel="noopener">Paper</a>');
    if (m.source_repo) links.push('<a href="' + esc(m.source_repo) + '" target="_blank" rel="noopener">Source</a>');
    return (pinnedView ? '<button type="button" class="close" title="Close">×</button>' : "") +
      "<h3>" + esc(m.name) + '</h3><div class="sub">' + esc(m.headline || "") + "</div>" +
      "<table>" + rows.join("") + "</table>" +
      '<div class="popup-links">' + links.join("") + "</div>";
  }

  function showPopup(info, x, y, isPinned) {
    var tip = $("tooltip"), wrap = $("chart-wrap");
    tip.innerHTML = popupHtml(info, isPinned);
    tip.classList.toggle("pinned", !!isPinned);
    tip.hidden = false;
    var w = tip.offsetWidth, h = tip.offsetHeight, W = wrap.clientWidth, H = wrap.clientHeight;
    var left = x + 16 + w > W ? x - 16 - w : x + 16;
    var top = Math.min(Math.max(4, y - 24), Math.max(4, H - h - 4));
    tip.style.left = Math.max(4, left) + "px";
    tip.style.top = top + "px";
    if (isPinned) tip.querySelector(".close").onclick = unpin;
  }

  function pin(info, element) {
    pinned = info;
    state.pin = info.m.id;
    writeHash();
    showPopup(info, element.x, element.y, true);
  }

  function pinModel(id) {
    // Pin the plotted (quantized / LLM) point of a model, if it is on the chart.
    var i = chart.data.datasets.length - 1;
    var data = chart.data.datasets[i].data;
    for (var j = 0; j < data.length; j++) {
      if (data[j].info.m.id === id) {
        pin(data[j].info, chart.getDatasetMeta(i).data[j]);
        return;
      }
    }
  }

  function unpin() {
    if (pinned || state.pin) {
      pinned = null;
      state.pin = "";
      writeHash();
    }
    $("tooltip").hidden = true;
  }

  function externalTooltip(context) {
    if (pinned) return;
    var t = context.tooltip;
    if (!t || t.opacity === 0 || !t.dataPoints || !t.dataPoints.length || !t.dataPoints[0].raw.info) {
      $("tooltip").hidden = true;
      return;
    }
    showPopup(t.dataPoints[0].raw.info, t.caretX, t.caretY, false);
  }

  /* ---------- table ---------- */

  function columns() {
    var acc = { key: "acc", label: "Accuracy", num: true, get: function (t) { return t.res && t.res.acc ? t.res.acc.value : null; },
      show: function (t) {
        var a = t.m.accuracy;
        return t.res && t.res.acc ? fmt(t.res.acc.value) + " " + esc(a.unit) : "—";
      } };
    var metric = { key: "metric", label: "Metric", get: function (t) { return t.m.accuracy ? t.m.accuracy.metric : ""; } };
    var name = { key: "name", label: "Model", get: function (t) { return t.m.name; },
      show: function (t) { return '<a href="' + esc(t.m.links.aihub) + '" target="_blank" rel="noopener">' + esc(t.m.name) + "</a>"; } };
    var prec = { key: "prec", label: "Precision", get: function (t) { return t.res ? t.res.precision : ""; } };
    var rt = { key: "rt", label: "Runtime", get: function (t) { return t.res ? runtimeLabel(t.res.runtime) : ""; } };
    var params = { key: "params", label: "Params", num: true, get: function (t) { return t.m.params; }, show: function (t) { return fmtParams(t.m.params); } };
    var gh = { key: "gh", label: "", get: function () { return ""; },
      show: function (t) { return '<a href="' + esc(t.m.links.github) + '" target="_blank" rel="noopener">GitHub</a>'; } };
    if (state.view === "llm") {
      return [name, prec, rt,
        { key: "ctx", label: "Context", num: true, get: function (t) { return t.res ? t.res.context : null; }, show: function (t) { return t.res ? esc(t.res.context) : "—"; } },
        { key: "tps", label: "Tokens/s", num: true, get: function (t) { return t.res ? t.res.tps : null; }, show: function (t) { return fmt(t.res && t.res.tps); } },
        { key: "pre", label: "Prefill tok/s", num: true, get: function (t) { return t.res ? t.res.prefill : null; }, show: function (t) { return fmt(t.res && t.res.prefill, 0); } },
        { key: "ttft", label: "TTFT min (ms)", num: true, get: function (t) { return t.res ? t.res.ttft[0] : null; }, show: function (t) { return fmt(t.res && t.res.ttft[0], 0); } },
        metric, acc, params, gh];
    }
    return [name, prec, rt,
      { key: "lat", label: "Latency (ms)", num: true, get: function (t) { return t.res ? t.res.latency : null; }, show: function (t) { return fmt(t.res && t.res.latency, 3); } },
      { key: "fps", label: "FPS", num: true, get: function (t) { return t.res ? t.res.fps : null; }, show: function (t) { return fmt(t.res && t.res.fps, 1); } },
      metric, acc,
      { key: "torch", label: "Float ref.", num: true, get: function (t) { return t.m.accuracy ? t.m.accuracy.torch : null; },
        show: function (t) { return t.m.accuracy && t.m.accuracy.torch != null ? fmt(t.m.accuracy.torch) : "—"; } },
      params,
      { key: "cu", label: "Compute", get: function (t) { return t.res ? t.res.cu : ""; },
        show: function (t) { return t.res ? '<span class="cu" style="background:' + cuColor(t.res.cu) + '"></span>' + esc(t.res.cu) : "—"; } },
      { key: "npu", label: "NPU layers", num: true, get: function (t) { return t.res && t.res.layers.total ? t.res.layers.npu / t.res.layers.total : null; },
        show: function (t) { return t.res && t.res.layers.total ? fmt(100 * t.res.layers.npu / t.res.layers.total, 0) + "%" : "—"; } },
      { key: "mem", label: "Memory (MB)", num: true, get: function (t) { return t.res ? t.res.mem[1] : null; }, show: function (t) { return t.res ? fmt(t.res.mem[0], 0) + "–" + fmt(t.res.mem[1], 0) : "—"; } },
      gh];
  }

  function drawTable(items, ctx) {
    var cols = columns();
    if (!cols.some(function (c) { return c.key === sort.key; })) sort = { key: state.view === "llm" ? "tps" : "fps", asc: false };
    var col = cols.filter(function (c) { return c.key === sort.key; })[0];
    var list = items.filter(function (t) { return !t.excluded; });
    list.sort(function (a, b) {
      var x = col.get(a), y = col.get(b);
      if (x == null || x === "") return 1;
      if (y == null || y === "") return -1;
      var d = typeof x === "number" ? x - y : String(x).localeCompare(String(y));
      return sort.asc ? d : -d;
    });
    $("table").querySelector("thead").innerHTML = "<tr>" + cols.map(function (c) {
      return '<th data-key="' + c.key + '" class="' + (c.num ? "num " : "") + (c.key === sort.key ? "sorted" + (sort.asc ? " asc" : "") : "") + '">' + esc(c.label) + "</th>";
    }).join("") + "</tr>";
    $("table").querySelector("tbody").innerHTML = list.map(function (t) {
      return '<tr class="' + (t.res ? "" : "dim") + '">' + cols.map(function (c) {
        var v = c.show ? c.show(t) : esc(c.get(t) || "—");
        return '<td class="' + (c.num ? "num" : "") + '">' + v + "</td>";
      }).join("") + "</tr>";
    }).join("");
    $("table-device").textContent = state.device + (state.view === "llm" ? " · LLMs" : " · " + state.task);
    var withRes = list.filter(function (t) { return t.res; }).length;
    $("table-count").textContent = withRes + " of " + list.length + " models have a result for this runtime / precision";
  }

  /* ---------- events ---------- */

  ["world", "device", "task", "metric", "runtime", "precision", "context", "xaxis"].forEach(function (id) {
    $(id).addEventListener("change", function () {
      state[id] = this.value;
      if (id === "task" || id === "world") { state.metric = ""; state.excluded = new Set(); }
      draw();
    });
  });
  document.querySelectorAll(".tab").forEach(function (t) {
    t.addEventListener("click", function () {
      if (state.view === t.dataset.view) return;
      state.view = t.dataset.view;
      state.metric = ""; state.runtime = "auto"; state.excluded = new Set();
      draw();
    });
  });
  $("labels").addEventListener("change", function () { state.labels = this.checked; chart.update("none"); });
  $("model-search").addEventListener("input", function () { state.search = this.value; refreshControls(); });
  $("model-list").addEventListener("change", function (e) {
    var id = e.target.dataset.id;
    if (!id) return;
    if (e.target.checked) state.excluded.delete(id); else state.excluded.add(id);
    draw();
  });
  function setAll(on) {
    var di = deviceIndex[state.device];
    viewModels().forEach(function (x) {
      if (state.view === "models" && x.m.use_case !== state.task) return;
      if (!hasData(x.i, di, state.view)) return;
      if (on) state.excluded.delete(x.m.id); else state.excluded.add(x.m.id);
    });
    draw();
  }
  $("models-all").addEventListener("click", function (e) { e.preventDefault(); setAll(true); });
  $("models-none").addEventListener("click", function (e) { e.preventDefault(); setAll(false); });
  $("reset").addEventListener("click", function () {
    var view = state.view;
    Object.assign(state, DEFAULTS, { view: view, excluded: new Set(), search: "" });
    $("model-search").value = "";
    draw();
  });
  $("zoom-in").addEventListener("click", function () { chart.zoom(1.25); });
  $("zoom-out").addEventListener("click", function () { chart.zoom(0.8); });
  $("zoom-reset").addEventListener("click", function () { chart.resetZoom(); });
  $("table").querySelector("thead").addEventListener("click", function (e) {
    var key = e.target.dataset && e.target.dataset.key;
    if (!key) return;
    sort = { key: key, asc: sort.key === key ? !sort.asc : !e.target.classList.contains("num") };
    draw();
  });
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
  }

  readHash();
  draw();
})();
