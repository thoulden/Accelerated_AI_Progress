/*
 * app.js — UI wiring + Plotly figure rendering for the standalone
 * (non-Streamlit) Accelerated AI Progress simulator.
 *
 * Figures reproduce the original matplotlib charts from single_sim.py /
 * multiple_sims.py one-for-one (same series, colours, reference lines,
 * annotations, log axes and titles).
 */
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };
  var num = function (id) { return parseFloat($(id).value); };
  var checked = function (id) { return $(id).checked; };

  // ---- shared Plotly config / layout helpers ---------------------------
  var PLOT_CONFIG = { responsive: true, displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'] };

  var AXIS = {
    showgrid: true, gridcolor: '#dcdfe4', griddash: 'dash',
    zeroline: false, linecolor: '#c2c7cf', ticks: 'outside', tickcolor: '#c2c7cf',
    titlefont: { size: 13, color: '#333' }, tickfont: { size: 11, color: '#444' }
  };

  function baseLayout(title, xlabel, ylabel, extra) {
    var lay = {
      title: { text: title, font: { size: 16, color: '#1b1f27' }, x: 0.5, xanchor: 'center' },
      margin: { l: 70, r: 24, t: 46, b: 54 },
      paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
      font: { family: '-apple-system, "Segoe UI", Roboto, Arial, sans-serif' },
      xaxis: Object.assign({}, AXIS, { title: { text: xlabel } }),
      yaxis: Object.assign({}, AXIS, { title: { text: ylabel } }),
      legend: { font: { size: 11 }, bgcolor: 'rgba(255,255,255,0.7)',
                bordercolor: '#e5e7eb', borderwidth: 1, x: 0.99, xanchor: 'right',
                y: 0.99, yanchor: 'top' },
      showlegend: true, hovermode: 'closest'
    };
    return Object.assign(lay, extra || {});
  }

  function plotCard(parent, id) {
    var card = document.createElement('div');
    card.className = 'plot-card';
    var div = document.createElement('div');
    div.className = 'plot';
    div.id = id;
    card.appendChild(div);
    parent.appendChild(card);
    return div;
  }

  function noteEl(parent, html) {
    var p = document.createElement('p');
    p.className = 'note';
    p.innerHTML = html;
    parent.appendChild(p);
  }

  function argminAbs(arr, target) {
    var best = 0, bestD = Infinity;
    for (var i = 0; i < arr.length; i++) {
      var d = Math.abs(arr[i] - target);
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  // Python-style round-half-to-even, then fixed 0 decimals + '%'  (mirrors :.0f)
  function pyRound0(x) {
    var f = Math.floor(x);
    var diff = x - f;
    if (diff < 0.5) return f;
    if (diff > 0.5) return f + 1;
    return (f % 2 === 0) ? f : f + 1; // exactly .5 -> nearest even
  }
  function fmtPct(p) { return pyRound0(p * 100) + '%'; }

  // ======================================================================
  // SINGLE SIMULATION
  // ======================================================================

  function readSingleParams() {
    var computeGrowth = checked('s-compute-growth');
    return {
      computeGrowth: computeGrowth,
      f: num('s-f'),
      fMin: num('s-f0'),
      fMax: num('s-fmax'),
      r0: num('s-r'),
      yrLeft: num('s-yr'),
      lambda: num('s-lambda'),
      softwareContribution: num('s-sc'),
      retrainingCost: checked('s-retrain'),
      constantR: checked('s-constant-r')
    };
  }

  function renderSingle() {
    var p = readSingleParams();
    // guard against NaN inputs (mid-typing)
    var required = [p.r0, p.yrLeft, p.lambda, p.softwareContribution];
    required.push(p.computeGrowth ? p.fMin : p.f);
    if (p.computeGrowth) required.push(p.fMax);
    for (var i = 0; i < required.length; i++) {
      if (isNaN(required[i])) return;
    }

    var res = Sim.runSingle(p);
    var host = $('results-single');
    host.innerHTML = '';

    var t = res.times_in_years;
    var yrLeft = p.yrLeft;

    // ---------- Figure 1: AI capabilities over time ----------
    var div1 = plotCard(host, 'fig-capabilities');
    var traces1 = [
      { x: t, y: res.transformed_sizes, mode: 'lines', name: 'AI Capabilities Simulation',
        line: { color: 'blue', width: 2 } },
      { x: t, y: t, mode: 'lines', name: 'Recent pace of progress',
        line: { color: 'black', width: 1.6, dash: 'dot' } },
      { x: [t[0], t[t.length - 1]], y: [yrLeft, yrLeft], mode: 'lines',
        line: { color: 'black', width: 0.5 }, showlegend: false, hoverinfo: 'skip' }
    ];
    var lay1 = baseLayout('AI capabilities over time', 'Time (years)',
      'AI capabilities<br>(years of progress at 2020-4 pace)');
    lay1.annotations = [{
      x: t.length > 2 ? t[2] : t[0], y: yrLeft, text: 'Ceiling', showarrow: false,
      font: { size: 8, color: 'black' }, xanchor: 'left', yanchor: 'bottom'
    }];
    Plotly.newPlot(div1, traces1, lay1, PLOT_CONFIG);
    noteEl(host, '<em>Note:</em> 3 years of progress was roughly the time from GPT-2 to ChatGPT.');

    // ---------- Figure 2: r over time ----------
    var div2 = plotCard(host, 'fig-r');
    var traces2 = [{ x: t, y: res.rs, mode: 'lines', name: 'r(t)', line: { color: 'magenta', width: 2 } }];
    Plotly.newPlot(div2, traces2, baseLayout('r Over Time', 'Time (years)', 'r'), PLOT_CONFIG);

    // ---------- Figure 3: acceleration factor (only with Gradual Boost) ----------
    if (p.computeGrowth) {
      var div3 = plotCard(host, 'fig-f');
      var traces3 = [{ x: t, y: res.f_values, mode: 'lines', name: 'f(t)', line: { color: 'green', width: 2 } }];
      var lay3 = baseLayout('Acceleration Factor Over Time', 'Time (years)', 'f (log scale)');
      lay3.yaxis.type = 'log';
      Plotly.newPlot(div3, traces3, lay3, PLOT_CONFIG);
    }

    // ---------- Figure 4: annualized growth rate ----------
    var div4 = plotCard(host, 'fig-growth');
    var gt = res.growth_times, gr = res.growth_rates;
    var g = 2.77;
    var xr = gt.length ? [gt[0], gt[gt.length - 1]] : [0, 1];
    var traces4 = [
      { x: gt, y: gr, mode: 'lines', name: 'Annualized Growth Rate', line: { color: 'blue', width: 2 } },
      { x: xr, y: [g, g], mode: 'lines', name: 'g = 2.77', line: { color: 'red', width: 1.5, dash: 'dash' } }
    ];
    var mults = [3, 10, 30];
    var mcolors = ['green', 'orange', 'purple'];
    for (var mi = 0; mi < mults.length; mi++) {
      var val = mults[mi] * g;
      traces4.push({ x: xr, y: [val, val], mode: 'lines',
        name: mults[mi] + 'x g = ' + (Math.round(val * 100) / 100),
        line: { color: mcolors[mi], width: 1.4, dash: 'dot' } });
    }
    var lay4 = baseLayout('Annualized Software Growth Rate Over Time', 'Time (years)', 'Annualized Growth Rate');
    lay4.yaxis.type = 'log';
    // Only show y ticks at 1, 10, 100 and no gridlines on this figure.
    lay4.yaxis.tickmode = 'array';
    lay4.yaxis.tickvals = [1, 10, 100];
    lay4.yaxis.ticktext = ['1', '10', '100'];
    lay4.yaxis.showgrid = false;
    lay4.yaxis.minor = { showgrid: false, ticks: '' };
    lay4.xaxis.showgrid = false;
    // Range chosen so the 1, 10 and 100 ticks are all visible.
    var finiteY = gr.filter(function (v) { return isFinite(v) && v > 0; });
    var loData = finiteY.length ? Math.min.apply(null, finiteY) : 1;
    var hiData = finiteY.length ? Math.max.apply(null, finiteY) : 100;
    var loBound = Math.min(loData, 1);
    var hiBound = Math.max(hiData, 100);
    lay4.yaxis.range = [Math.log10(loBound) - 0.08, Math.log10(hiBound) + 0.08];
    Plotly.newPlot(div4, traces4, lay4, PLOT_CONFIG);
    noteEl(host, '<em>Note:</em> an annual growth rate of 2.77 corresponds to doubling every 3 months.');
  }

  // ======================================================================
  // MULTIPLE SIMULATIONS
  // ======================================================================

  function clampField(id, minVal, floorId) {
    var v = num(id);
    if (isNaN(v)) return NaN;
    if (floorId != null) {
      var lo = num(floorId);
      if (!isNaN(lo) && v < lo) { v = lo; $(id).value = v; }
    }
    if (minVal != null && v < minVal) { v = minVal; $(id).value = v; }
    return v;
  }

  function readMultipleBounds() {
    // enforce upper >= lower (mirrors Streamlit min_value = lower)
    var ib_low = clampField('m-ib-low', 0.1, null);
    var ib_high = clampField('m-ib-high', 0.1, 'm-ib-low');
    var r_low = clampField('m-r-low', 0.01, null);
    var r_high = clampField('m-r-high', 0.01, 'm-r-low');
    var ly_low = clampField('m-ly-low', 1, null);
    var ly_high = clampField('m-ly-high', 1, 'm-ly-low');
    var lf_low = clampField('m-lf-low', 0.01, null);
    var lf_high = clampField('m-lf-high', 0.01, 'm-lf-low');
    return { ib_low: ib_low, ib_high: ib_high, r_low: r_low, r_high: r_high,
             ly_low: ly_low, ly_high: ly_high, lf_low: lf_low, lf_high: lf_high };
  }

  var lastMultiForCsv = null;

  function renderMultiple() {
    var bounds = readMultipleBounds();
    var sc = num('m-sc');
    var nSims = Math.round(num('m-nsims'));
    var flags = {
      computeGrowth: checked('m-compute-growth'),
      retrainingCost: checked('m-retrain'),
      constantR: checked('m-constant-r')
    };
    if (isNaN(sc) || isNaN(nSims) || nSims < 1) return;

    var host = $('results-multiple');
    host.innerHTML = '';

    // progress bar
    var pWrap = document.createElement('div');
    pWrap.className = 'progress-wrap';
    var pBar = document.createElement('div');
    pBar.className = 'progress-bar';
    pWrap.appendChild(pBar);
    host.appendChild(pWrap);

    // Run (async so the progress bar can paint) --------------------------
    $('run-multiple').disabled = true;
    setTimeout(function () {
      var out = Sim.runMultiple(nSims, bounds, sc, flags, function (frac) {
        pBar.style.width = (frac * 100).toFixed(1) + '%';
      });
      pWrap.remove();
      $('run-multiple').disabled = false;
      renderMultipleResults(host, out, sc, nSims);
    }, 20);
  }

  function renderMultipleResults(host, out, sc, nSims) {
    var prob = out.probabilities;

    // ---------- Results table ----------
    var h = document.createElement('h3');
    h.className = 'subsection';
    h.textContent = 'Years of AI progress compressed into different time periods';
    host.appendChild(h);

    var table = document.createElement('table');
    table.className = 'result-table';
    table.innerHTML =
      '<thead><tr><th>Years of progress</th><th>Compressed into ≤1 year</th>' +
      '<th>Compressed into ≤4 months</th></tr></thead><tbody>' +
      '<tr><td>≥3 years</td><td>' + fmtPct(prob['12_3']) + '</td><td>' + fmtPct(prob['4_10']) + '</td></tr>' +
      '<tr><td>≥10 years</td><td>' + fmtPct(prob['12_10']) + '</td><td>' + fmtPct(prob['4_30']) + '</td></tr>' +
      '</tbody>';
    host.appendChild(table);

    // ---------- CDF figure ----------
    var cdf12 = Sim.calculateYearsCompressedCdf(out.timesMatrix, sc, 12, 20, 200);
    var cdf4 = Sim.calculateYearsCompressedCdf(out.timesMatrix, sc, 4, 20, 200);

    var div = plotCard(host, 'fig-cdf');
    var traces = [
      { x: cdf12.years, y: cdf12.fractions, mode: 'lines', name: 'Compressed into ≤1 year',
        line: { color: 'blue', width: 2 } },
      { x: cdf4.years, y: cdf4.fractions, mode: 'lines', name: 'Compressed into ≤4 months',
        line: { color: 'purple', width: 2 } },
      // legend proxies for the reference lines (match original de-duped legend)
      { x: [null], y: [null], mode: 'lines', name: '3 years reference',
        line: { color: 'red', width: 1.4, dash: 'dash' } },
      { x: [null], y: [null], mode: 'lines', name: '10 years reference',
        line: { color: 'green', width: 1.4, dash: 'dash' } }
    ];

    var lay = baseLayout('Probability of Compressing ≥X Years into One Year or Four Months',
      'Years Compressed', 'Probability');
    lay.xaxis.range = [0, 20];
    lay.yaxis.range = [0, 1];
    lay.shapes = [
      { type: 'line', x0: 3, x1: 3, y0: 0, y1: 1, line: { color: 'red', width: 1.2, dash: 'dash' }, opacity: 0.5 },
      { type: 'line', x0: 3, x1: 3, y0: 0, y1: 1, line: { color: 'red', width: 1.2, dash: 'dot' }, opacity: 0.5 },
      { type: 'line', x0: 10, x1: 10, y0: 0, y1: 1, line: { color: 'green', width: 1.2, dash: 'dash' }, opacity: 0.5 },
      { type: 'line', x0: 10, x1: 10, y0: 0, y1: 1, line: { color: 'green', width: 1.2, dash: 'dot' }, opacity: 0.5 }
    ];

    // annotations — percentage labels at key points (mirror original)
    var anns = [];
    [3, 10].forEach(function (years) {
      var idx = argminAbs(cdf12.years, years);
      var p12 = cdf12.fractions[idx];
      anns.push({ x: years + 0.5, y: Math.min(1, p12 + 0.05), text: fmtPct(p12),
        showarrow: false, font: { size: 11, color: 'blue' }, xanchor: 'left' });
    });
    var idx43 = argminAbs(cdf4.years, 10 / 3);
    var p43 = cdf4.fractions[idx43];
    anns.push({ x: 3 + 0.5, y: Math.max(0, p43 - 0.1), text: fmtPct(p43),
      showarrow: false, font: { size: 11, color: 'purple' }, xanchor: 'left' });
    var idx410 = argminAbs(cdf4.years, 10);
    var p410 = cdf4.fractions[idx410];
    anns.push({ x: 10 + 0.5, y: Math.max(0, p410 - 0.1), text: fmtPct(p410),
      showarrow: false, font: { size: 11, color: 'purple' }, xanchor: 'left' });
    lay.annotations = anns;

    Plotly.newPlot(div, traces, lay, PLOT_CONFIG);

    // ---------- CSV download / info ----------
    if (nSims <= 2000) {
      var btn = document.createElement('button');
      btn.className = 'download-btn';
      btn.textContent = 'Download Simulation Results (CSV)';
      btn.addEventListener('click', function () { downloadCsv(out); });
      host.appendChild(btn);
    } else {
      var info = document.createElement('div');
      info.className = 'infobox neutral';
      info.textContent = 'CSV download of simulation results unavailable above 2000 simulations.';
      host.appendChild(info);
    }
  }

  function downloadCsv(out) {
    var cols = ['Simulation', 'r_initial', 'initial_factor_increase_time', 'limit_years',
      'f_0', 'f_max', 'lambda_factor', 'software_contribution', 'Time (Months)', 'Size'];
    var lines = [cols.join(',')];
    for (var i = 0; i < out.timesMatrix.length; i++) {
      var times = out.timesMatrix[i];
      var sizes = out.sizesMatrix[i];
      var pr = out.paramsList[i];
      for (var j = 0; j < times.length; j++) {
        lines.push([
          i + 1, pr.r_initial, pr.initial_factor_increase_time, pr.limit_years,
          pr.f_0, pr.f_max, pr.lambda_factor, pr.software_contribution_param,
          times[j], sizes[j]
        ].join(','));
      }
    }
    var blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'simulation_results.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function showMultiplePlaceholder() {
    var host = $('results-multiple');
    host.innerHTML = '<div class="placeholder">Press \'Run Simulation\' to view results.</div>';
  }

  // ======================================================================
  // SPEED-UP CALCULATOR  (no-SIE regime)
  //   Before automation, labour is exogenous at g_L^before, giving
  //       g_before = r(α·g_L^before + (1-α)·g_C).
  //   After automation, quality-adjusted labour is L_after ∝ C^(1+γ) S, so
  //   g_L = (1+γ) g_C + g_S; with law of motion g_S = L^α C_E^(1-α) S^(-1/r)
  //   the BGP gives g_after = r(1 + α·γ) g_C / (1 - r·α).
  //   Speed-up = g_after / g_before  (→ (1+α·γ)/(1-r·α) when g_L^before = g_C).
  // ======================================================================

  function readSpeedupParams() {
    return {
      alpha: num('sp-alpha'),
      gamma: num('sp-gamma'),
      r: num('sp-r'),
      gLbefore: num('sp-glbefore'),
      gC: num('sp-gc')
    };
  }

  function fmtRatePct(x) { return Math.round(x * 100) + '%/yr'; }

  function renderSpeedup() {
    var p = readSpeedupParams();
    if (isNaN(p.alpha) || isNaN(p.gamma) || isNaN(p.r) || isNaN(p.gLbefore) || isNaN(p.gC)) return;
    var ra = p.r * p.alpha;
    var gBefore = p.r * (p.alpha * p.gLbefore + (1 - p.alpha) * p.gC);   // exogenous pre-automation baseline
    var multEl = $('sp-multiplier');
    var subEl = $('sp-subnote');

    $('sp-gbefore').textContent = fmtRatePct(gBefore);

    if (ra >= 1) {
      multEl.textContent = '∞';
      multEl.classList.add('explosion');
      subEl.textContent = 'r·α = ' + ra.toFixed(2) + ' ≥ 1  →  software-only intelligence explosion';
      $('sp-gafter').textContent = '∞ (explosive)';
      $('sp-mult-inline').textContent = '∞';
    } else {
      multEl.classList.remove('explosion');
      var gAfter = p.r * (1 + p.alpha * p.gamma) * p.gC / (1 - ra);
      var M = (gBefore > 0) ? gAfter / gBefore : Infinity;
      var Mtxt = isFinite(M) ? M.toFixed(1) : '∞';
      multEl.textContent = Mtxt + '×';
      subEl.innerHTML = '= g<sub>S</sub><sup>after</sup> / g<sub>S</sub><sup>before</sup> &nbsp;(r·α = ' + ra.toFixed(2) + ')';
      $('sp-gafter').textContent = fmtRatePct(gAfter);
      $('sp-mult-inline').textContent = (isFinite(M) ? M.toFixed(2) : '∞') + '×';
    }
  }

  // ======================================================================
  // TAB SWITCHING + WIRING
  // ======================================================================

  var MODES = ['single', 'multiple', 'speedup'];

  function setMode(mode) {
    MODES.forEach(function (mo) {
      var on = mo === mode;
      $('tab-' + mo).classList.toggle('active', on);
      $('tab-' + mo).setAttribute('aria-selected', on ? 'true' : 'false');
      $('controls-' + mo).classList.toggle('hidden', !on);
      $('results-' + mo).classList.toggle('hidden', !on);
    });
    // The shared "Model Parameters and Estimates" block (and everything below
    // it) describes the simulation model; hide it on the Speed-Up Calculator tab.
    $('shared-model-info').classList.toggle('hidden', mode === 'speedup');
    // Plotly needs a resize nudge when a hidden container becomes visible
    window.dispatchEvent(new Event('resize'));
  }

  function toggleGradualBoost() {
    var on = checked('s-compute-growth');
    $('s-f-single-wrap').classList.toggle('hidden', on);
    $('s-f0-wrap').classList.toggle('hidden', !on);
    $('s-fmax-wrap').classList.toggle('hidden', !on);
  }

  // Convert native `title` tooltips into immediate custom hover tooltips.
  function initTooltips() {
    var infos = document.querySelectorAll('.info[title]');
    for (var i = 0; i < infos.length; i++) {
      var el = infos[i];
      var tip = el.getAttribute('title');
      el.removeAttribute('title');           // suppress the slow native tooltip
      if (tip && tip !== '...') {
        el.setAttribute('data-tip', tip);
      } else {
        // no real help text: drop the marker so it isn't a dead "?"
        el.classList.add('no-tip');
      }
    }
  }

  function init() {
    initTooltips();

    // Tabs
    $('tab-single').addEventListener('click', function () { setMode('single'); });
    $('tab-multiple').addEventListener('click', function () { setMode('multiple'); });
    $('tab-speedup').addEventListener('click', function () { setMode('speedup'); });

    // Single-sim: live recompute on any control change + explicit button
    var singleInputs = ['s-f', 's-f0', 's-fmax', 's-r', 's-yr', 's-lambda', 's-sc'];
    singleInputs.forEach(function (id) {
      $(id).addEventListener('input', renderSingle);
    });
    ['s-retrain', 's-constant-r'].forEach(function (id) {
      $(id).addEventListener('change', renderSingle);
    });
    $('s-compute-growth').addEventListener('change', function () {
      toggleGradualBoost();
      renderSingle();
    });
    $('run-single').addEventListener('click', renderSingle);

    // Multiple-sim: only on button press
    $('run-multiple').addEventListener('click', renderMultiple);

    // Speed-up calculator: live recompute on any control change
    ['sp-alpha', 'sp-gamma', 'sp-r', 'sp-glbefore', 'sp-gc'].forEach(function (id) {
      $(id).addEventListener('input', renderSpeedup);
    });

    // Initial state
    toggleGradualBoost();
    renderSingle();          // single sim auto-runs on load (matches original)
    showMultiplePlaceholder();
    renderSpeedup();         // speed-up calculator computes on load
    setMode('single');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
