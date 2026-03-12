/**
 * BirdNET Geomodel – Interactive Web Demo
 *
 * Runs the ONNX FP16 model entirely client-side via ONNX Runtime Web.
 * Three modes:
 *   1. Range Map    – species probability heatmap on a Leaflet map
 *   2. Richness Map – predicted species count per cell
 *   3. Species List – click a location to see predicted species
 *
 * The model input is (batch, 3) = [lat, lon, week] and output is
 * (batch, n_species) sigmoid probabilities.
 */

(function () {
  "use strict";

  // ---- Configuration -------------------------------------------------------
  var MODEL_URL = "geomodel_fp16.onnx";
  var LABELS_URL = "labels.txt";

  // Grid resolution per zoom level (degrees per cell)
  var ZOOM_STEP = { 2: 3, 3: 2, 4: 1 };

  // Perceptual scaling: gamma < 1 stretches low values for visibility
  var DISPLAY_GAMMA = 0.5;

  // Preselected species (species code for quick access)
  var FEATURED_SPECIES = [
    { key: "9515886", sci: "Hirundo rustica", common: "Barn Swallow" },
    { key: "5229230", sci: "Sterna paradisaea", common: "Arctic Tern" },
    { key: "5231918", sci: "Cuculus canorus", common: "Common Cuckoo" },
    { key: "2493052", sci: "Phylloscopus trochilus", common: "Willow Warbler" },
    { key: "9705453", sci: "Parus major", common: "Great Tit" },
    { key: "2490384", sci: "Cardinalis cardinalis", common: "Northern Cardinal" },
    { key: "9510564", sci: "Turdus migratorius", common: "American Robin" },
    { key: "5232445", sci: "Branta sandvicensis", common: "Hawaiian Goose (N\u0113n\u0113)" },
    { key: "2479593", sci: "Nestor notabilis", common: "Kea" },
    { key: "2480569", sci: "Buteo galapagoensis", common: "Gal\u00e1pagos Hawk" },
    { key: "2495144", sci: "Apteryx mantelli", common: "North Island Brown Kiwi" },
    { key: "5219243", sci: "Vulpes vulpes", common: "Red Fox" },
    { key: "2436940", sci: "Oryctolagus cuniculus", common: "European Rabbit" },
    { key: "5219681", sci: "Sciurus carolinensis", common: "Eastern Gray Squirrel" },
    { key: "5218786", sci: "Procyon lotor", common: "Common Raccoon" },
  ];

  // ---- Colormap ------------------------------------------------------------
  var COLORMAP = (function () {
    var stops = [
      [0.0,   0,   0,   4],
      [0.14, 31,  12,  72],
      [0.28, 85,  15, 109],
      [0.42, 136,  8,  79],
      [0.56, 186, 54,  36],
      [0.70, 227, 105,  5],
      [0.84, 249, 174,  10],
      [1.0,  252, 255, 164],
    ];
    var ramp = new Array(256);
    for (var i = 0; i < 256; i++) {
      var t = i / 255;
      var lo = stops[0], hi = stops[stops.length - 1];
      for (var s = 0; s < stops.length - 1; s++) {
        if (t >= stops[s][0] && t <= stops[s + 1][0]) {
          lo = stops[s]; hi = stops[s + 1]; break;
        }
      }
      var f = (t - lo[0]) / (hi[0] - lo[0] || 1);
      ramp[i] = [
        Math.round(lo[1] + f * (hi[1] - lo[1])),
        Math.round(lo[2] + f * (hi[2] - lo[2])),
        Math.round(lo[3] + f * (hi[3] - lo[3])),
      ];
    }
    return ramp;
  })();

  function colormapLookup(p) {
    var idx = Math.max(0, Math.min(255, Math.round(p * 255)));
    return COLORMAP[idx] || [0, 0, 0];
  }

  // ---- Utilities -----------------------------------------------------------
  function perceptualNorm(raw, maxVal) {
    var out = new Float32Array(raw.length);
    if (maxVal > 0) {
      for (var i = 0; i < raw.length; i++)
        out[i] = Math.pow(raw[i] / maxVal, DISPLAY_GAMMA);
    }
    return out;
  }

  function wrapLon(v) { return ((((v + 180) % 360) + 360) % 360) - 180; }

  function setStatus(msg) {
    var el = document.getElementById("demo-status");
    if (el) el.textContent = msg;
  }

  function weekText(w) {
    var months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    var period = ["early", "mid", "late", "late"];
    var mi = Math.floor((w - 1) / 4);
    var pi = (w - 1) % 4;
    return "Week " + w + " (" + period[pi] + " " + (months[mi] || "Dec") + ")";
  }

  // ---- State ---------------------------------------------------------------
  var worker = null;
  var inferenceId = 0;
  var pendingInferences = new Map();
  var labels = [];
  var labelsByKey = {};
  var map = null;
  var overlayCanvas = null;
  var cachedRender = null;
  var renderCache = new Map();
  var RENDER_CACHE_MAX = 50;
  var marker = null;
  var currentMode = "range";
  var rendering = false;
  var renderGeneration = 0;
  var moveEndTimer = null;

  // Capture script location at parse time (before DOMContentLoaded fires)
  var SCRIPT_BASE = (document.currentScript && document.currentScript.src)
    ? document.currentScript.src : window.location.href;

  // ---- Bootstrap -----------------------------------------------------------
  document.addEventListener("DOMContentLoaded", init);

  async function init() {
    var root = document.getElementById("demo-root");
    if (!root) return;

    root.innerHTML =
      '<div id="demo-loading"><div class="spinner"></div>Loading model &amp; labels\u2026</div>' +
      '<div id="demo-app" style="display:none">' +
        '<div id="demo-controls">' +
          '<div class="ctrl-group">' +
            '<label for="mode-select">Mode</label>' +
            '<select id="mode-select">' +
              '<option value="range">Species Range</option>' +
              '<option value="richness">Species Richness</option>' +
              '<option value="list">Species List (click map)</option>' +
            '</select>' +
          '</div>' +
          '<div class="ctrl-group" id="species-search-wrap">' +
            '<label for="species-search">Species</label>' +
            '<input id="species-search" type="text" autocomplete="off" placeholder="Search species\u2026" />' +
            '<div id="species-results"></div>' +
          '</div>' +
          '<div class="ctrl-group" id="week-select-wrap">' +
            '<label for="week-select">Week</label>' +
            '<select id="week-select">' +
              '<option value="1">Week 1 (early Jan)</option>' +
              '<option value="26">Week 26 (late Jun)</option>' +
            '</select>' +
          '</div>' +
          '<div class="ctrl-group" id="threshold-wrap" style="display:none">' +
            '<label for="threshold-select">Min probability</label>' +
            '<select id="threshold-select">' +
              '<option value="1">1%</option>' +
              '<option value="5" selected>5%</option>' +
              '<option value="10">10%</option>' +
              '<option value="25">25%</option>' +
              '<option value="50">50%</option>' +
            '</select>' +
          '</div>' +
        '</div>' +
        '<div id="demo-status">&nbsp;</div>' +
        '<div id="demo-map-wrap">' +
          '<div id="demo-map"></div>' +
          '<div id="demo-computing" style="display:none">' +
            '<div class="spinner"></div>' +
            '<div id="computing-text">Computing\u2026</div>' +
            '<div id="computing-progress-wrap"><div id="computing-progress-bar"></div></div>' +
          '</div>' +
          '<div id="demo-legend"></div>' +
        '</div>' +
        '<div id="species-panel">' +
          '<h3 id="sp-title">Species at location</h3>' +
          '<div class="sp-coords" id="sp-coords"></div>' +
          '<table id="species-list-table">' +
            '<thead><tr><th>#</th><th>Species</th><th>Scientific name</th><th>Probability</th><th></th></tr></thead>' +
            '<tbody id="sp-tbody"></tbody>' +
          '</table>' +
        '</div>' +
      '</div>';

    try {
      await Promise.all([initWorker(), loadLabels()]);
      document.getElementById("demo-loading").style.display = "none";
      document.getElementById("demo-app").style.display = "block";
      initMap();
      bindControls();
      setStatus("Select a species to view its predicted range map.");
    } catch (e) {
      document.getElementById("demo-loading").innerHTML =
        '<span style="color:red">Failed to load: ' + e.message + '</span>';
      console.error(e);
    }
  }

  // ---- Model & labels ------------------------------------------------------
  async function initWorker() {
    setStatus("Loading ONNX model\u2026");
    worker = new Worker(new URL("inference-worker.js", SCRIPT_BASE).href);
    worker.onerror = function (err) { console.error("Worker error:", err); };

    await new Promise(function (resolve, reject) {
      worker.onmessage = function (e) {
        if (e.data.type === "init") {
          if (e.data.ok) resolve();
          else reject(new Error(e.data.error || "Worker init failed"));
        }
      };
      worker.postMessage({ type: "init", modelUrl: new URL(MODEL_URL, SCRIPT_BASE).href });
    });

    worker.onmessage = function (e) {
      var msg = e.data;
      if (msg.type !== "infer") return;
      var p = pendingInferences.get(msg.id);
      if (!p) return;
      pendingInferences.delete(msg.id);
      if (msg.error) p.reject(new Error(msg.error));
      else p.resolve(new Float32Array(msg.data));
    };
  }

  async function loadLabels() {
    var resp = await fetch(new URL(LABELS_URL, SCRIPT_BASE).href);
    var text = await resp.text();
    labels = text.trim().split("\n").map(function (line, i) {
      var parts = line.split("\t");
      return { key: parts[0], sci: parts[1] || "", common: parts[2] || parts[1] || "", index: i };
    });
    labelsByKey = {};
    labels.forEach(function (l) { labelsByKey[l.key] = l; });
  }

  // ---- Map setup -----------------------------------------------------------
  function initMap() {
    map = L.map("demo-map", { center: [30, 0], zoom: 2, minZoom: 2, maxZoom: 4 });

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
      maxZoom: 4, subdomains: "abcd",
    }).addTo(map);

    map.on("click", onMapClick);
    map.on("moveend", function () {
      if (currentMode !== "range" && currentMode !== "richness") return;
      paintOverlay();
      clearTimeout(moveEndTimer);
      moveEndTimer = setTimeout(triggerRender, 300);
    });
  }

  function triggerRender() {
    if (currentMode === "richness") renderRichness();
    else if (currentMode === "range") renderRangeMap();
  }

  // ---- Controls ------------------------------------------------------------
  function bindControls() {
    var modeEl = document.getElementById("mode-select");
    modeEl.addEventListener("change", function () {
      currentMode = modeEl.value;
      document.getElementById("species-search-wrap").style.display = currentMode === "range" ? "" : "none";
      document.getElementById("threshold-wrap").style.display = currentMode === "list" ? "" : "none";
      document.getElementById("species-panel").style.display = "none";
      if (cachedRender) clearOverlay();
      if (marker) { map.removeLayer(marker); marker = null; }
      updateLegend();
      if (currentMode === "range" || currentMode === "richness") triggerRender();
    });

    document.getElementById("week-select").addEventListener("change", function () {
      if (currentMode === "range" || currentMode === "richness") showCachedWeek();
      else if (marker) { var ll = marker.getLatLng(); renderSpeciesList(ll.lat, ll.lng); }
      updateLegend();
    });

    document.getElementById("threshold-select").addEventListener("change", function () {
      if (currentMode === "list" && marker) { var ll = marker.getLatLng(); renderSpeciesList(ll.lat, ll.lng); }
    });

    var searchEl = document.getElementById("species-search");
    var resultsEl = document.getElementById("species-results");
    var selIdx = -1;

    searchEl.addEventListener("focus", function () { showSearch(searchEl, resultsEl); });
    searchEl.addEventListener("input", function () { selIdx = -1; showSearch(searchEl, resultsEl); });
    searchEl.addEventListener("keydown", function (e) {
      var items = resultsEl.querySelectorAll(".sr-item");
      if (e.key === "ArrowDown") { e.preventDefault(); selIdx = Math.min(selIdx + 1, items.length - 1); highlightItem(items, selIdx); }
      else if (e.key === "ArrowUp") { e.preventDefault(); selIdx = Math.max(selIdx - 1, 0); highlightItem(items, selIdx); }
      else if (e.key === "Enter" && selIdx >= 0 && items[selIdx]) { e.preventDefault(); items[selIdx].click(); }
      else if (e.key === "Escape") { resultsEl.style.display = "none"; }
    });
    document.addEventListener("click", function (e) {
      if (!resultsEl.contains(e.target) && e.target !== searchEl) resultsEl.style.display = "none";
    });
  }

  function showSearch(inputEl, resultsEl) {
    var q = inputEl.value.trim().toLowerCase();
    var matches;
    if (q.length === 0) {
      matches = FEATURED_SPECIES.map(function (f) { return labelsByKey[f.key]; }).filter(Boolean);
    } else {
      matches = labels.filter(function (l) {
        return l.common.toLowerCase().includes(q) || l.sci.toLowerCase().includes(q) || l.key.includes(q);
      }).slice(0, 30);
    }
    resultsEl.innerHTML = matches.map(function (l) {
      return '<div class="sr-item" data-key="' + l.key + '">' + l.common + ' <span class="sr-sci">' + l.sci + '</span></div>';
    }).join("");
    resultsEl.style.display = matches.length ? "block" : "none";
    resultsEl.querySelectorAll(".sr-item").forEach(function (el) {
      el.addEventListener("click", function () {
        selectSpecies(el.dataset.key);
        inputEl.value = "";
        resultsEl.style.display = "none";
      });
    });
  }

  function highlightItem(items, idx) {
    items.forEach(function (el, i) { el.classList.toggle("active", i === idx); });
    if (items[idx]) items[idx].scrollIntoView({ block: "nearest" });
  }

  function selectSpecies(key) {
    var lbl = labelsByKey[key];
    if (!lbl) return;
    var el = document.getElementById("species-search");
    el.placeholder = lbl.common + " (" + lbl.sci + ")";
    el.dataset.selectedKey = key;
    if (currentMode === "range") renderRangeMap();
  }

  // ---- Inference -----------------------------------------------------------
  async function runInference(flatInputs, batchSize) {
    var id = ++inferenceId;
    return new Promise(function (resolve, reject) {
      pendingInferences.set(id, { resolve: resolve, reject: reject });
      var buf = new Float32Array(flatInputs).buffer;
      worker.postMessage({ type: "infer", id: id, flatInputs: buf, batchSize: batchSize }, [buf]);
    });
  }

  // ---- Overlay -------------------------------------------------------------
  function ensureOverlayCanvas() {
    if (overlayCanvas) return;
    overlayCanvas = document.createElement("canvas");
    overlayCanvas.className = "heatmap-overlay";
    overlayCanvas.style.position = "absolute";
    overlayCanvas.style.pointerEvents = "none";
    map.getPane("overlayPane").appendChild(overlayCanvas);
  }

  function clearOverlay() {
    cachedRender = null;
    if (overlayCanvas) { overlayCanvas.width = 0; overlayCanvas.height = 0; }
  }

  function paintOverlay() {
    if (!cachedRender || !map) return;
    ensureOverlayCanvas();

    var g = cachedRender.grid, probs = cachedRender.probs;
    var size = map.getSize();
    overlayCanvas.width = size.x;
    overlayCanvas.height = size.y;
    L.DomUtil.setPosition(overlayCanvas, map.containerPointToLayerPoint([0, 0]));

    var ctx = overlayCanvas.getContext("2d");
    var pi = 0;
    for (var iLat = 0; iLat < g.nLat; iLat++) {
      var latN = g.north - iLat * g.step, latS = latN - g.step;
      for (var iLon = 0; iLon < g.nLon; iLon++) {
        var lonW = g.west + iLon * g.step, lonE = lonW + g.step;
        var p = probs[pi++];
        if (!(p >= 0.01)) continue;
        var nw = map.latLngToContainerPoint([latN, lonW]);
        var se = map.latLngToContainerPoint([latS, lonE]);
        var x = Math.floor(nw.x), y = Math.floor(nw.y);
        var w = Math.ceil(se.x) - x, h = Math.ceil(se.y) - y;
        if (x + w < 0 || y + h < 0 || x > size.x || y > size.y) continue;
        var c = colormapLookup(p);
        var alpha = Math.min(1, 0.25 + p * 0.75);
        ctx.fillStyle = "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + alpha.toFixed(3) + ")";
        ctx.fillRect(x, y, w, h);
      }
    }
  }

  // ---- Viewport grid -------------------------------------------------------
  function viewportGrid() {
    var b = map.getBounds();
    var south = Math.max(b.getSouth(), -90), north = Math.min(b.getNorth(), 90);
    var west = b.getWest(), east = b.getEast();
    if (east - west >= 360) { west = -180; east = 180; }
    else { west = wrapLon(west); east = wrapLon(east); if (east <= west) east += 360; }
    if (north - south < 0.1) north = south + 0.1;
    if (east - west < 0.1) east = west + 0.1;
    var step = ZOOM_STEP[map.getZoom()] || 3;
    south = Math.max(Math.floor(south / step) * step, -90);
    north = Math.min(Math.ceil(north / step) * step, 90);
    west = Math.floor(west / step) * step;
    east = Math.ceil(east / step) * step;
    return { south: south, north: north, west: west, east: east, step: step,
             nLat: Math.round((north - south) / step), nLon: Math.round((east - west) / step) };
  }

  // ---- Cell cache ----------------------------------------------------------
  function cacheKey(speciesKey, week) { return speciesKey + ":" + week; }
  function cellId(lat, lon) { return Math.round(lat * 100) + "," + Math.round(lon * 100); }

  function getCellMap(key, step) {
    var entry = renderCache.get(key);
    if (!entry || entry.step !== step) {
      entry = { step: step, cells: new Map() };
      renderCache.set(key, entry);
      if (renderCache.size > RENDER_CACHE_MAX) renderCache.delete(renderCache.keys().next().value);
    }
    return entry.cells;
  }

  function viewportMissing(cellMap, grid) {
    var pts = [];
    for (var iLat = 0; iLat < grid.nLat; iLat++) {
      var lat = grid.north - (iLat + 0.5) * grid.step;
      for (var iLon = 0; iLon < grid.nLon; iLon++) {
        var lon = wrapLon(grid.west + (iLon + 0.5) * grid.step);
        if (!cellMap.has(cellId(lat, lon))) pts.push({ lat: lat, lon: lon });
      }
    }
    return pts;
  }

  function buildViewportArray(cellMap, grid) {
    var arr = new Float32Array(grid.nLat * grid.nLon);
    var i = 0;
    for (var iLat = 0; iLat < grid.nLat; iLat++) {
      var lat = grid.north - (iLat + 0.5) * grid.step;
      for (var iLon = 0; iLon < grid.nLon; iLon++) {
        var lon = wrapLon(grid.west + (iLon + 0.5) * grid.step);
        arr[i++] = cellMap.get(cellId(lat, lon)) || 0;
      }
    }
    return arr;
  }

  function normaliseProbs(raw) {
    var maxProb = 0;
    for (var i = 0; i < raw.length; i++) if (raw[i] > maxProb) maxProb = raw[i];
    return { probs: perceptualNorm(raw, maxProb), maxProb: maxProb };
  }

  function allWeeks() {
    return Array.from(document.getElementById("week-select").options, function (o) { return +o.value; });
  }

  // ---- Range map -----------------------------------------------------------
  async function renderRangeMap() {
    var key = document.getElementById("species-search").dataset.selectedKey;
    if (!key || !labelsByKey[key]) return;
    if (rendering) { renderGeneration++; return; }
    var gen = ++renderGeneration;
    var lbl = labelsByKey[key], speciesIdx = lbl.index;
    var selectedWeek = +document.getElementById("week-select").value;
    var weeks = allWeeks(), nSpecies = labels.length, CHUNK = 4096;
    var g = viewportGrid(), totalPoints = g.nLat * g.nLon;

    // Find weeks with missing cells
    var weekMissing = [];
    weeks.forEach(function (w) {
      var cm = getCellMap(cacheKey(key, w), g.step);
      var miss = viewportMissing(cm, g);
      if (miss.length > 0) weekMissing.push({ week: w, missing: miss, cellMap: cm });
    });

    // Fast path: all cached
    if (weekMissing.length === 0) {
      var nr = normaliseProbs(buildViewportArray(getCellMap(cacheKey(key, selectedWeek), g.step), g));
      cachedRender = { grid: g, probs: nr.probs, maxProb: nr.maxProb };
      paintOverlay();
      setStatus(lbl.common + " \u2013 " + weekText(selectedWeek) + " \u00b7 " + totalPoints.toLocaleString() + " cells (" + g.step + "\u00b0) [cached]");
      updateLegend();
      return;
    }

    rendering = true;
    showComputingOverlay(true, lbl.common);
    try {
      for (var wi = 0; wi < weekMissing.length; wi++) {
        var wm = weekMissing[wi];
        setStatus("Computing " + lbl.common + " \u2013 " + weekText(wm.week) + " \u00b7 " + wm.missing.length + " new cells [" + (wi + 1) + "/" + weekMissing.length + "]\u2026");
        var inputs = new Float32Array(wm.missing.length * 3);
        for (var i = 0; i < wm.missing.length; i++) {
          inputs[i * 3] = wm.missing[i].lat;
          inputs[i * 3 + 1] = wm.missing[i].lon;
          inputs[i * 3 + 2] = wm.week;
        }
        var rawProbs = new Float32Array(wm.missing.length);
        for (var start = 0; start < wm.missing.length; start += CHUNK) {
          if (gen !== renderGeneration) return;
          var end = Math.min(start + CHUNK, wm.missing.length);
          var out = await runInference(inputs.subarray(start * 3, end * 3), end - start);
          for (var j = 0; j < end - start; j++) rawProbs[start + j] = out[j * nSpecies + speciesIdx];
        }
        if (gen !== renderGeneration) return;
        for (var k = 0; k < wm.missing.length; k++) wm.cellMap.set(cellId(wm.missing[k].lat, wm.missing[k].lon), rawProbs[k]);
        if (wm.week === selectedWeek) {
          var nr2 = normaliseProbs(buildViewportArray(wm.cellMap, g));
          cachedRender = { grid: g, probs: nr2.probs, maxProb: nr2.maxProb };
          paintOverlay();
        }
      }
      var nrF = normaliseProbs(buildViewportArray(getCellMap(cacheKey(key, selectedWeek), g.step), g));
      cachedRender = { grid: g, probs: nrF.probs, maxProb: nrF.maxProb };
      paintOverlay();
      setStatus(lbl.common + " \u2013 " + weekText(selectedWeek) + " \u00b7 " + totalPoints.toLocaleString() + " cells (" + g.step + "\u00b0)");
      updateLegend();
    } catch (e) { setStatus("Error: " + e.message); console.error(e); }
    finally { rendering = false; showComputingOverlay(false); if (gen !== renderGeneration) triggerRender(); }
  }

  // ---- Cached week switch --------------------------------------------------
  function showCachedWeek() {
    var week = +document.getElementById("week-select").value;
    var g = viewportGrid();

    if (currentMode === "richness") {
      var cm = getCellMap(cacheKey("__richness__", week), g.step);
      if (viewportMissing(cm, g).length === 0) {
        var raw = buildViewportArray(cm, g);
        var maxVal = 0;
        for (var i = 0; i < raw.length; i++) if (raw[i] > maxVal) maxVal = raw[i];
        cachedRender = { grid: g, probs: perceptualNorm(raw, maxVal), maxVal: maxVal, product: "richness" };
        paintOverlay();
        setStatus("Species richness \u2013 " + weekText(week) + " \u00b7 " + (g.nLat * g.nLon) + " cells (" + g.step + "\u00b0) [cached]");
        updateLegend();
      } else { renderRichness(); }
      return;
    }

    var key = document.getElementById("species-search").dataset.selectedKey;
    if (!key || !labelsByKey[key]) return;
    var cm2 = getCellMap(cacheKey(key, week), g.step);
    if (viewportMissing(cm2, g).length === 0) {
      var nr = normaliseProbs(buildViewportArray(cm2, g));
      cachedRender = { grid: g, probs: nr.probs, maxProb: nr.maxProb };
      paintOverlay();
      setStatus(labelsByKey[key].common + " \u2013 " + weekText(week) + " \u00b7 " + (g.nLat * g.nLon) + " cells (" + g.step + "\u00b0) [cached]");
      updateLegend();
    } else { renderRangeMap(); }
  }

  // ---- Species richness ----------------------------------------------------
  var RICHNESS_THRESHOLD = 0.05;

  async function renderRichness() {
    if (rendering) { renderGeneration++; return; }
    var gen = ++renderGeneration;
    var selectedWeek = +document.getElementById("week-select").value;
    var weeks = allWeeks(), nSpecies = labels.length, CHUNK = 4096;
    var g = viewportGrid(), totalPoints = g.nLat * g.nLon;

    var weekMissing = [];
    weeks.forEach(function (w) {
      var cm = getCellMap(cacheKey("__richness__", w), g.step);
      var miss = viewportMissing(cm, g);
      if (miss.length > 0) weekMissing.push({ week: w, missing: miss, cellMap: cm });
    });

    if (weekMissing.length === 0) {
      var raw = buildViewportArray(getCellMap(cacheKey("__richness__", selectedWeek), g.step), g);
      var maxVal = 0;
      for (var i = 0; i < raw.length; i++) if (raw[i] > maxVal) maxVal = raw[i];
      cachedRender = { grid: g, probs: perceptualNorm(raw, maxVal), maxVal: maxVal, product: "richness" };
      paintOverlay();
      setStatus("Species richness \u2013 " + weekText(selectedWeek) + " \u00b7 " + totalPoints.toLocaleString() + " cells (" + g.step + "\u00b0) [cached]");
      updateLegend();
      return;
    }

    rendering = true;
    showComputingOverlay(true, "species richness");
    try {
      for (var wi = 0; wi < weekMissing.length; wi++) {
        var wm = weekMissing[wi];
        setStatus("Computing richness \u2013 " + weekText(wm.week) + " \u00b7 " + wm.missing.length + " new cells [" + (wi + 1) + "/" + weekMissing.length + "]\u2026");
        var inputs = new Float32Array(wm.missing.length * 3);
        for (var ii = 0; ii < wm.missing.length; ii++) {
          inputs[ii * 3] = wm.missing[ii].lat;
          inputs[ii * 3 + 1] = wm.missing[ii].lon;
          inputs[ii * 3 + 2] = wm.week;
        }
        var counts = new Float32Array(wm.missing.length);
        for (var start = 0; start < wm.missing.length; start += CHUNK) {
          if (gen !== renderGeneration) return;
          var end = Math.min(start + CHUNK, wm.missing.length);
          var out = await runInference(inputs.subarray(start * 3, end * 3), end - start);
          for (var j = 0; j < end - start; j++) {
            var count = 0, base = j * nSpecies;
            for (var s = 0; s < nSpecies; s++) if (out[base + s] >= RICHNESS_THRESHOLD) count++;
            counts[start + j] = count;
          }
        }
        if (gen !== renderGeneration) return;
        for (var k = 0; k < wm.missing.length; k++) wm.cellMap.set(cellId(wm.missing[k].lat, wm.missing[k].lon), counts[k]);
        if (wm.week === selectedWeek) {
          var rawW = buildViewportArray(wm.cellMap, g);
          var maxV = 0;
          for (var m = 0; m < rawW.length; m++) if (rawW[m] > maxV) maxV = rawW[m];
          cachedRender = { grid: g, probs: perceptualNorm(rawW, maxV), maxVal: maxV, product: "richness" };
          paintOverlay();
        }
      }
      var rawF = buildViewportArray(getCellMap(cacheKey("__richness__", selectedWeek), g.step), g);
      var maxF = 0;
      for (var n = 0; n < rawF.length; n++) if (rawF[n] > maxF) maxF = rawF[n];
      cachedRender = { grid: g, probs: perceptualNorm(rawF, maxF), maxVal: maxF, product: "richness" };
      paintOverlay();
      setStatus("Species richness \u2013 " + weekText(selectedWeek) + " \u00b7 " + totalPoints.toLocaleString() + " cells (" + g.step + "\u00b0)");
      updateLegend();
    } catch (e) { setStatus("Error: " + e.message); console.error(e); }
    finally { rendering = false; showComputingOverlay(false); if (gen !== renderGeneration) triggerRender(); }
  }

  // ---- Species list --------------------------------------------------------
  function onMapClick(e) {
    if (currentMode !== "list") return;
    if (marker) map.removeLayer(marker);
    marker = L.marker([e.latlng.lat, e.latlng.lng]).addTo(map);
    renderSpeciesList(e.latlng.lat, e.latlng.lng);
  }

  async function renderSpeciesList(lat, lon) {
    var week = +document.getElementById("week-select").value;
    var threshold = +document.getElementById("threshold-select").value / 100;
    setStatus("Predicting species at (" + lat.toFixed(2) + ", " + lon.toFixed(2) + ") week " + week + "\u2026");
    try {
      var out = await runInference(new Float32Array([lat, lon, week]), 1);
      var results = [];
      for (var i = 0; i < labels.length; i++) {
        if (out[i] >= threshold) results.push({ label: labels[i], prob: out[i] });
      }
      results.sort(function (a, b) { return b.prob - a.prob; });
      document.getElementById("sp-coords").textContent =
        lat.toFixed(4) + "\u00b0, " + lon.toFixed(4) + "\u00b0 \u00b7 Week " + week + " \u00b7 " + results.length + " species above " + (threshold * 100).toFixed(0) + "%";
      document.getElementById("sp-tbody").innerHTML = results.map(function (r, idx) {
        return '<tr><td>' + (idx + 1) + '</td><td>' + r.label.common + '</td><td style="font-style:italic">' +
               r.label.sci + '</td><td>' + (r.prob * 100).toFixed(1) + '%</td><td class="prob-bar-cell"><div class="prob-bar" style="width:' +
               Math.round(r.prob * 100) + '%"></div></td></tr>';
      }).join("");
      document.getElementById("species-panel").style.display = "block";
      setStatus(results.length + " species above " + (threshold * 100).toFixed(0) + "% at (" + lat.toFixed(2) + ", " + lon.toFixed(2) + ")");
    } catch (e) { setStatus("Error: " + e.message); console.error(e); }
  }

  // ---- Computing overlay ---------------------------------------------------
  function showComputingOverlay(show, name) {
    var el = document.getElementById("demo-computing");
    if (!el) return;
    el.style.display = show ? "flex" : "none";
    if (show) {
      document.getElementById("computing-text").textContent = "Computing " + (name || "") + "\u2026";
      document.getElementById("computing-progress-bar").style.width = "0%";
    }
  }

  // ---- Legend ---------------------------------------------------------------
  function updateLegend() {
    var el = document.getElementById("demo-legend");
    if (!el) return;
    if ((currentMode !== "range" && currentMode !== "richness") || !cachedRender) { el.style.display = "none"; return; }

    var isRichness = currentMode === "richness";
    var maxVal = isRichness && cachedRender.maxVal ? Math.round(cachedRender.maxVal) : 0;
    var maxProb = !isRichness && cachedRender.maxProb ? cachedRender.maxProb : 1;

    var html = '<div class="legend-title">' + (isRichness ? "Predicted species count" : "Occurrence probability") + '</div><div class="legend-bar">';
    var stops = [];
    for (var i = 0; i <= 10; i++) {
      var t = i / 10, c = colormapLookup(t);
      stops.push("rgb(" + c[0] + "," + c[1] + "," + c[2] + ") " + Math.round(t * 100) + "%");
    }
    html += '<div class="legend-gradient" style="background:linear-gradient(to right,' + stops.join(",") + ')"></div>';
    html += '<div class="legend-ticks">';
    [0, 0.5, 1].forEach(function (t) {
      var rawT = Math.pow(t, 1 / DISPLAY_GAMMA);
      html += "<span>" + (isRichness ? Math.round(rawT * maxVal) : Math.round(rawT * maxProb * 100) + "%") + "</span>";
    });
    html += "</div></div>";
    el.innerHTML = html;
    el.style.display = "block";
  }

})();
