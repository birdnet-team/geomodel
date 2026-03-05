/**
 * BirdNET Geomodel – Interactive Web Demo
 *
 * Runs the ONNX FP16 model entirely client-side via ONNX Runtime Web.
 * Two modes:
 *   1. Range Map  – renders a species probability heatmap on a Leaflet map
 *   2. Species List – click a location to see predicted species for that spot
 *
 * The model input is (batch, 3) = [lat, lon, week] and output is
 * (batch, n_species) sigmoid probabilities.
 */

(function () {
  "use strict";

  // ---- Configuration -------------------------------------------------------
  const MODEL_URL = "geomodel_fp16.onnx";
  const LABELS_URL = "labels.txt";

  // Grid resolution per zoom level (degrees per cell)
  const ZOOM_STEP = { 2: 3, 3: 2, 4: 1 };

  // Preselected species (taxonKey → common name for quick access)
  // Curated to showcase migrants, residents, endemics, and non-birds
  const FEATURED_SPECIES = [
    // ── Long-distance migrants (dramatic week 1 vs 26 difference) ──
    { key: "9515886", sci: "Hirundo rustica", common: "Barn Swallow" },
    { key: "5229230", sci: "Sterna paradisaea", common: "Arctic Tern" },
    { key: "5231918", sci: "Cuculus canorus", common: "Common Cuckoo" },
    { key: "2493052", sci: "Phylloscopus trochilus", common: "Willow Warbler" },
    // ── Residents (stable year-round range) ──
    { key: "9705453", sci: "Parus major", common: "Great Tit" },
    { key: "2490384", sci: "Cardinalis cardinalis", common: "Northern Cardinal" },
    { key: "9510564", sci: "Turdus migratorius", common: "American Robin" },
    // ── Endemics (restricted range) ──
    { key: "5232445", sci: "Branta sandvicensis", common: "Hawaiian Goose (Nēnē)" },
    { key: "2479593", sci: "Nestor notabilis", common: "Kea" },
    { key: "2480569", sci: "Buteo galapagoensis", common: "Galápagos Hawk" },
    { key: "2495144", sci: "Apteryx mantelli", common: "North Island Brown Kiwi" },
    // ── Non-birds ──
    { key: "5219243", sci: "Vulpes vulpes", common: "Red Fox" },
    { key: "2436940", sci: "Oryctolagus cuniculus", common: "European Rabbit" },
    { key: "5219681", sci: "Sciurus carolinensis", common: "Eastern Gray Squirrel" },
    { key: "5218786", sci: "Procyon lotor", common: "Common Raccoon" },
  ];

  // Viridis-like colour ramp (prob 0→1)
  const COLORMAP = buildViridis();

  // ---- State ---------------------------------------------------------------
  let worker = null; // Web Worker for ONNX inference
  let inferenceId = 0; // monotonic request counter
  const pendingInferences = new Map(); // id → { resolve, reject }
  let labels = []; // [{key, sci, common, index}, ...]
  let labelsByKey = {}; // taxonKey → label object
  let map = null; // Leaflet map
  let overlayCanvas = null; // Canvas element in Leaflet overlay pane
  let cachedRender = null; // { grid, probs } for repaint on pan/zoom
  const renderCache = new Map(); // "speciesKey:week" → { step, cells: Map<cellKey,rawValue> }
  const RENDER_CACHE_MAX = 50; // evict oldest entries beyond this
  let marker = null; // Leaflet marker for species-list mode
  let currentMode = "range"; // 'range' | 'richness' | 'list'
  let rendering = false;
  let renderGeneration = 0; // incremented to cancel stale renders
  let moveEndTimer = null; // debounce timer for map move/zoom

  // ---- Bootstrap -----------------------------------------------------------
  // Capture script location at parse time (before DOMContentLoaded fires)
  const SCRIPT_BASE = (function () {
    if (document.currentScript && document.currentScript.src)
      return document.currentScript.src;
    return window.location.href;
  })();

  document.addEventListener("DOMContentLoaded", init);

  async function init() {
    const root = document.getElementById("demo-root");
    if (!root) return;

    root.innerHTML = `
      <div id="demo-loading"><div class="spinner"></div>Loading model &amp; labels…</div>
      <div id="demo-app" style="display:none">
        <div id="demo-controls">
          <div class="ctrl-group">
            <label for="mode-select">Mode</label>
            <select id="mode-select">
              <option value="range">Species Range</option>
              <option value="richness">Species Richness</option>
              <option value="list">Species List (click map)</option>
            </select>
          </div>
          <div class="ctrl-group" id="species-search-wrap">
            <label for="species-search">Species</label>
            <input id="species-search" type="text" autocomplete="off"
                   placeholder="Search species…" />
            <div id="species-results"></div>
          </div>
          <div class="ctrl-group" id="week-select-wrap">
            <label for="week-select">Week</label>
            <select id="week-select">
              <option value="1">Week 1 (early Jan)</option>
              <option value="26">Week 26 (late Jun)</option>
            </select>
          </div>
          <div class="ctrl-group" id="threshold-wrap" style="display:none">
            <label for="threshold-select">Min probability</label>
            <select id="threshold-select">
              <option value="1">1%</option>
              <option value="5" selected>5%</option>
              <option value="10">10%</option>
              <option value="25">25%</option>
              <option value="50">50%</option>
            </select>
          </div>
        </div>
        <div id="demo-status">&nbsp;</div>
        <div id="demo-map-wrap">
          <div id="demo-map"></div>
          <div id="demo-computing" style="display:none">
            <div class="spinner"></div>
            <div id="computing-text">Computing…</div>
            <div id="computing-progress-wrap"><div id="computing-progress-bar"></div></div>
          </div>
          <div id="demo-legend"></div>
        </div>
        <div id="species-panel">
          <h3 id="sp-title">Species at location</h3>
          <div class="sp-coords" id="sp-coords"></div>
          <table id="species-list-table">
            <thead><tr><th>#</th><th>Species</th><th>Scientific name</th><th>Probability</th><th></th></tr></thead>
            <tbody id="sp-tbody"></tbody>
          </table>
        </div>
      </div>`;

    try {
      await Promise.all([initWorker(), loadLabels()]);
      document.getElementById("demo-loading").style.display = "none";
      document.getElementById("demo-app").style.display = "block";
      initMap();
      bindControls();
      setStatus("Select a species to view its predicted range map.");
    } catch (e) {
      document.getElementById("demo-loading").innerHTML =
        `<span style="color:red">Failed to load: ${e.message}</span>`;
      console.error(e);
    }
  }

  // ---- Model & labels loading ---------------------------------------------
  async function initWorker() {
    setStatus("Loading ONNX model…");
    const modelUrl = resolveUrl(MODEL_URL);
    const workerUrl = resolveUrl("inference-worker.js");
    worker = new Worker(workerUrl);
    worker.onerror = (err) => console.error("Worker error:", err);

    await new Promise((resolve, reject) => {
      worker.onmessage = (e) => {
        if (e.data.type === "init") {
          if (e.data.ok) resolve();
          else reject(new Error(e.data.error || "Worker init failed"));
        }
      };
      worker.postMessage({ type: "init", modelUrl });
    });

    // Switch to inference message handler after init
    worker.onmessage = handleWorkerMessage;
  }

  /** Handle inference results from the worker. */
  function handleWorkerMessage(e) {
    const msg = e.data;
    if (msg.type !== "infer") return;
    const pending = pendingInferences.get(msg.id);
    if (!pending) return;
    pendingInferences.delete(msg.id);
    if (msg.error) pending.reject(new Error(msg.error));
    else pending.resolve(new Float32Array(msg.data));
  }

  async function loadLabels() {
    const url = resolveUrl(LABELS_URL);
    const resp = await fetch(url);
    const text = await resp.text();
    labels = text
      .trim()
      .split("\n")
      .map((line, i) => {
        const parts = line.split("\t");
        return {
          key: parts[0],
          sci: parts[1] || "",
          common: parts[2] || parts[1] || "",
          index: i,
        };
      });
    labelsByKey = {};
    labels.forEach((l) => (labelsByKey[l.key] = l));
  }

  function resolveUrl(relative) {
    return new URL(relative, SCRIPT_BASE).href;
  }

  // ---- Map setup -----------------------------------------------------------
  function initMap() {
    map = L.map("demo-map", {
      center: [30, 0],
      zoom: 2,
      minZoom: 2,
      maxZoom: 4,
    });

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
      maxZoom: 4,
      subdomains: "abcd",
    }).addTo(map);

    map.on("click", onMapClick);

    // Repaint cached overlay immediately on pan/zoom for smooth tracking,
    // then debounce a full re-render (inference) for the new viewport.
    map.on("moveend", () => {
      if (currentMode !== "range" && currentMode !== "richness") return;
      paintOverlay(); // instant repaint of cached cells
      clearTimeout(moveEndTimer);
      moveEndTimer = setTimeout(() => triggerRender(), 300);
    });
  }

  /** Dispatch rendering to the right product handler. */
  function triggerRender() {
    if (currentMode === "richness") renderRichness();
    else if (currentMode === "range") renderRangeMap();
  }

  // ---- Controls ------------------------------------------------------------
  function bindControls() {
    // Mode
    const modeEl = document.getElementById("mode-select");
    modeEl.addEventListener("change", () => {
      currentMode = modeEl.value;
      const isMap = currentMode === "range" || currentMode === "richness";
      document.getElementById("species-search-wrap").style.display =
        currentMode === "range" ? "" : "none";
      document.getElementById("threshold-wrap").style.display =
        currentMode === "list" ? "" : "none";
      document.getElementById("species-panel").style.display = "none";
      if (cachedRender) { clearOverlay(); }
      if (marker) { map.removeLayer(marker); marker = null; }
      updateLegend();
      if (isMap) triggerRender();
    });

    // Week dropdown – show cached result (both weeks pre-computed)
    const weekEl = document.getElementById("week-select");
    weekEl.addEventListener("change", () => {
      if (currentMode === "range" || currentMode === "richness") showCachedWeek();
      else if (marker) {
        const ll = marker.getLatLng();
        renderSpeciesList(ll.lat, ll.lng);
      }
      updateLegend();
    });

    // Threshold dropdown (species-list mode)
    const threshEl = document.getElementById("threshold-select");
    threshEl.addEventListener("change", () => {
      if (currentMode === "list" && marker) {
        const ll = marker.getLatLng();
        renderSpeciesList(ll.lat, ll.lng);
      }
    });

    // Species search
    const searchEl = document.getElementById("species-search");
    const resultsEl = document.getElementById("species-results");
    let selIdx = -1;

    searchEl.addEventListener("focus", () => showSearch(searchEl, resultsEl));
    searchEl.addEventListener("input", () => {
      selIdx = -1;
      showSearch(searchEl, resultsEl);
    });
    searchEl.addEventListener("keydown", (e) => {
      const items = resultsEl.querySelectorAll(".sr-item");
      if (e.key === "ArrowDown") {
        e.preventDefault();
        selIdx = Math.min(selIdx + 1, items.length - 1);
        highlightItem(items, selIdx);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        selIdx = Math.max(selIdx - 1, 0);
        highlightItem(items, selIdx);
      } else if (e.key === "Enter" && selIdx >= 0 && items[selIdx]) {
        e.preventDefault();
        items[selIdx].click();
      } else if (e.key === "Escape") {
        resultsEl.style.display = "none";
      }
    });
    document.addEventListener("click", (e) => {
      if (!resultsEl.contains(e.target) && e.target !== searchEl)
        resultsEl.style.display = "none";
    });
  }

  function showSearch(inputEl, resultsEl) {
    const q = inputEl.value.trim().toLowerCase();
    let matches;
    if (q.length === 0) {
      // Show featured species
      matches = FEATURED_SPECIES.map((f) => labelsByKey[f.key]).filter(Boolean);
    } else {
      matches = labels
        .filter(
          (l) =>
            l.common.toLowerCase().includes(q) ||
            l.sci.toLowerCase().includes(q) ||
            l.key.includes(q)
        )
        .slice(0, 30);
    }
    resultsEl.innerHTML = matches
      .map(
        (l) =>
          `<div class="sr-item" data-key="${l.key}">${l.common} <span class="sr-sci">${l.sci}</span></div>`
      )
      .join("");
    resultsEl.style.display = matches.length ? "block" : "none";
    resultsEl.querySelectorAll(".sr-item").forEach((el) => {
      el.addEventListener("click", () => {
        selectSpecies(el.dataset.key);
        inputEl.value = "";
        resultsEl.style.display = "none";
      });
    });
  }

  function highlightItem(items, idx) {
    items.forEach((el, i) => el.classList.toggle("active", i === idx));
    if (items[idx]) items[idx].scrollIntoView({ block: "nearest" });
  }

  function selectSpecies(key) {
    const lbl = labelsByKey[key];
    if (!lbl) return;
    const searchEl = document.getElementById("species-search");
    searchEl.placeholder = `${lbl.common} (${lbl.sci})`;
    searchEl.dataset.selectedKey = key;
    if (currentMode === "range") renderRangeMap();
  }

  // ---- Inference -----------------------------------------------------------
  /**
   * Run model on a batch of inputs.
   * @param {Float32Array} flatInputs - Flat array of [lat,lon,week, lat,lon,week, ...]
   * @param {number} batchSize
   * @returns {Float32Array} - Flat output probabilities (batchSize * n_species)
   */
  async function runInference(flatInputs, batchSize) {
    const id = ++inferenceId;
    return new Promise((resolve, reject) => {
      pendingInferences.set(id, { resolve, reject });
      const buf = new Float32Array(flatInputs).buffer;
      worker.postMessage(
        { type: "infer", id, flatInputs: buf, batchSize },
        [buf]
      );
    });
  }

  // ---- Range map rendering -------------------------------------------------

  /**
   * Ensure the overlay canvas exists in Leaflet's overlay pane.
   * The canvas is positioned and repainted on every view change.
   */
  function ensureOverlayCanvas() {
    if (overlayCanvas) return;
    const pane = map.getPane("overlayPane");
    overlayCanvas = document.createElement("canvas");
    overlayCanvas.className = "heatmap-overlay";
    overlayCanvas.style.position = "absolute";
    overlayCanvas.style.pointerEvents = "none";
    pane.appendChild(overlayCanvas);
  }

  /** Clear the overlay canvas and cached data. */
  function clearOverlay() {
    cachedRender = null;
    if (overlayCanvas) {
      overlayCanvas.width = 0;
      overlayCanvas.height = 0;
    }
  }

  /**
   * Repaint the cached grid cells onto the overlay canvas using
   * Mercator-correct screen-pixel positioning.
   */
  function paintOverlay() {
    if (!cachedRender || !map) return;
    ensureOverlayCanvas();

    const { grid: g, probs } = cachedRender;
    const size = map.getSize();
    const topLeft = map.containerPointToLayerPoint([0, 0]);

    overlayCanvas.width = size.x;
    overlayCanvas.height = size.y;
    L.DomUtil.setPosition(overlayCanvas, topLeft);

    const ctx = overlayCanvas.getContext("2d");

    let pi = 0;
    for (let iLat = 0; iLat < g.nLat; iLat++) {
      const latN = g.north - iLat * g.step;
      const latS = latN - g.step;
      for (let iLon = 0; iLon < g.nLon; iLon++) {
        const lonW = g.west + iLon * g.step;
        const lonE = lonW + g.step;
        const p = probs[pi++];
        if (p < 0.01) continue;

        // Project cell corners to container (screen) pixels
        const nw = map.latLngToContainerPoint([latN, lonW]);
        const se = map.latLngToContainerPoint([latS, lonE]);

        const x = Math.floor(nw.x);
        const y = Math.floor(nw.y);
        const w = Math.ceil(se.x) - x;
        const h = Math.ceil(se.y) - y;

        // Skip cells entirely off-screen
        if (x + w < 0 || y + h < 0 || x > size.x || y > size.y) continue;

        const [cr, cg, cb] = colormapLookup(p);
        const alpha = Math.min(1, 0.25 + p * 0.75);
        ctx.fillStyle = `rgba(${cr},${cg},${cb},${alpha.toFixed(3)})`;
        ctx.fillRect(x, y, w, h);
      }
    }
  }

  /**
   * Compute the grid step and bounds for the current map viewport.
   * Returns { south, north, west, east, step, nLat, nLon }.
   */
  function viewportGrid() {
    const b = map.getBounds();
    let south = Math.max(b.getSouth(), -90);
    let north = Math.min(b.getNorth(), 90);
    let west = b.getWest();
    let east = b.getEast();

    // Wrap longitude into [-180, 180] range for the model,
    // but allow spans wider than 360° (Leaflet shows wrapped tiles)
    if (east - west >= 360) {
      west = -180;
      east = 180;
    } else {
      west = wrapLon(west);
      east = wrapLon(east);
      if (east <= west) east += 360; // viewport crosses antimeridian
    }

    // Safety: ensure non-degenerate bounds
    if (north - south < 0.1) north = south + 0.1;
    if (east - west < 0.1) east = west + 0.1;

    const step = ZOOM_STEP[map.getZoom()] || 3;

    // Expand bounds to align to step grid
    south = Math.floor(south / step) * step;
    north = Math.ceil(north / step) * step;
    west  = Math.floor(west / step) * step;
    east  = Math.ceil(east / step) * step;
    south = Math.max(south, -90);
    north = Math.min(north, 90);

    const nLat = Math.round((north - south) / step);
    const nLon = Math.round((east - west) / step);

    return { south, north, west, east, step, nLat, nLon };
  }

  /**
   * Build a cache key string for the render cache.
   */
  function renderCacheKey(speciesKey, week) {
    return `${speciesKey}:${week}`;
  }

  // ---- Cell-level cache helpers -------------------------------------------
  const wrapLon = (v) => ((((v + 180) % 360) + 360) % 360) - 180;

  /** Stable string key for a grid cell centre. */
  function cellKey(lat, lon) {
    return Math.round(lat * 100) + "," + Math.round(lon * 100);
  }

  /** Get (or create) the cell Map for a cache entry, resetting if step changed. */
  function getCellMap(cacheKey, step) {
    let entry = renderCache.get(cacheKey);
    if (!entry || entry.step !== step) {
      entry = { step, cells: new Map() };
      renderCache.set(cacheKey, entry);
      if (renderCache.size > RENDER_CACHE_MAX) {
        const oldest = renderCache.keys().next().value;
        renderCache.delete(oldest);
      }
    }
    return entry.cells;
  }

  /** Return array of {lat, lon} for viewport cells missing from cellMap. */
  function viewportMissing(cellMap, grid) {
    const pts = [];
    for (let iLat = 0; iLat < grid.nLat; iLat++) {
      const lat = grid.north - (iLat + 0.5) * grid.step;
      for (let iLon = 0; iLon < grid.nLon; iLon++) {
        const lon = wrapLon(grid.west + (iLon + 0.5) * grid.step);
        if (!cellMap.has(cellKey(lat, lon))) pts.push({ lat, lon });
      }
    }
    return pts;
  }

  /** Assemble a flat Float32Array of values from cellMap for the viewport grid. */
  function buildViewportArray(cellMap, grid) {
    const n = grid.nLat * grid.nLon;
    const arr = new Float32Array(n);
    let i = 0;
    for (let iLat = 0; iLat < grid.nLat; iLat++) {
      const lat = grid.north - (iLat + 0.5) * grid.step;
      for (let iLon = 0; iLon < grid.nLon; iLon++) {
        const lon = wrapLon(grid.west + (iLon + 0.5) * grid.step);
        arr[i++] = cellMap.get(cellKey(lat, lon)) || 0;
      }
    }
    return arr;
  }

  /** Normalise raw probs → [0,1] and return { probs, maxProb }. */
  function normaliseProbs(raw) {
    let maxProb = 0;
    for (let i = 0; i < raw.length; i++) if (raw[i] > maxProb) maxProb = raw[i];
    const probs = new Float32Array(raw.length);
    if (maxProb > 0) for (let i = 0; i < raw.length; i++) probs[i] = raw[i] / maxProb;
    return { probs, maxProb };
  }

  /** All week values from the week dropdown. */
  function allWeeks() {
    const opts = document.getElementById("week-select").options;
    return Array.from(opts, (o) => +o.value);
  }

  /**
   * Render the range map for the currently selected species.
   * Computes ALL weeks in the dropdown so switching is instant.
   * Only infers cells that are not already in the cell-level cache.
   */
  async function renderRangeMap() {
    const key = document.getElementById("species-search").dataset.selectedKey;
    if (!key || !labelsByKey[key]) return;
    if (rendering) {
      renderGeneration++;
      return;
    }
    const gen = ++renderGeneration;
    const lbl = labelsByKey[key];
    const speciesIdx = lbl.index;
    const selectedWeek = +document.getElementById("week-select").value;
    const weeks = allWeeks();
    const nSpecies = labels.length;
    const CHUNK = 4096;

    const g = viewportGrid();
    const totalPoints = g.nLat * g.nLon;

    // Determine which weeks have missing cells
    const weekMissing = []; // [{week, missing, cellMap}]
    for (const w of weeks) {
      const cm = getCellMap(renderCacheKey(key, w), g.step);
      const miss = viewportMissing(cm, g);
      if (miss.length > 0) weekMissing.push({ week: w, missing: miss, cellMap: cm });
    }

    // Fast path: every viewport cell is cached for all weeks
    if (weekMissing.length === 0) {
      const cm = getCellMap(renderCacheKey(key, selectedWeek), g.step);
      const raw = buildViewportArray(cm, g);
      const { probs, maxProb } = normaliseProbs(raw);
      cachedRender = { grid: g, probs, maxProb };
      paintOverlay();
      setStatus(
        `${lbl.common} – ${weekText(selectedWeek)} · ${totalPoints.toLocaleString()} cells (${g.step}°) [cached]`
      );
      updateLegend();
      return;
    }

    rendering = true;
    showComputingOverlay(true, lbl.common);

    try {
      for (let wi = 0; wi < weekMissing.length; wi++) {
        const { week, missing, cellMap } = weekMissing[wi];
        setStatus(
          `Computing ${lbl.common} – ${weekText(week)} · ${missing.length} new cells (${g.step}°) [${wi + 1}/${weekMissing.length}]…`
        );

        // Build input batch for missing cells only
        const nMiss = missing.length;
        const inputs = new Float32Array(nMiss * 3);
        for (let i = 0; i < nMiss; i++) {
          inputs[i * 3] = missing[i].lat;
          inputs[i * 3 + 1] = missing[i].lon;
          inputs[i * 3 + 2] = week;
        }

        // Run inference in chunks
        const rawProbs = new Float32Array(nMiss);
        for (let start = 0; start < nMiss; start += CHUNK) {
          if (gen !== renderGeneration) return;
          const end = Math.min(start + CHUNK, nMiss);
          const batchSize = end - start;
          const chunk = inputs.subarray(start * 3, end * 3);
          const out = await runInference(chunk, batchSize);
          for (let i = 0; i < batchSize; i++) {
            rawProbs[start + i] = out[i * nSpecies + speciesIdx];
          }
        }

        if (gen !== renderGeneration) return;

        // Merge results into cell map
        for (let i = 0; i < nMiss; i++) {
          cellMap.set(cellKey(missing[i].lat, missing[i].lon), rawProbs[i]);
        }

        // Paint selected week immediately as cells arrive
        if (week === selectedWeek) {
          const raw = buildViewportArray(cellMap, g);
          const { probs, maxProb } = normaliseProbs(raw);
          cachedRender = { grid: g, probs, maxProb };
          paintOverlay();
        }
      }

      // Final paint for selected week
      const cm = getCellMap(renderCacheKey(key, selectedWeek), g.step);
      const raw = buildViewportArray(cm, g);
      const { probs, maxProb } = normaliseProbs(raw);
      cachedRender = { grid: g, probs, maxProb };
      paintOverlay();

      setStatus(
        `${lbl.common} – ${weekText(selectedWeek)} · ${totalPoints.toLocaleString()} cells (${g.step}°)`
      );
      updateLegend();
    } catch (e) {
      setStatus(`Error: ${e.message}`);
      console.error(e);
    } finally {
      rendering = false;
      showComputingOverlay(false);
      if (gen !== renderGeneration) triggerRender();
    }
  }

  /**
   * Show the cached render for the currently selected week (no inference).
   * Called when the week dropdown changes – results are already computed.
   */
  function showCachedWeek() {
    const week = +document.getElementById("week-select").value;
    const g = viewportGrid();

    if (currentMode === "richness") {
      const cm = getCellMap(renderCacheKey("__richness__", week), g.step);
      if (viewportMissing(cm, g).length === 0) {
        const raw = buildViewportArray(cm, g);
        let maxVal = 0;
        for (let i = 0; i < raw.length; i++) if (raw[i] > maxVal) maxVal = raw[i];
        const probs = new Float32Array(raw.length);
        if (maxVal > 0) for (let i = 0; i < raw.length; i++) probs[i] = raw[i] / maxVal;
        cachedRender = { grid: g, probs, maxVal, product: "richness" };
        paintOverlay();
        setStatus(`Species richness – ${weekText(week)} · ${g.nLat * g.nLon} cells (${g.step}°) [cached]`);
        updateLegend();
      } else {
        renderRichness();
      }
      return;
    }

    const key = document.getElementById("species-search").dataset.selectedKey;
    if (!key || !labelsByKey[key]) return;
    const cm = getCellMap(renderCacheKey(key, week), g.step);
    if (viewportMissing(cm, g).length === 0) {
      const raw = buildViewportArray(cm, g);
      const { probs, maxProb } = normaliseProbs(raw);
      cachedRender = { grid: g, probs, maxProb };
      paintOverlay();
      setStatus(
        `${labelsByKey[key].common} – ${weekText(week)} · ${g.nLat * g.nLon} cells (${g.step}°) [cached]`
      );
      updateLegend();
    } else {
      renderRangeMap();
    }
  }

  // ---- Species richness rendering ------------------------------------------
  const RICHNESS_THRESHOLD = 0.05; // count species with prob > 5%

  /**
   * Render species richness: number of species predicted above threshold
   * at each grid cell. Computes all weeks at once.
   */
  async function renderRichness() {
    if (rendering) {
      renderGeneration++;
      return;
    }
    const gen = ++renderGeneration;
    const selectedWeek = +document.getElementById("week-select").value;
    const weeks = allWeeks();
    const nSpecies = labels.length;
    const CHUNK = 4096;

    const g = viewportGrid();
    const totalPoints = g.nLat * g.nLon;

    // Determine which weeks have missing cells
    const weekMissing = [];
    for (const w of weeks) {
      const cm = getCellMap(renderCacheKey("__richness__", w), g.step);
      const miss = viewportMissing(cm, g);
      if (miss.length > 0) weekMissing.push({ week: w, missing: miss, cellMap: cm });
    }

    // Fast path: all cached
    if (weekMissing.length === 0) {
      const cm = getCellMap(renderCacheKey("__richness__", selectedWeek), g.step);
      const raw = buildViewportArray(cm, g);
      let maxVal = 0;
      for (let i = 0; i < raw.length; i++) if (raw[i] > maxVal) maxVal = raw[i];
      const probs = new Float32Array(raw.length);
      if (maxVal > 0) for (let i = 0; i < raw.length; i++) probs[i] = raw[i] / maxVal;
      cachedRender = { grid: g, probs, maxVal, product: "richness" };
      paintOverlay();
      setStatus(`Species richness – ${weekText(selectedWeek)} · ${totalPoints.toLocaleString()} cells (${g.step}°) [cached]`);
      updateLegend();
      return;
    }

    rendering = true;
    showComputingOverlay(true, "species richness");

    try {
      for (let wi = 0; wi < weekMissing.length; wi++) {
        const { week, missing, cellMap } = weekMissing[wi];
        setStatus(
          `Computing richness – ${weekText(week)} · ${missing.length} new cells [${wi + 1}/${weekMissing.length}]…`
        );

        // Build input for missing cells only
        const nMiss = missing.length;
        const inputs = new Float32Array(nMiss * 3);
        for (let i = 0; i < nMiss; i++) {
          inputs[i * 3] = missing[i].lat;
          inputs[i * 3 + 1] = missing[i].lon;
          inputs[i * 3 + 2] = week;
        }

        // Run inference in chunks – count species per cell
        const counts = new Float32Array(nMiss);
        for (let start = 0; start < nMiss; start += CHUNK) {
          if (gen !== renderGeneration) return;
          const end = Math.min(start + CHUNK, nMiss);
          const batchSize = end - start;
          const chunk = inputs.subarray(start * 3, end * 3);
          const out = await runInference(chunk, batchSize);
          for (let i = 0; i < batchSize; i++) {
            let count = 0;
            const base = i * nSpecies;
            for (let s = 0; s < nSpecies; s++) {
              if (out[base + s] >= RICHNESS_THRESHOLD) count++;
            }
            counts[start + i] = count;
          }
        }

        if (gen !== renderGeneration) return;

        // Merge raw counts into cell map (store raw count, normalise at paint)
        for (let i = 0; i < nMiss; i++) {
          cellMap.set(cellKey(missing[i].lat, missing[i].lon), counts[i]);
        }

        if (week === selectedWeek) {
          const raw = buildViewportArray(cellMap, g);
          let maxVal = 0;
          for (let i = 0; i < raw.length; i++) if (raw[i] > maxVal) maxVal = raw[i];
          const probs = new Float32Array(raw.length);
          if (maxVal > 0) for (let i = 0; i < raw.length; i++) probs[i] = raw[i] / maxVal;
          cachedRender = { grid: g, probs, maxVal, product: "richness" };
          paintOverlay();
        }
      }

      // Final paint for selected week
      const cm = getCellMap(renderCacheKey("__richness__", selectedWeek), g.step);
      const raw = buildViewportArray(cm, g);
      let maxVal = 0;
      for (let i = 0; i < raw.length; i++) if (raw[i] > maxVal) maxVal = raw[i];
      const probs = new Float32Array(raw.length);
      if (maxVal > 0) for (let i = 0; i < raw.length; i++) probs[i] = raw[i] / maxVal;
      cachedRender = { grid: g, probs, maxVal, product: "richness" };
      paintOverlay();

      setStatus(`Species richness – ${weekText(selectedWeek)} · ${totalPoints.toLocaleString()} cells (${g.step}°)`);
      updateLegend();
    } catch (e) {
      setStatus(`Error: ${e.message}`);
      console.error(e);
    } finally {
      rendering = false;
      showComputingOverlay(false);
      if (gen !== renderGeneration) triggerRender();
    }
  }

  // ---- Species list on click -----------------------------------------------
  function onMapClick(e) {
    if (currentMode !== "list") return;
    const { lat, lng } = e.latlng;
    if (marker) map.removeLayer(marker);
    marker = L.marker([lat, lng]).addTo(map);
    renderSpeciesList(lat, lng);
  }

  async function renderSpeciesList(lat, lon) {
    const week = +document.getElementById("week-select").value;
    const threshold =
      +document.getElementById("threshold-select").value / 100;
    const panel = document.getElementById("species-panel");

    setStatus(`Predicting species at (${lat.toFixed(2)}, ${lon.toFixed(2)}) week ${week}…`);

    try {
      const inputs = new Float32Array([lat, lon, week]);
      const out = await runInference(inputs, 1);

      // Collect species above threshold
      const results = [];
      for (let i = 0; i < labels.length; i++) {
        if (out[i] >= threshold) {
          results.push({ label: labels[i], prob: out[i] });
        }
      }
      results.sort((a, b) => b.prob - a.prob);

      // Render table
      document.getElementById("sp-coords").textContent =
        `${lat.toFixed(4)}°, ${lon.toFixed(4)}° · Week ${week} · ${results.length} species above ${(threshold * 100).toFixed(0)}%`;
      const tbody = document.getElementById("sp-tbody");
      tbody.innerHTML = results
        .map(
          (r, i) => `<tr>
          <td>${i + 1}</td>
          <td>${r.label.common}</td>
          <td style="font-style:italic">${r.label.sci}</td>
          <td>${(r.prob * 100).toFixed(1)}%</td>
          <td class="prob-bar-cell"><div class="prob-bar" style="width:${Math.round(r.prob * 100)}%"></div></td>
        </tr>`
        )
        .join("");
      panel.style.display = "block";
      setStatus(
        `${results.length} species above ${(threshold * 100).toFixed(0)}% at (${lat.toFixed(2)}, ${lon.toFixed(2)})`
      );
    } catch (e) {
      setStatus(`Error: ${e.message}`);
      console.error(e);
    }
  }

  // ---- Computing overlay ---------------------------------------------------
  function showComputingOverlay(show, speciesName) {
    const el = document.getElementById("demo-computing");
    if (!el) return;
    el.style.display = show ? "flex" : "none";
    if (show) {
      document.getElementById("computing-text").textContent =
        `Computing ${speciesName || ""}…`;
      document.getElementById("computing-progress-bar").style.width = "0%";
    }
  }

  function updateComputingProgress(week, total, speciesName) {
    const pct = Math.round((week / total) * 100);
    const bar = document.getElementById("computing-progress-bar");
    const txt = document.getElementById("computing-text");
    if (bar) bar.style.width = pct + "%";
    if (txt) txt.textContent = `Computing ${speciesName} – week ${week}/${total}`;
  }

  // ---- Legend ---------------------------------------------------------------
  function updateLegend() {
    const el = document.getElementById("demo-legend");
    if (!el) return;

    if (currentMode !== "range" && currentMode !== "richness" || !cachedRender) {
      el.style.display = "none";
      return;
    }

    const isRichness = currentMode === "richness";
    const maxVal = isRichness && cachedRender.maxVal ? Math.round(cachedRender.maxVal) : 0;
    const maxProb = !isRichness && cachedRender.maxProb ? cachedRender.maxProb : 1;

    let html = '<div class="legend-title">' +
      (isRichness ? "Predicted species count" : "Occurrence probability") +
      '</div><div class="legend-bar">';

    // Build gradient string from colormap
    const gradStops = [];
    for (let i = 0; i <= 10; i++) {
      const t = i / 10;
      const [r, g, b] = colormapLookup(t);
      gradStops.push(`rgb(${r},${g},${b}) ${Math.round(t * 100)}%`);
    }
    html += `<div class="legend-gradient" style="background:linear-gradient(to right,${gradStops.join(",")})"></div>`;

    // Tick labels (start, middle, end only)
    html += '<div class="legend-ticks">';
    for (const t of [0, 0.5, 1]) {
      let label;
      if (isRichness) {
        label = Math.round(t * maxVal).toString();
      } else {
        label = Math.round(t * maxProb * 100) + "%";
      }
      html += `<span>${label}</span>`;
    }
    html += "</div></div>";

    el.innerHTML = html;
    el.style.display = "block";
  }

  // ---- Helpers --------------------------------------------------------------
  function setStatus(msg) {
    const el = document.getElementById("demo-status");
    if (el) el.textContent = msg;
  }

  function weekText(w) {
    // Approximate month label for a week number (1–48)
    const months = [
      "Jan", "Feb", "Mar", "Apr", "May", "Jun",
      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    const period = ["early", "mid", "late", "late"];
    const mi = Math.floor((w - 1) / 4);
    const pi = (w - 1) % 4;
    return `Week ${w} (${period[pi]} ${months[mi] || "Dec"})`;
  }

  /** Build a 256-entry inferno colour ramp (dark → fiery → bright). */
  function buildViridis() {
    const stops = [
      [0.0,   0,   0,   4],
      [0.14, 31,  12,  72],
      [0.28, 85,  15, 109],
      [0.42, 136,  8,  79],
      [0.56, 186, 54,  36],
      [0.70, 227, 105,  5],
      [0.84, 249, 174,  10],
      [1.0,  252, 255, 164],
    ];
    const ramp = new Array(256);
    for (let i = 0; i < 256; i++) {
      const t = i / 255;
      // Find surrounding stops
      let lo = stops[0],
        hi = stops[stops.length - 1];
      for (let s = 0; s < stops.length - 1; s++) {
        if (t >= stops[s][0] && t <= stops[s + 1][0]) {
          lo = stops[s];
          hi = stops[s + 1];
          break;
        }
      }
      const f = (t - lo[0]) / (hi[0] - lo[0] || 1);
      ramp[i] = [
        Math.round(lo[1] + f * (hi[1] - lo[1])),
        Math.round(lo[2] + f * (hi[2] - lo[2])),
        Math.round(lo[3] + f * (hi[3] - lo[3])),
      ];
    }
    return ramp;
  }

  function colormapLookup(prob) {
    const idx = Math.max(0, Math.min(255, Math.round(prob * 255)));
    return COLORMAP[idx];
  }
})();
