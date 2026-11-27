---
hide:
  - navigation
  - toc
---

# Interactive Demo

<p style="margin-bottom: 0.5em;">
The geomodel running in your browser: the FP16 ONNX model is loaded once and every
prediction is computed locally with
<a href="https://onnxruntime.ai/docs/tutorials/web/" target="_blank">ONNX Runtime Web</a>.
Pick a species for its weekly range, click the map for the species list of a place,
or switch to species richness or weekly bar charts.
</p>

<div id="demo-root"></div>

!!! note "What the map shows"
    Scores follow how often a species is encountered at a place and time, learned
    from citizen-science records. Where few people record anything, the model fills
    the gaps from similar places and errs on the generous side, because it is built
    to filter audio detections. It is not a definitive range map; for conservation
    or legal questions use authoritative sources such as IUCN, BirdLife or local atlases.

<div markdown="0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<link rel="stylesheet" href="demo.css">
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<script src="demo.js"></script>
