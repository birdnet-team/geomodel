---
hide:
  - navigation
  - toc
---

# Interactive Demo

<p style="margin-bottom: 0.5em;">
Run the BirdNET Geomodel directly in your browser — no server required.
The ONNX FP16 model (~14 MB) is loaded once and all inference happens locally via
<a href="https://onnxruntime.ai/docs/tutorials/web/" target="_blank">ONNX Runtime Web</a>.
</p>

<div id="demo-root"></div>

<div markdown="0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<link rel="stylesheet" href="demo.css">
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<script src="demo.js"></script>
