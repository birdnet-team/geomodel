/**
 * BirdNET Geomodel – Inference Web Worker
 *
 * Runs ONNX Runtime Web in a dedicated thread so the UI stays responsive.
 *
 * Protocol (postMessage):
 *   Main -> Worker:  { type: "init",  modelUrl }
 *   Worker -> Main:  { type: "init",  ok, error? }
 *   Main -> Worker:  { type: "infer", id, flatInputs: ArrayBuffer, batchSize }
 *   Worker -> Main:  { type: "infer", id, data: ArrayBuffer }
 *                   | { type: "infer", id, error }
 */

/* global ort */
// 1.18: later versions load their WebAssembly through dynamic import(), which classic workers
// in older browsers do not support
var ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
importScripts(ORT_CDN + "ort.min.js");
ort.env.wasm.wasmPaths = ORT_CDN;

var session = null;

self.onmessage = async function (e) {
  var type = e.data.type, id = e.data.id;

  if (type === "init") {
    try {
      session = await ort.InferenceSession.create(e.data.modelUrl, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      self.postMessage({ type: "init", ok: true });
    } catch (err) {
      self.postMessage({ type: "init", ok: false, error: err.message });
    }
    return;
  }

  if (type === "infer") {
    try {
      var flatInputs = new Float32Array(e.data.flatInputs);
      var batchSize = e.data.batchSize;
      var tensor = new ort.Tensor("float32", flatInputs, [batchSize, 3]);
      var results = await session.run({ input: tensor });
      var outKey = Object.keys(results)[0];
      var output = new Float32Array(results[outKey].data);
      self.postMessage({ type: "infer", id: id, data: output.buffer }, [output.buffer]);
    } catch (err) {
      self.postMessage({ type: "infer", id: id, error: err.message });
    }
  }
};
