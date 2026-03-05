/**
 * BirdNET Geomodel – Inference Web Worker
 *
 * Runs ONNX Runtime Web in a dedicated thread so the UI stays responsive.
 *
 * Protocol (postMessage):
 *   Main → Worker:  { type: 'init',  modelUrl }
 *   Worker → Main:  { type: 'init',  ok, error? }
 *   Main → Worker:  { type: 'infer', id, flatInputs: ArrayBuffer, batchSize }
 *   Worker → Main:  { type: 'infer', id, data: ArrayBuffer }
 *                   | { type: 'infer', id, error }
 */

/* global ort */
const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
importScripts(ORT_CDN + "ort.min.js");
ort.env.wasm.wasmPaths = ORT_CDN;

let session = null;

self.onmessage = async function (e) {
  const { type, id } = e.data;

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
      const flatInputs = new Float32Array(e.data.flatInputs);
      const batchSize = e.data.batchSize;
      const tensor = new ort.Tensor("float32", flatInputs, [batchSize, 3]);
      const results = await session.run({ input: tensor });
      const outKey = Object.keys(results)[0];
      // Copy from WASM memory into a transferable buffer
      const output = new Float32Array(results[outKey].data);
      self.postMessage(
        { type: "infer", id, data: output.buffer },
        [output.buffer]
      );
    } catch (err) {
      self.postMessage({ type: "infer", id, error: err.message });
    }
  }
};
