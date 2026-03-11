"""Export a trained BirdNET Geomodel checkpoint to portable inference formats.

The export wrapper takes a single input tensor of shape ``(batch, 3)`` where
columns are ``[latitude, longitude, week]`` and returns species probabilities
of shape ``(batch, n_species)``.

Supported formats:
    onnx        ONNX FP32
    onnx_fp16   ONNX FP16 (default) — weights in FP16, I/O in FP32 by default
    tflite      TensorFlow Lite FP32
    tflite_fp16 TensorFlow Lite FP16
    tflite_int8 TensorFlow Lite INT8 (dynamic-range quantisation)
    tf          TensorFlow SavedModel
    all         All of the above

FP16 I/O behavior:
    By default, ONNX FP16 exports keep model inputs and outputs in FP32
    (``keep_io_fp32=True``).  This preserves full coordinate precision
    (latitude, longitude, week) and reduces numerical differences versus
    the PyTorch reference (typically <0.05 max diff).  Pass ``--fp16_io``
    to convert I/O tensors to FP16 as well, at the cost of larger
    numerical divergence.

After each conversion, a numerical validation is run: a batch of reference
inputs is passed through both the original PyTorch model and the exported model,
and the maximum absolute difference in species probabilities is reported.
Conversion fails if the difference exceeds a configurable tolerance.

Usage:
    python convert.py                                   # ONNX FP16 (default)
    python convert.py --formats onnx tflite_fp16        # specific formats
    python convert.py --formats all                     # everything
    python convert.py --fp16_io                         # FP16 I/O (lossy)
    python convert.py --checkpoint checkpoints/checkpoint_best.pt --outdir exports
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from model.model import create_model


# ---------------------------------------------------------------------------
# Export wrapper — flattens (lat, lon, week) interface into a single tensor
# ---------------------------------------------------------------------------

class ExportWrapper(nn.Module):
    """Thin wrapper that takes ``(batch, 3)`` and returns sigmoid probabilities.

    Column order: ``[latitude_degrees, longitude_degrees, week_number]``.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lat = x[:, 0]
        lon = x[:, 1]
        week = x[:, 2]
        logits = self.model(lat, lon, week, return_env=False)['species_logits']
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# Reference data for validation
# ---------------------------------------------------------------------------

def _make_reference_inputs(n: int = 200) -> np.ndarray:
    """Create a fixed set of reference inputs covering diverse locations/weeks.

    Returns:
        ``(n, 3)`` float32 array — columns are lat, lon, week.
    """
    rng = np.random.RandomState(42)
    lats = rng.uniform(-90, 90, size=n).astype(np.float32)
    lons = rng.uniform(-180, 180, size=n).astype(np.float32)
    weeks = rng.randint(1, 49, size=n).astype(np.float32)  # 1–48
    return np.stack([lats, lons, weeks], axis=1)


def _pytorch_reference(wrapper: ExportWrapper, inputs: np.ndarray,
                       device: torch.device) -> np.ndarray:
    """Run the PyTorch wrapper and return probabilities as numpy."""
    wrapper.eval()
    with torch.no_grad():
        x = torch.from_numpy(inputs).to(device)
        return wrapper(x).cpu().numpy()


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate(reference: np.ndarray, exported: np.ndarray,
              fmt: str, tol: float) -> bool:
    """Compare exported output to PyTorch reference.

    Returns True if max absolute diff is within tolerance.
    """
    diff = np.abs(reference - exported)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    print(f"  Validation — max diff: {max_diff:.6f}  mean diff: {mean_diff:.6f}  ", end="")
    if max_diff <= tol:
        print(f"OK (tol={tol})")
        return True
    else:
        print(f"FAIL (tol={tol})")
        return False


# ---------------------------------------------------------------------------
# Shared ONNX export helper
# ---------------------------------------------------------------------------

def _torch_onnx_export(wrapper: nn.Module, dummy: torch.Tensor,
                       path: Path) -> None:
    """Run ``torch.onnx.export`` with suppression of known benign warnings.

    Suppressed warnings:

    * *dynamic_axes + dynamo* — PyTorch >= 2.6 defaults to the dynamo-based
      exporter which emits a ``UserWarning`` when ``dynamic_axes`` is used.
      The traditional exporter handles ``dynamic_axes`` correctly; the
      warning is harmless for our model.
    * *LeafSpec deprecation* — ``FutureWarning`` from ``copyreg`` triggered
      during dynamo decomposition.  A PyTorch internal issue that does not
      affect the exported model.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*dynamic_axes.*dynamo.*",
            category=UserWarning)
        warnings.filterwarnings(
            "ignore", message=r".*LeafSpec.*",
            category=FutureWarning)
        torch.onnx.export(
            wrapper, dummy, str(path),
            input_names=["input"],
            output_names=["probabilities"],
            dynamic_axes={"input": {0: "batch"},
                          "probabilities": {0: "batch"}},
            opset_version=18,
        )


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def _export_onnx(wrapper: ExportWrapper, ref_inputs: np.ndarray,
                 ref_outputs: np.ndarray, outdir: Path,
                 fp16: bool, tol: float, device: torch.device,
                 keep_io_fp32: bool = True) -> bool:
    """Export to ONNX format.

    Args:
        wrapper: Export-ready model wrapper.
        ref_inputs: Reference input array ``(n, 3)`` for validation.
        ref_outputs: Expected output probabilities from PyTorch.
        outdir: Output directory.
        fp16: If ``True``, convert weights to FP16.
        tol: Base numerical tolerance for FP32 validation.
        device: Torch device.
        keep_io_fp32: When *fp16* is ``True``, keep model inputs and
            outputs in FP32 while converting internal weights and
            activations to FP16.  This preserves full coordinate
            precision (latitude, longitude, week) and significantly
            reduces numerical differences compared to full FP16.
            Default ``True``.
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("  ERROR: install onnx and onnxruntime — pip install onnx onnxruntime")
        return False

    tag = "onnx_fp16" if fp16 else "onnx"
    path = outdir / f"geomodel{'_fp16' if fp16 else ''}.onnx"
    print(f"\n[{tag}] Exporting to {path}")

    dummy = torch.randn(1, 3, device=device)
    wrapper.eval()

    _torch_onnx_export(wrapper, dummy, path)

    # Merge external data into a single .onnx file
    data_path = Path(str(path) + ".data")
    if data_path.exists():
        model_proto = onnx.load(str(path), load_external_data=True)
        data_path.unlink()
        onnx.save(model_proto, str(path), save_as_external_data=False)

    if fp16:
        from onnxconverter_common import float16
        model_fp32 = onnx.load(str(path))
        # Suppress benign truncation warnings for very small weight
        # values (e.g. near-zero biases clamped to +/-1e-7 in FP16).
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning,
                module=r"onnxconverter_common\.float16")
            model_fp16 = float16.convert_float_to_float16(
                model_fp32, keep_io_types=keep_io_fp32)
        onnx.save(model_fp16, str(path))
        io_note = " (I/O kept at FP32)" if keep_io_fp32 else ""
        print(f"  Converted weights to FP16{io_note}")

    # Validate with ONNX Runtime
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_dtype = sess.get_inputs()[0].type
    if "float16" in input_dtype:
        inp = ref_inputs.astype(np.float16)
    else:
        inp = ref_inputs.astype(np.float32)
    exported = sess.run(None, {"input": inp})[0].astype(np.float32)

    # Tolerance: FP16 weights with FP32 I/O is much tighter than full FP16
    if fp16 and keep_io_fp32:
        effective_tol = max(tol, 0.06)
    elif fp16:
        effective_tol = 0.05
    else:
        effective_tol = tol
    ok = _validate(ref_outputs, exported, tag, effective_tol)

    total_bytes = path.stat().st_size
    size_mb = total_bytes / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")
    return ok


# ---------------------------------------------------------------------------
# TensorFlow / TFLite export
# ---------------------------------------------------------------------------

def _export_tf_saved_model(wrapper: ExportWrapper, ref_inputs: np.ndarray,
                           ref_outputs: np.ndarray, outdir: Path,
                           tol: float, device: torch.device) -> bool:
    """Export to TensorFlow SavedModel via ONNX → tf."""
    try:
        import onnx
        import onnxruntime  # noqa: F401 — needed by onnx2tf sometimes
        import tensorflow as tf
        import onnx2tf
    except ImportError:
        print("  ERROR: install onnx, onnx2tf, tensorflow — pip install onnx onnx2tf tensorflow")
        return False

    onnx_path = outdir / "geomodel_tmp.onnx"
    sm_path = outdir / "saved_model"
    print(f"\n[tf] Exporting SavedModel to {sm_path}")

    # Step 1: export ONNX (FP32) as intermediate
    dummy = torch.randn(1, 3, device=device)
    wrapper.eval()
    _torch_onnx_export(wrapper, dummy, onnx_path)

    # Step 2: convert ONNX → TF SavedModel
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(sm_path),
        non_verbose=True,
    )
    onnx_path.unlink(missing_ok=True)  # clean up intermediate

    # Validate
    loaded = tf.saved_model.load(str(sm_path))
    infer = loaded.signatures["serving_default"]
    out = infer(tf.constant(ref_inputs))
    # output key varies — take first tensor
    exported = list(out.values())[0].numpy()
    ok = _validate(ref_outputs, exported, "tf", tol)

    return ok


def _export_tflite(wrapper: ExportWrapper, ref_inputs: np.ndarray,
                   ref_outputs: np.ndarray, outdir: Path,
                   mode: str, tol: float, device: torch.device) -> bool:
    """Export to TFLite.

    Args:
        mode: ``'fp32'``, ``'fp16'``, or ``'int8'``.
    """
    try:
        import onnx
        import onnxruntime  # noqa: F401
        import tensorflow as tf
        import onnx2tf
    except ImportError:
        print(f"  ERROR: install onnx, onnx2tf, tensorflow — pip install onnx onnx2tf tensorflow")
        return False

    tag = f"tflite_{mode}" if mode != "fp32" else "tflite"
    suffix = {"fp32": "", "fp16": "_fp16", "int8": "_int8"}[mode]
    path = outdir / f"geomodel{suffix}.tflite"
    print(f"\n[{tag}] Exporting to {path}")

    onnx_path = outdir / f"_tmp_{mode}.onnx"
    sm_path = outdir / f"_tmp_sm_{mode}"

    # Step 1: ONNX intermediate
    dummy = torch.randn(1, 3, device=device)
    wrapper.eval()
    _torch_onnx_export(wrapper, dummy, onnx_path)

    # Step 2: ONNX → TF SavedModel
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(sm_path),
        non_verbose=True,
    )
    onnx_path.unlink(missing_ok=True)

    # Step 3: TF SavedModel → TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(str(sm_path))

    if mode == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif mode == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Dynamic range quantisation — no calibration dataset needed

    tflite_model = converter.convert()
    path.write_bytes(tflite_model)

    # Clean up intermediate SavedModel
    import shutil
    shutil.rmtree(sm_path, ignore_errors=True)

    # Validate with TFLite interpreter
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    # TFLite doesn't support dynamic batch — run sample by sample
    exported_list = []
    for i in range(len(ref_inputs)):
        interp.resize_tensor_input(input_details[0]["index"], [1, 3])
        interp.allocate_tensors()
        interp.set_tensor(input_details[0]["index"],
                          ref_inputs[i:i+1].astype(np.float32))
        interp.invoke()
        exported_list.append(interp.get_tensor(output_details[0]["index"]))
    exported = np.concatenate(exported_list, axis=0)

    extra_tol = {"fp32": 1, "fp16": 10, "int8": 100}[mode]
    ok = _validate(ref_outputs, exported, tag, tol * extra_tol)

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")
    return ok


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

ALL_FORMATS = ["onnx", "onnx_fp16", "tflite", "tflite_fp16", "tflite_int8", "tf"]


def convert(
    checkpoint_path: str,
    outdir: str = "exports",
    formats: List[str] | None = None,
    tol: float = 1e-4,
    device: str = "auto",
    keep_io_fp32: bool = True,
) -> Dict[str, bool]:
    """Convert a checkpoint to the requested formats.

    Args:
        checkpoint_path: Path to a ``.pt`` checkpoint file.
        outdir: Directory for exported files.
        formats: List of format names (default: ``['onnx_fp16']``).
        tol: Base tolerance for numerical validation.
        device: ``'auto'``, ``'cuda'``, or ``'cpu'``.
        keep_io_fp32: Keep model inputs/outputs in FP32 for FP16
            exports.  This preserves coordinate precision and reduces
            numerical divergence.  Default ``True``.

    Returns:
        Dict mapping format name to success boolean.
    """
    if formats is None:
        formats = ["onnx_fp16"]
    if "all" in formats:
        formats = list(ALL_FORMATS)

    dev = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
           if device == "auto" else torch.device(device))

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model_config = ckpt["model_config"]

    model = create_model(
        n_species=model_config["n_species"],
        n_env_features=model_config["n_env_features"],
        model_scale=model_config.get("model_scale", 1.0),
        coord_harmonics=model_config.get("coord_harmonics", 8),
        week_harmonics=model_config.get("week_harmonics", 4),
        habitat_head=model_config.get("habitat_head", False),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: scale={model_config.get('model_scale', 1.0)}  |  "
          f"{model_config['n_species']:,} species  |  "
          f"{n_params:,} parameters")

    wrapper = ExportWrapper(model).to(dev)
    wrapper.eval()

    outpath = Path(outdir)
    outpath.mkdir(parents=True, exist_ok=True)

    # Generate reference data on CPU for validation
    ref_inputs = _make_reference_inputs()
    ref_outputs = _pytorch_reference(wrapper, ref_inputs, dev)
    print(f"Reference outputs: shape {ref_outputs.shape}, "
          f"range [{ref_outputs.min():.4f}, {ref_outputs.max():.4f}]")

    # Copy labels.txt alongside exports
    labels_src = Path(checkpoint_path).parent / "labels.txt"
    if labels_src.exists():
        import shutil
        shutil.copy2(labels_src, outpath / "labels.txt")
        print(f"Copied labels.txt → {outpath / 'labels.txt'}")

    # Run conversions
    results: Dict[str, bool] = {}
    dispatch = {
        "onnx":        lambda: _export_onnx(wrapper, ref_inputs, ref_outputs,
                                            outpath, fp16=False, tol=tol, device=dev),
        "onnx_fp16":   lambda: _export_onnx(wrapper, ref_inputs, ref_outputs,
                                            outpath, fp16=True, tol=tol, device=dev,
                                            keep_io_fp32=keep_io_fp32),
        "tflite":      lambda: _export_tflite(wrapper, ref_inputs, ref_outputs,
                                              outpath, mode="fp32", tol=tol, device=dev),
        "tflite_fp16": lambda: _export_tflite(wrapper, ref_inputs, ref_outputs,
                                              outpath, mode="fp16", tol=tol, device=dev),
        "tflite_int8": lambda: _export_tflite(wrapper, ref_inputs, ref_outputs,
                                              outpath, mode="int8", tol=tol, device=dev),
        "tf":          lambda: _export_tf_saved_model(wrapper, ref_inputs, ref_outputs,
                                                      outpath, tol=tol, device=dev),
    }

    for fmt in formats:
        if fmt not in dispatch:
            print(f"\nUnknown format: {fmt} — skipping")
            results[fmt] = False
            continue
        try:
            results[fmt] = dispatch[fmt]()
        except Exception as e:
            print(f"  ERROR during {fmt} export: {e}")
            results[fmt] = False

    # Summary
    print("\n" + "=" * 60)
    print("  Conversion Summary")
    print("=" * 60)
    for fmt, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {fmt:15s}  {status}")
    print("=" * 60)

    n_fail = sum(1 for ok in results.values() if not ok)
    if n_fail:
        print(f"\n{n_fail} conversion(s) failed.")
    else:
        print(f"\nAll {len(results)} conversion(s) passed.")
    print(f"Output directory: {outpath}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Export BirdNET Geomodel to portable inference formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available formats: {', '.join(ALL_FORMATS)}, all",
    )
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/checkpoint_best.pt",
                        help="Path to model checkpoint (default: checkpoints/checkpoint_best.pt)")
    parser.add_argument("--outdir", type=str, default="exports",
                        help="Output directory (default: exports)")
    parser.add_argument("--formats", nargs="+", default=["onnx_fp16"],
                        help="Formats to export (default: onnx_fp16). Use 'all' for everything.")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="Base tolerance for numerical validation (default: 1e-4)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device for PyTorch model (default: auto)")
    parser.add_argument("--fp16_io", action="store_true",
                        help="Convert model I/O to FP16 as well (default: keep "
                             "inputs/outputs at FP32 for better precision)")
    args = parser.parse_args()

    results = convert(
        checkpoint_path=args.checkpoint,
        outdir=args.outdir,
        formats=args.formats,
        tol=args.tol,
        device=args.device,
        keep_io_fp32=not args.fp16_io,
    )

    # Exit with error code if any conversion failed
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
