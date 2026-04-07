"""Export PaddlePaddle inference models to ONNX format.

Usage (run from everything_paddle/):
    # Export all production + training models:
    uv run python export.py

    # Export a specific model directory:
    uv run python export.py training_data/rank_division/inference

    # Deploy training model outputs to production:
    uv run python export.py --deploy

Exported .onnx files are placed next to the source inference.json.
With --deploy, training models are copied to ../overwatchlooker/models/.
"""
import argparse
import shutil
import sys
from pathlib import Path

import paddle2onnx.paddle2onnx_cpp2py_export as _p2o

_HERE = Path(__file__).parent
_PROD_DIR = _HERE.parent / "overwatchlooker" / "models"

# (name, model_dir, is_production)
_MODELS = [
    ("panel_labels",    _PROD_DIR / "panel_labels",                              True),
    ("panel_values",    _PROD_DIR / "panel_values",                              True),
    ("panel_featured",  _PROD_DIR / "panel_featured",                            True),
    ("team_side",       _PROD_DIR / "team_side",                                 True),
    ("rank_division",   _PROD_DIR / "rank_division",                             True),
    ("rank_division*",  _HERE / "training_data" / "rank_division" / "inference", False),
    ("modifiers",       _HERE / "training_data" / "modifiers" / "inference",     False),
]


def export_one(name: str, model_dir: Path) -> bool:
    """Export a single model to ONNX. Returns True on success."""
    json_file = model_dir / "inference.json"
    params_file = model_dir / "inference.pdiparams"
    out_file = model_dir / "inference.onnx"

    if not json_file.exists() or not params_file.exists():
        print(f"  SKIP: missing inference.json or inference.pdiparams in {model_dir}")
        return False

    if out_file.exists():
        out_file.unlink()

    print(f"\n=== {name} ===")
    print(f"  {model_dir}")

    try:
        onnx_bytes = _p2o.export(
            str(json_file),       # model_file
            str(params_file),     # params_file
            14,                   # opset_version
            True,                 # auto_upgrade_opset
            False,                # verbose
            True,                 # enable_onnx_checker
            True,                 # enable_experimental
            True,                 # enable_optimize
            {},                   # custom_ops
            "",                   # deploy_backend
            "",                   # calibration_file
            "",                   # external_file
            False,                # export_fp16_model
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

    out_file.write_bytes(onnx_bytes)
    raw_mb = out_file.stat().st_size / (1024 * 1024)

    if raw_mb < 1:
        print(f"  FAILED: output is only {raw_mb:.3f} MB (expected ~72 MB)")
        out_file.unlink()
        return False

    # Optimize: constant folding, dead code elimination, fused ops.
    # Removes unused initializers so ONNX Runtime doesn't warn at load time.
    from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
    opts = SessionOptions()
    opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.optimized_model_filepath = str(out_file)
    opts.log_severity_level = 3
    InferenceSession(str(out_file), sess_options=opts,
                     providers=["CPUExecutionProvider"])
    opt_mb = out_file.stat().st_size / (1024 * 1024)

    print(f"  -> {opt_mb:.1f} MB (optimized from {raw_mb:.1f} MB)")
    return True


def deploy() -> None:
    """Copy training model outputs to production locations."""
    pairs = [
        (_HERE / "training_data" / "rank_division" / "inference",
         _PROD_DIR / "rank_division",
         _HERE / "training_data" / "rank_division"),
        (_HERE / "training_data" / "modifiers" / "inference",
         _PROD_DIR / "modifiers",
         _HERE / "training_data" / "modifiers"),
    ]
    for src_dir, dst_dir, dict_dir in pairs:
        onnx = src_dir / "inference.onnx"
        dct = dict_dir / "dict.txt"
        if not onnx.exists():
            print(f"SKIP {src_dir.parent.name}: no inference.onnx (run export first)")
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx, dst_dir / "inference.onnx")
        if dct.exists():
            shutil.copy2(dct, dst_dir / "dict.txt")
        print(f"Deployed {src_dir.parent.name} -> {dst_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PaddlePaddle models to ONNX")
    parser.add_argument("model_dir", nargs="?", help="Specific model directory to export")
    parser.add_argument("--deploy", action="store_true",
                        help="Copy training model outputs to production locations")
    args = parser.parse_args()

    if args.deploy:
        deploy()
        return

    if args.model_dir:
        model_dir = Path(args.model_dir).resolve()
        name = model_dir.parent.name if model_dir.name == "inference" else model_dir.name
        if not export_one(name, model_dir):
            sys.exit(1)
        return

    failed = []
    for name, model_dir, _is_prod in _MODELS:
        if not export_one(name, model_dir):
            failed.append(name)

    if failed:
        print(f"\nFailed: {failed}")
        sys.exit(1)
    print("\nAll models exported.")


if __name__ == "__main__":
    main()
