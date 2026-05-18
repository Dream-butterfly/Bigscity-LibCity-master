"""Diagnostic script: verify R² and MAPE correctness for new_fuzzy.

Validates:
1. R² in standardized vs original (inverse-scaled) space
2. Multi-feature R² breakdown (flow/occupancy/speed separately)
3. HA (historical average) baseline R² as sanity lower bound
4. Masked vs unmasked MAPE comparison (with loss.py fix)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from GNNTP.utils.paths import OUTPUT_ROOT
from GNNTP.data.artifact_io import load_data_artifact, deserialize_scaler


def load_model_from_run(run_id: str, data_feature: dict,
                        input_window: int, output_window: int):
    """Load trained NewFuzzy model from a run output directory."""
    from GNNTP.models.new.new_fuzzy.model import NewFuzzy

    run_dir = OUTPUT_ROOT / run_id
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No run_meta.json in {run_dir}")

    with open(meta_path) as f:
        run_meta = json.load(f)

    # Find checkpoint
    cache_dir = run_dir / "model_cache"
    checkpoint_path = None
    for p in sorted(cache_dir.glob("*.m"), reverse=True):
        checkpoint_path = p
        break
    if checkpoint_path is None:
        for p in sorted(cache_dir.glob("*.tar"), reverse=True):
            checkpoint_path = p
            break
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint (.m or .tar) in {cache_dir}")

    print(f"  Loading checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    # Handle two checkpoint formats:
    #   .m  → tuple (model_state_dict, optimizer_state_dict)
    #   .tar → dict with "model_state_dict" key
    if isinstance(checkpoint, tuple):
        # .m format: (model_state_dict, optimizer_state_dict)
        state_dict = checkpoint[0]
    elif isinstance(checkpoint, dict):
        # .tar format: {"model_state_dict": ..., "optimizer_state_dict": ..., ...}
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    else:
        raise TypeError(f"Unknown checkpoint format: {type(checkpoint)}")

    # Try to find key prefix (DDP wraps with "module." prefix)
    sample_key = next(iter(state_dict.keys()))
    if sample_key.startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    # Infer hidden_dim from state_dict
    # input_projection.weight shape: [hidden_dim, input_dim]
    hidden_dim = state_dict["condition_encoder.input_projection.weight"].shape[0]

    # Infer feature_dim and num_nodes from data_feature
    feature_dim = data_feature.get("feature_dim", 1)
    num_nodes = data_feature.get("num_nodes", 1)
    output_dim = data_feature.get("output_dim", 1)
    num_heads = 2
    if hidden_dim >= 128:
        num_heads = 4

    # Infer encoder/decoder layers from state_dict keys
    encoder_layers = sum(1 for k in state_dict if k.startswith("condition_encoder.blocks.") and k.endswith(".norm_temporal.weight"))
    decoder_layers = sum(1 for k in state_dict if k.startswith("future_decoder.blocks.") and k.endswith(".norm_temporal.weight"))
    use_adaptive_graph = any("adaptive_graph_learner" in k for k in state_dict)
    use_fuzzy_graph = any("fuzzy_centers" in k for k in state_dict)
    use_spatiotemporal = any("spatiotemporal_attention" in k for k in state_dict)
    use_temporal_pe = any("temporal_position_embedding" in k for k in state_dict)

    config = {
        "input_window": input_window,
        "output_window": output_window,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "encoder_layers": encoder_layers,
        "decoder_layers": decoder_layers,
        "ffn_hidden_dim": hidden_dim * 2,
        "graph_k_hop": 2,
        "dropout": 0.1,
        "device": torch.device("cpu"),
        "use_adaptive_graph": use_adaptive_graph,
        "adaptive_graph_embed_dim": 32,
        "adaptive_graph_topk": 12,
        "adaptive_graph_blend_init": 0.5,
        "use_fuzzy_graph": use_fuzzy_graph,
        "fuzzy_graph_num_sets": 3,
        "fuzzy_graph_sigma_init": 0.7,
        "use_spatiotemporal_attention": use_spatiotemporal,
        "use_temporal_position_embedding": use_temporal_pe,
        "use_gradient_checkpointing": False,
        "conservation_loss_weight": 0.0,
        "use_fuzzy_conservation": False,
        "conservation_steps_per_epoch": 80,
    }

    model = NewFuzzy(config, data_feature)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"  Model loaded: hidden_dim={hidden_dim}, enc_layers={encoder_layers}, "
          f"dec_layers={decoder_layers}, adaptive={use_adaptive_graph}, fuzzy={use_fuzzy_graph}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    return model


def compute_r2(y_pred, y_true):
    """Compute R² per feature dimension and overall."""
    from sklearn.metrics import r2_score
    results = {}
    n_features = y_true.shape[-1]
    for f in range(n_features):
        p = y_pred[..., f].flatten()
        t = y_true[..., f].flatten()
        results[f"feature_{f}"] = float(r2_score(t, p))
    p_all = y_pred.flatten()
    t_all = y_true.flatten()
    results["overall"] = float(r2_score(t_all, p_all))
    return results


def compute_mape(y_pred, y_true, mask_threshold=1e-3):
    """Compute MAPE with near-zero filtering (matching loss.py fix)."""
    valid = np.abs(y_true) > mask_threshold
    if valid.sum() == 0:
        return 0.0
    loss = np.abs((y_pred - y_true) / (y_true + 1e-5))
    return float((loss * valid).sum() / valid.sum())


def main():
    parser = argparse.ArgumentParser(description="Verify new_fuzzy evaluation metrics")
    parser.add_argument("--run_id", required=True,
                        help="Run ID e.g. 20260518_215722__traffic_state_pred__new_fuzzy__PEMSD4")
    parser.add_argument("--artifact_id", help="Artifact ID (auto-detected from run_meta if omitted)")
    parser.add_argument("--no_model", action="store_true", help="Skip model loading, only data checks")
    args = parser.parse_args()

    run_id = args.run_id
    run_dir = OUTPUT_ROOT / run_id
    if not run_dir.exists():
        print(f"ERROR: run directory not found: {run_dir}")
        sys.exit(1)

    # --- Load run metadata ---
    meta_path = run_dir / "run_meta.json"
    with open(meta_path) as f:
        run_meta = json.load(f)
    artifact_id = args.artifact_id or run_meta.get("artifact_id")
    task = run_meta.get("task", "unknown")
    dataset = run_meta.get("dataset", "unknown")

    print(f"Run:     {run_id}")
    print(f"Task:    {task}")
    print(f"Dataset: {dataset}")
    print(f"Artifact: {artifact_id}")

    # --- Load artifact ---
    bundle = load_data_artifact(artifact_id=artifact_id)
    arrays = np.load(str(bundle["artifact_dir"] / "arrays.npz"))  # re-open for direct key access
    data_feature = bundle["data_feature"]
    meta = bundle["meta"]

    # Supplement data_feature with critical fields from meta
    for key in ("scaler", "input_window", "output_window", "feature_dim", "output_dim"):
        if key in meta:
            data_feature.setdefault(key, meta[key])

    # Add num_nodes from shape
    test_shape = meta.get("test_shape", [0, 0, 0, 0])
    data_feature.setdefault("num_nodes", test_shape[2] if len(test_shape) > 2 else 1)
    # For new_fuzzy, we also need adj_mx
    data_feature.setdefault("adj_mx", np.eye(data_feature.get("num_nodes", 1), dtype=np.float32))

    # Reconstruct scaler
    scaler_payload = meta.get("scaler")
    scaler = deserialize_scaler(scaler_payload) if scaler_payload else None
    scaler_name = scaler_payload.get("type", "none") if isinstance(scaler_payload, dict) else "none"
    print(f"Scaler:  {scaler_name}")

    # --- Extract arrays ---
    x_test = np.asarray(arrays["test_x"])
    y_test = np.asarray(arrays["test_y"])
    x_train = np.asarray(arrays.get("train_x", arrays.get("train_x", None)))
    y_train = np.asarray(arrays.get("train_y", arrays.get("train_y", None)))

    output_window = y_test.shape[1]
    n_features = y_test.shape[-1]
    n_nodes = y_test.shape[2]

    print(f"\nData shapes:")
    if x_train is not None:
        print(f"  train_x: {x_train.shape}, train_y: {y_train.shape}")
    print(f"  test_x:  {x_test.shape}, test_y:  {y_test.shape}")
    print(f"  output_window={output_window}, features={n_features}, nodes={n_nodes}")

    # --- Load model & predict ---
    y_pred = None
    if not args.no_model:
        print("\n--- Loading model ---")
        try:
            model = load_model_from_run(run_id, data_feature,
                                        input_window=x_test.shape[1],
                                        output_window=y_test.shape[1])
        except Exception as e:
            print(f"WARNING: Could not load model: {e}")
            print("Continuing with data-only checks.\n")
            model = None

        if model is not None:
            x_test_t = torch.from_numpy(x_test).float()
            batch_size = 64
            all_preds = []
            with torch.no_grad():
                for i in range(0, len(x_test_t), batch_size):
                    x_batch = x_test_t[i:i + batch_size]
                    batch = {"X": x_batch}
                    pred = model.predict(batch)
                    all_preds.append(pred.cpu().numpy())
            y_pred = np.concatenate(all_preds, axis=0)
            print(f"  Predictions shape: {y_pred.shape}")

    # ================================================================
    # 1. R² in standardized space
    # ================================================================
    print("\n" + "=" * 60)
    print("1. R² ANALYSIS (standardized space, per time step)")
    print("=" * 60)

    if y_pred is not None:
        for step in [0, 2, 5, output_window - 1]:
            if step >= y_pred.shape[1]:
                break
            print(f"\n  Step {step + 1}:")
            r2_values = compute_r2(y_pred[:, step], y_test[:, step])
            for k, v in r2_values.items():
                print(f"    R²({k:>10s}): {v:.6f}")

    # ================================================================
    # 2. HA baseline
    # ================================================================
    print("\n" + "=" * 60)
    print("2. HISTORICAL AVERAGE BASELINE")
    print("=" * 60)

    if y_train is not None:
        # Global mean from training set (naive HA)
        global_mean = y_train.mean(axis=(0, 1, 2), keepdims=True)
        ha_preds = np.tile(global_mean, (y_test.shape[0], output_window, n_nodes, 1))

        for step in [0, 2, 5, output_window - 1]:
            if step >= y_test.shape[1]:
                break
            print(f"\n  Step {step + 1}:")
            ha_r2 = compute_r2(ha_preds[:, step], y_test[:, step])
            for k, v in ha_r2.items():
                print(f"    HA R²({k:>10s}): {v:.6f}")

        # Delta
        if y_pred is not None:
            print(f"\n  ΔR² (model - HA) overall:")
            for step in [0, 2, 5, output_window - 1]:
                if step >= y_test.shape[1]:
                    break
                ha_r2 = compute_r2(ha_preds[:, step], y_test[:, step])["overall"]
                model_r2 = compute_r2(y_pred[:, step], y_test[:, step])["overall"]
                flag = "✓" if model_r2 > ha_r2 + 0.05 else ("⚠" if model_r2 > ha_r2 else "✗")
                print(f"    {flag} Step {step + 1}: model={model_r2:.4f}  HA={ha_r2:.4f}  Δ={model_r2 - ha_r2:+.4f}")

    # ================================================================
    # 3. R² in original (inverse-scaled) space
    # ================================================================
    print("\n" + "=" * 60)
    print("3. R² IN ORIGINAL (INVERSE-SCALED) SPACE")
    print("=" * 60)

    if scaler is not None and y_pred is not None:
        try:
            n_samples, n_steps, n_nodes_local, n_feat = y_pred.shape
            y_pred_flat = y_pred.reshape(-1, n_feat)
            y_test_flat = y_test.reshape(-1, n_feat)
            y_pred_orig = scaler.inverse_transform(y_pred_flat).reshape(y_pred.shape)
            y_test_orig = scaler.inverse_transform(y_test_flat).reshape(y_test.shape)

            feature_names = run_meta.get("data_col", [f"feat_{i}" for i in range(n_feat)])

            for step in [0, 2, 5, output_window - 1]:
                if step >= y_pred.shape[1]:
                    break
                print(f"\n  Step {step + 1}:")
                r2_values = compute_r2(y_pred_orig[:, step], y_test_orig[:, step])
                for k, v in r2_values.items():
                    label = feature_names[int(k.split("_")[1])] if "feature_" in k else k
                    print(f"    R²({label:>15s}): {v:.6f}")

            # MAE in original space
            from sklearn.metrics import mean_absolute_error
            print(f"\n  MAE (original scale):")
            for step in [0, 5, output_window - 1]:
                if step >= y_pred.shape[1]:
                    break
                maes = []
                for f in range(n_feat):
                    mae = mean_absolute_error(
                        y_test_orig[:, step, :, f].flatten(),
                        y_pred_orig[:, step, :, f].flatten()
                    )
                    maes.append(mae)
                labels = [feature_names[f] if f < len(feature_names) else f"f{f}" for f in range(n_feat)]
                print(f"    step {step + 1}: " + ", ".join(f"{l}={m:.3f}" for l, m in zip(labels, maes)))
        except Exception as e:
            print(f"  ERROR during inverse transform: {e}")
            import traceback
            traceback.print_exc()
    else:
        reason = "no scaler" if scaler is None else "no predictions"
        print(f"  Skipped: {reason}")

    # ================================================================
    # 4. MAPE diagnostics
    # ================================================================
    print("\n" + "=" * 60)
    print("4. MAPE DIAGNOSTICS (after loss.py fix)")
    print("=" * 60)

    if y_pred is not None:
        for step in [0, 5, output_window - 1]:
            if step >= y_pred.shape[1]:
                break
            mape_all = compute_mape(y_pred[:, step], y_test[:, step], mask_threshold=1e-3)
            mape_strict = compute_mape(y_pred[:, step], y_test[:, step], mask_threshold=0.1)
            print(f"  Step {step + 1}: filtered MAPE={mape_all:.4f}  strict(|label|>0.1) MAPE={mape_strict:.4f}")

        print(f"\n  Feature breakdown (step 1):")
        for f in range(n_feat):
            mape_f = compute_mape(y_pred[:, 0, :, f], y_test[:, 0, :, f], mask_threshold=1e-3)
            label = feature_names[f] if f < len(feature_names) else f"feat_{f}"
            print(f"    {label}: MAPE={mape_f:.4f}")
    else:
        print("  Skipped: no predictions")

    # ================================================================
    # 5. Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("5. SUMMARY")
    print("=" * 60)

    if y_pred is not None:
        r2_s1 = compute_r2(y_pred[:, 0], y_test[:, 0])["overall"]
        r2_sN = compute_r2(y_pred[:, output_window - 1], y_test[:, output_window - 1])["overall"]
        mape_s1 = compute_mape(y_pred[:, 0], y_test[:, 0], mask_threshold=1e-3)

        print(f"  R² step 1:  {r2_s1:.4f}")
        print(f"  R² step {output_window}: {r2_sN:.4f}")
        print(f"  MAPE step 1 (fixed): {mape_s1:.4f}")

        # Sanity checks
        print(f"\n  Verdict:")
        if y_train is not None:
            ha_r2_s1 = compute_r2(ha_preds[:, 0], y_test[:, 0])["overall"]
            if r2_s1 > ha_r2_s1 * 1.1:
                print(f"    ✓ Model significantly beats HA (ΔR²={r2_s1 - ha_r2_s1:+.4f})")
            elif r2_s1 > ha_r2_s1:
                print(f"    ⚠ Model marginally beats HA (ΔR²={r2_s1 - ha_r2_s1:+.4f})")
            else:
                print(f"    ✗ Model worse than HA — fundamental problem")

        if r2_s1 > 0.99:
            print(f"    ⚠ R² > 0.99 — extremely high, verify no data leakage")
        elif r2_s1 > 0.95:
            print(f"    ⚠ R² > 0.95 — very high for traffic data; verify evaluation logic")
        elif r2_s1 > 0.80:
            print(f"    ✓ R² in reasonable range for good traffic models")

        if mape_s1 < 1.0:
            print(f"    ✓ MAPE now meaningful ({mape_s1:.4f}) — loss.py fix working")
        elif mape_s1 < 50:
            print(f"    ⚠ MAPE {mape_s1:.2f} — somewhat high but reasonable")
        else:
            print(f"    ✗ MAPE still unacceptably high ({mape_s1:.2f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
