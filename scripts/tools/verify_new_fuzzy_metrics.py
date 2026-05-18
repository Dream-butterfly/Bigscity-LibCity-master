"""Diagnostic script: verify R² and MAPE correctness for new_fuzzy.

Validates:
1. R² in standardized vs original (inverse-scaled) space
2. Multi-feature R² breakdown (flow/occupancy/speed separately)
3. HA (historical average) baseline R² as sanity lower bound
4. Masked vs unmasked MAPE comparison
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

from GNNTP.utils.paths import OUTPUT_ROOT, CACHE_ROOT


def load_artifact(artifact_id: str):
    """Load data artifact arrays and data_feature."""
    artifact_dir = CACHE_ROOT / "data_artifacts" / artifact_id
    arrays = np.load(artifact_dir / "arrays.npz")
    import pickle
    with open(artifact_dir / "data_feature.pkl", "rb") as f:
        data_feature = pickle.load(f)
    return arrays, data_feature


def load_model_checkpoint(run_id: str):
    """Load trained model checkpoint."""
    run_dir = OUTPUT_ROOT / run_id
    model_path = None
    cache_dir = run_dir / "model_cache"
    if cache_dir.exists():
        for p in sorted(cache_dir.glob("*.m")):
            model_path = p
            break
        if model_path is None:
            for p in sorted(cache_dir.glob("*.tar"), reverse=True):
                model_path = p
                break
    if model_path is None:
        raise FileNotFoundError(f"No model checkpoint found in {cache_dir}")
    return torch.load(model_path, map_location="cpu", weights_only=False)


def historical_average_baseline(y_true_train, y_true_test, output_window):
    """Simple HA: predict mean of last input_window steps."""
    # HA for traffic: repeat average of historical window
    # This is approximate; proper HA uses day-of-week/time-of-day averages
    ha_preds = np.tile(y_true_train.mean(axis=0, keepdims=True), (y_true_test.shape[0], 1, 1, 1))
    ha_preds = np.tile(ha_preds, (1, output_window, 1, 1))
    # Actually, let's use a simplified HA: repeat the last observed value
    # Since we don't have the input data here, use test set mean as naive baseline
    global_mean = y_true_train.mean(axis=(0, 1, 2), keepdims=True)
    ha_preds = np.tile(global_mean, (y_true_test.shape[0], output_window, y_true_test.shape[2], 1))
    return ha_preds


def compute_r2(y_pred, y_true):
    """Compute R² per feature dimension."""
    from sklearn.metrics import r2_score
    results = {}
    n_features = y_true.shape[-1]
    for f in range(n_features):
        p = y_pred[..., f].flatten()
        t = y_true[..., f].flatten()
        results[f"feature_{f}"] = r2_score(t, p)
    # Overall
    p_all = y_pred.flatten()
    t_all = y_true.flatten()
    results["overall"] = r2_score(t_all, p_all)
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
    parser.add_argument("--run_id", required=True, help="Run ID e.g. 20260518_215722__traffic_state_pred__new_fuzzy__PEMSD4")
    parser.add_argument("--artifact_id", help="Artifact ID (auto-detected from run_meta if omitted)")
    args = parser.parse_args()

    run_id = args.run_id
    run_dir = OUTPUT_ROOT / run_id
    if not run_dir.exists():
        print(f"ERROR: run directory not found: {run_dir}")
        sys.exit(1)

    # Load run metadata
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        artifact_id = args.artifact_id or meta.get("artifact_id")
        print(f"Run: {run_id}")
        print(f"Artifact: {artifact_id}")
        print(f"Dataset: {meta.get('dataset', 'unknown')}")
    else:
        print("WARNING: no run_meta.json, need --artifact_id")
        if not args.artifact_id:
            sys.exit(1)
        artifact_id = args.artifact_id

    # Load artifact
    arrays, data_feature = load_artifact(artifact_id)
    scaler = data_feature.get("scaler")
    print(f"Scaler: {type(scaler).__name__ if scaler else 'None'}")

    # Get train/valid/test splits
    # (These are loaded directly from the arrays file)
    y_train = arrays.get("y_train")  # shape [N_train, T_out, N, C]
    y_val = arrays.get("y_val")
    y_test = arrays.get("y_test")

    if y_test is None:
        # Try to figure out the split from config
        print("ERROR: Could not load y_test from artifact")
        sys.exit(1)

    print(f"\nData shapes:")
    print(f"  y_train: {y_train.shape if y_train is not None else 'N/A'}")
    print(f"  y_val:   {y_val.shape if y_val is not None else 'N/A'}")
    print(f"  y_test:  {y_test.shape}")

    output_window = y_test.shape[1]

    # Load model and get predictions
    print("\n--- Loading model ---")
    try:
        checkpoint = load_model_checkpoint(run_id)
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("Cannot load model for prediction verification.")
        print("Will only perform data-level sanity checks.\n")
        checkpoint = None

    y_pred = None
    if checkpoint is not None:
        # Get model class and instantiate
        from GNNTP.models.new.new_fuzzy.model import NewFuzzy
        # Build config from run_meta + defaults
        config = {
            "input_window": 12,
            "output_window": output_window,
            "hidden_dim": 64,
            "num_heads": 2,
            "encoder_layers": 2,
            "decoder_layers": 2,
            "ffn_hidden_dim": 128,
            "graph_k_hop": 2,
            "dropout": 0.1,
            "device": torch.device("cpu"),
            "use_adaptive_graph": True,
            "adaptive_graph_embed_dim": 32,
            "use_fuzzy_graph": True,
            "fuzzy_graph_num_sets": 3,
            "fuzzy_graph_sigma_init": 0.7,
            "use_spatiotemporal_attention": True,
            "use_temporal_position_embedding": True,
            "use_gradient_checkpointing": False,
            "conservation_loss_weight": 0.0,
            "use_fuzzy_conservation": False,
        }
        model = NewFuzzy(config, data_feature)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.eval()

        # Run prediction on test set (in batches to avoid OOM)
        x_test = arrays.get("x_test")  # [N, T_in, N, C_in]
        if x_test is None:
            print("ERROR: x_test not in artifact arrays")
        else:
            batch_size = 64
            all_preds = []
            with torch.no_grad():
                for i in range(0, len(x_test), batch_size):
                    x_batch = torch.from_numpy(x_test[i:i + batch_size]).float()
                    batch = {"X": x_batch}
                    pred = model.predict(batch)
                    all_preds.append(pred.cpu().numpy())
            y_pred = np.concatenate(all_preds, axis=0)
            print(f"  Predictions shape: {y_pred.shape}")

    # ================================================================
    # 1. R² in standardized vs original space
    # ================================================================
    print("\n" + "=" * 60)
    print("1. R² ANALYSIS (standardized space)")
    print("=" * 60)

    if y_pred is not None:
        for step in [0, 2, 5, 11]:
            if step >= y_pred.shape[1]:
                break
            print(f"\n  Step {step + 1}:")
            r2_values = compute_r2(y_pred[:, step], y_test[:, step])
            for k, v in r2_values.items():
                print(f"    R²({k:>10s}): {v:.6f}")

    # ================================================================
    # 2. HA baseline R²
    # ================================================================
    print("\n" + "=" * 60)
    print("2. HISTORICAL AVERAGE BASELINE")
    print("=" * 60)

    if y_train is not None:
        # Use training set global mean as naive baseline
        global_mean = y_train.mean(axis=(0, 1, 2), keepdims=True)
        ha_preds = np.tile(global_mean, (y_test.shape[0], output_window, y_test.shape[2], 1))
        for step in [0, 2, 5, 11]:
            if step >= y_test.shape[1]:
                break
            print(f"\n  Step {step + 1}:")
            r2_values = compute_r2(ha_preds[:, step], y_test[:, step])
            for k, v in r2_values.items():
                print(f"    HA R²({k:>10s}): {v:.6f}")

        # Improvement over HA
        if y_pred is not None:
            print(f"\n  R² improvement over HA (overall):")
            for step in [0, 2, 5, 11]:
                if step >= y_test.shape[1]:
                    break
                ha_r2 = compute_r2(ha_preds[:, step], y_test[:, step])["overall"]
                model_r2 = compute_r2(y_pred[:, step], y_test[:, step])["overall"]
                print(f"    Step {step + 1}: model={model_r2:.4f}  HA={ha_r2:.4f}  Δ={model_r2 - ha_r2:+.4f}")
    else:
        print("  Skipped: no y_train in artifact")

    # ================================================================
    # 3. R² in original (inverse-scaled) space
    # ================================================================
    print("\n" + "=" * 60)
    print("3. R² IN ORIGINAL (INVERSE-SCALED) SPACE")
    print("=" * 60)

    if scaler is not None and y_pred is not None:
        try:
            y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, y_pred.shape[-1])).reshape(y_pred.shape)
            y_test_orig = scaler.inverse_transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
            for step in [0, 2, 5, 11]:
                if step >= y_pred.shape[1]:
                    break
                print(f"\n  Step {step + 1}:")
                r2_values = compute_r2(y_pred_orig[:, step], y_test_orig[:, step])
                for k, v in r2_values.items():
                    print(f"    R²({k:>10s}): {v:.6f}")

            # Also compute MAE in original space
            from sklearn.metrics import mean_absolute_error
            for step in [0, 5, 11]:
                if step >= y_pred.shape[1]:
                    break
                mae_per_feat = []
                for f in range(y_pred.shape[-1]):
                    mae = mean_absolute_error(
                        y_test_orig[:, step, :, f].flatten(),
                        y_pred_orig[:, step, :, f].flatten()
                    )
                    mae_per_feat.append(mae)
                print(f"\n  MAE (original scale) @ step {step + 1}: {[f'{m:.3f}' for m in mae_per_feat]}")
        except Exception as e:
            print(f"  ERROR during inverse transform: {e}")
    else:
        print("  Skipped: no scaler or no predictions")

    # ================================================================
    # 4. MAPE comparison: masked vs unmasked
    # ================================================================
    print("\n" + "=" * 60)
    print("4. MAPE DIAGNOSTICS")
    print("=" * 60)

    if y_pred is not None:
        for step in [0, 5, 11]:
            if step >= y_pred.shape[1]:
                break
            # Unmasked (new behavior with filter)
            mape_unmasked = compute_mape(y_pred[:, step], y_test[:, step])
            # Masked (|label| > 0.1)
            mape_masked = compute_mape(y_pred[:, step], y_test[:, step], mask_threshold=0.1)
            print(f"  Step {step + 1}: unmasked MAPE={mape_unmasked:.4f}  masked(|label|>0.1) MAPE={mape_masked:.4f}")

        # Feature breakdown
        print(f"\n  Feature breakdown (step 1, unmasked):")
        for f in range(y_pred.shape[-1]):
            mape_f = compute_mape(y_pred[:, 0, :, f], y_test[:, 0, :, f])
            print(f"    feature_{f}: MAPE={mape_f:.4f}")
    else:
        print("  Skipped: no predictions")

    # ================================================================
    # 5. Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("5. DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"  Run ID: {run_id}")
    print(f"  Artifact: {artifact_id}")
    print(f"  Dataset: {meta.get('dataset', 'unknown') if meta_path.exists() else 'unknown'}")
    print(f"  Output dim: {y_test.shape[-1]}, Nodes: {y_test.shape[2]}, Steps: {y_test.shape[1]}")
    print(f"\n  Key checks:")
    if y_pred is not None:
        r2_step1 = compute_r2(y_pred[:, 0], y_test[:, 0])["overall"]
        r2_step12 = compute_r2(y_pred[:, 11], y_test[:, 11])["overall"]
        print(f"    Model R²: step1={r2_step1:.4f}, step12={r2_step12:.4f}")
        if y_train is not None:
            ha_r2_step1 = compute_r2(ha_preds[:, 0], y_test[:, 0])["overall"]
            print(f"    HA R²:    step1={ha_r2_step1:.4f}")
            if r2_step1 > ha_r2_step1 * 1.05:
                print(f"    ✓ Model beats HA by {r2_step1 - ha_r2_step1:+.4f}")
            else:
                print(f"    ⚠ Model barely beats HA (Δ={r2_step1 - ha_r2_step1:+.4f})")
        if r2_step1 > 0.99:
            print(f"    ⚠ R² > 0.99 — suspiciously high, verify data leakage?")
        elif r2_step1 > 0.95:
            print(f"    ⚠ R² > 0.95 — unusually high for traffic prediction")
        elif r2_step1 > 0.85:
            print(f"    ✓ R² in expected range (0.85-0.95) for good traffic models")
        else:
            print(f"    ⚠ R² < 0.85 — model may be underperforming")

    mape_step1 = compute_mape(y_pred[:, 0], y_test[:, 0])
    if mape_step1 < 1.0:
        print(f"    ✓ MAPE after fix = {mape_step1:.4f} (previously was 6000+)")
    else:
        print(f"    ⚠ MAPE still high: {mape_step1:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
