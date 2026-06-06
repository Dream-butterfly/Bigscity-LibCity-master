#!/usr/bin/env python
"""Quick smoke test for final_T2 model registration."""

import sys
import numpy as np
import torch

print("=" * 60)
print("Testing final_T2 model registration and imports")
print("=" * 60)

# Test 1: Locator can find the model
print("\n[1] Testing locator...")
try:
    from GNNTP.models.locator import get_model_component, get_model_metadata

    metadata = get_model_metadata('traffic_state_pred', 'final_T2')
    print(f"✓ Metadata loaded:")
    print(f"  - model: {metadata.get('model')}")
    print(f"  - task: {metadata.get('task')}")
    print(f"  - model_entry: {metadata.get('model_entry')}")
    print(f"  - package: {metadata.get('package')}")
except Exception as e:
    print(f"✗ Metadata load failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Model class can be imported
print("\n[2] Testing model import...")
try:
    ModelClass = get_model_component('traffic_state_pred', 'final_T2', 'model')
    print(f"✓ Model class loaded: {ModelClass.__name__}")
    print(f"  - Full path: {ModelClass.__module__}")
except Exception as e:
    print(f"✗ Model load failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Model can be instantiated
print("\n[3] Testing model instantiation...")
try:
    config = {
        "input_window": 12,
        "output_window": 12,
        "hidden_dim": 32,
        "num_heads": 2,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "ffn_hidden_dim": 64,
        "graph_k_hop": 2,
        "use_fuzzy_graph": True,
        "fuzzy_num_sets": 3,
        "use_cell_attention": True,
        "num_cells": 4,
        "use_hollow_kernel": True,
        "cell_blend_init": 0.3,
        "dropout": 0.1,
        "graph_closure_steps": 0,
        "device": torch.device("cpu"),
    }

    data_feature = {
        "num_nodes": 5,
        "feature_dim": 3,
        "output_dim": 3,
        "adj_mx": np.eye(5, dtype=np.float32),
        "scaler": None,
    }

    model = ModelClass(config, data_feature)
    print(f"✓ Model instantiated successfully")
    print(f"  - # parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"✗ Model instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Forward pass
print("\n[4] Testing forward pass...")
try:
    model.eval()
    batch = {
        "X": torch.randn(2, config["input_window"], data_feature["num_nodes"], data_feature["feature_dim"]),
        "y": torch.randn(2, config["output_window"], data_feature["num_nodes"], data_feature["output_dim"]),
    }

    with torch.no_grad():
        output = model.predict(batch)

    print(f"✓ Forward pass successful")
    print(f"  - Input shape: {batch['X'].shape}")
    print(f"  - Output shape: {output.shape}")
    assert output.shape == (2, config["output_window"], data_feature["num_nodes"], data_feature["output_dim"])
    print(f"  - Shape check passed ✓")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Type-2 info retrieval
print("\n[5] Testing Type-2 info retrieval...")
try:
    if model.fuzzy_graph is not None:
        history = batch["X"]
        R, fou = model.fuzzy_graph.get_type2_info(history)
        print(f"✓ Type-2 info retrieved:")
        print(f"  - Graph R shape: {R.shape}")
        print(f"  - FOU shape: {fou.shape}")
        print(f"  - FOU range: [{fou.min():.4f}, {fou.max():.4f}]")
    else:
        print("⊘ Fuzzy graph not enabled, skipping")
except Exception as e:
    print(f"✗ Type-2 info retrieval failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)

