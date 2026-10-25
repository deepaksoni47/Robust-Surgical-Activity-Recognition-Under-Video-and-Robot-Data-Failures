import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.kinematics_mlp import KinematicsMLP


def test_kinematics_mlp_forward_shapes() -> None:
    """Test 2D and 3D input tensor handling and forward output shapes."""
    model = KinematicsMLP(input_dim=76, num_classes=11, hidden_dims=[128, 64])

    # 1. 2D single-frame input: (B, 76)
    x_2d = torch.randn(8, 76)
    out_2d = model(x_2d)
    assert out_2d.shape == (8, 11)
    assert torch.all(torch.isfinite(out_2d))

    # 2. 3D windowed input: (B, W, 76)
    x_3d = torch.randn(8, 16, 76)
    out_3d = model(x_3d)
    assert out_3d.shape == (8, 11)
    assert torch.all(torch.isfinite(out_3d))

    # 3. Sequence forward pass: (B, W, C)
    seq_out = model.forward_sequence(x_3d)
    assert seq_out.shape == (8, 16, 11)

    # 4. Feature extraction: (B, 64)
    feats = model.extract_features(x_3d)
    assert feats.shape == (8, 64)


def test_kinematics_mlp_gradient_flow() -> None:
    """Test backward pass and gradient generation across all trainable parameters."""
    model = KinematicsMLP(input_dim=76, num_classes=11)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(4, 16, 76)
    target = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    logits = model(x)
    loss = criterion(logits, target)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"
        assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradient in {name}"

    optimizer.step()


def test_kinematics_mlp_overfit_tiny_batch() -> None:
    """Verify that model capacity is sufficient to memorize a tiny batch (sanity check)."""
    torch.manual_seed(42)
    model = KinematicsMLP(input_dim=76, num_classes=4, hidden_dims=[64, 32], dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(4, 76)
    target = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    initial_loss = criterion(model(x), target).item()

    for _ in range(50):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(x), target).item()
    assert final_loss < initial_loss
    assert final_loss < 0.1

    preds = torch.argmax(model(x), dim=-1)
    assert torch.equal(preds, target)


def test_kinematics_mlp_invalid_dimensions() -> None:
    """Verify that unsupported input dimensions raise ValueError."""
    model = KinematicsMLP(input_dim=76, num_classes=5)
    with pytest.raises(ValueError, match="Expected 2D or 3D tensor"):
        model(torch.randn(4, 16, 76, 2))  # 4D tensor


def test_train_and_eval_fold_smoke(tmp_path) -> None:
    """Verify train_and_eval_fold runs 1 epoch and outputs valid test metrics."""
    from scripts.train_kinematics_mlp import train_and_eval_fold

    cfg = {
        "model": {"input_dim": 76, "hidden_dims": [32, 16], "dropout": 0.1, "activation": "gelu", "use_layer_norm": True},
        "training": {"epochs": 1, "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-4, "use_class_weights": False, "patience": 1},
        "data": {"dataset_dir": "dataset", "window_size": 16, "train_stride": 32, "eval_stride": 32},
    }
    ckpt_path = tmp_path / "model.pt"
    metrics = train_and_eval_fold(
        task="Suturing",
        fold_num=1,
        cfg=cfg,
        device=torch.device("cpu"),
        save_checkpoint_path=ckpt_path,
    )
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "balanced_accuracy" in metrics
    assert "edit_score" in metrics
    assert ckpt_path.is_file()
