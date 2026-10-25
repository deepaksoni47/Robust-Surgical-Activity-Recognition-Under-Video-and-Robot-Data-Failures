from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.kinematics_gru import KinematicsGRU


def test_kinematics_gru_forward_shapes() -> None:
    """Test 2D and 3D input tensor handling and forward output shapes for GRU."""
    model = KinematicsGRU(
        input_dim=76,
        proj_dim=64,
        hidden_dim=64,
        num_classes=11,
        num_layers=2,
    )

    # 1. 3D windowed input: (B, W, 76)
    x_3d = torch.randn(4, 16, 76)
    out_3d = model(x_3d)
    assert out_3d.shape == (4, 11)
    assert torch.all(torch.isfinite(out_3d))

    # 2. 2D single-frame input: (B, 76)
    x_2d = torch.randn(4, 76)
    out_2d = model(x_2d)
    assert out_2d.shape == (4, 11)

    # 3. Sequence forward pass: (B, W, C)
    seq_out = model.forward_sequence(x_3d)
    assert seq_out.shape == (4, 16, 11)

    # 4. Feature extraction: (B, hidden_dim)
    feats = model.extract_features(x_3d)
    assert feats.shape == (4, 64)


def test_kinematics_gru_gradient_flow() -> None:
    """Test backward pass through GRU recurrent cells."""
    model = KinematicsGRU(input_dim=76, proj_dim=32, hidden_dim=32, num_classes=5)
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


def test_kinematics_gru_overfit_tiny_batch() -> None:
    """Verify GRU capacity to memorize sequence targets on a tiny batch."""
    torch.manual_seed(42)
    model = KinematicsGRU(
        input_dim=76,
        proj_dim=32,
        hidden_dim=32,
        num_classes=4,
        num_layers=1,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(4, 8, 76)
    target = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    initial_loss = criterion(model(x), target).item()

    for _ in range(60):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(x), target).item()
    assert final_loss < initial_loss
    assert final_loss < 0.2


def test_kinematics_gru_train_and_eval_fold_smoke(tmp_path: Path) -> None:
    """Verify train_and_eval_fold runs 1 epoch and outputs valid test metrics."""
    from scripts.train_kinematics_gru import train_and_eval_fold

    cfg = {
        "model": {"input_dim": 76, "proj_dim": 32, "hidden_dim": 32, "num_layers": 1, "dropout": 0.1, "bidirectional": False},
        "training": {"epochs": 1, "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-4, "use_class_weights": False, "patience": 1},
        "data": {"dataset_dir": "dataset", "window_size": 16, "train_stride": 32, "eval_stride": 32},
    }
    ckpt_path = tmp_path / "gru_model.pt"
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
