"""Compute trial-level Segmental Edit Scores for AdaptiveReliabilityFusion across all folds."""

import json
from pathlib import Path
import numpy as np
import torch

from src.data.multimodal_dataset import get_multimodal_fold_datasets
from src.data.splits import get_louo_splits
from src.data.transcriptions import get_gesture_vocabulary
from src.evaluation.metrics import compute_edit_score
from src.models.kinematics_gru import KinematicsGRU
from src.models.reliability_gate import AdaptiveReliabilityFusion
from src.models.video_head import VideoGRU
from src.utils.reproducibility import get_device

TASKS = {
    "Suturing": {"folds": list(range(1, 9)), "num_classes": 11},
    "Knot_Tying": {"folds": list(range(1, 9)), "num_classes": 7},
    "Needle_Passing": {"folds": [1, 2, 3, 4, 5, 7, 8], "num_classes": 11},
}

def compute_adaptive_edit_scores():
    device = get_device()
    results = {}

    for task, info in TASKS.items():
        folds = info["folds"]
        num_classes = info["num_classes"]
        fold_edit_scores = []
        print(f"\nComputing trial Edit Scores for Adaptive Fusion on {task}...")

        for fold in folds:
            ckpt_path = Path(f"checkpoints/adaptive_fusion_{task}_fold{fold}.pt")
            if not ckpt_path.is_file():
                print(f"Missing ckpt: {ckpt_path}")
                continue

            v_model = VideoGRU(input_dim=512, proj_dim=128, hidden_dim=128, num_classes=num_classes, num_layers=2)
            k_model = KinematicsGRU(input_dim=76, proj_dim=128, hidden_dim=128, num_classes=num_classes, num_layers=2)
            model = AdaptiveReliabilityFusion(
                num_classes=num_classes,
                video_model=v_model,
                kinematics_model=k_model,
                video_dim=num_classes,
                kinematics_dim=num_classes,
            )
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.to(device)
            model.eval()

            ds = get_multimodal_fold_datasets(
                task=task,
                fold_num=fold,
                window_size=16,
                train_stride=8,
                eval_stride=8,
            )
            test_ds = ds["test"]

            true_seqs, pred_seqs = [], []
            with torch.no_grad():
                for rec in test_ds.trial_records:
                    v_arr = torch.from_numpy(rec["video"]).float().to(device).unsqueeze(0)
                    k_arr = torch.from_numpy(rec["kinematics"]).float().to(device).unsqueeze(0)
                    v_seq = model.video_model.forward_sequence(v_arr).squeeze(0)
                    k_seq = model.kinematics_model.forward_sequence(k_arr).squeeze(0)
                    fused_seq, _ = model.forward_logits(v_seq, k_seq)
                    preds = torch.argmax(fused_seq, dim=-1).cpu().numpy().tolist()
                    true_seqs.append(rec["labels"].tolist())
                    pred_seqs.append(preds)

            edit = compute_edit_score(true_seqs, pred_seqs)
            fold_edit_scores.append(edit)
            print(f"  Fold {fold}: Edit Score = {edit:.2f}%")

        mean_edit = float(np.mean(fold_edit_scores))
        std_edit = float(np.std(fold_edit_scores))
        results[task] = {
            "mean": mean_edit,
            "std": std_edit,
            "min": float(np.min(fold_edit_scores)),
            "max": float(np.max(fold_edit_scores)),
            "folds": fold_edit_scores,
        }
        print(f"--> {task} Adaptive Fusion Edit Score: {mean_edit:.2f} ± {std_edit:.2f}% (range [{min(fold_edit_scores):.2f}, {max(fold_edit_scores):.2f}])")

    with open("artifacts/analysis/adaptive_fusion_edit_scores.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to artifacts/analysis/adaptive_fusion_edit_scores.json")

if __name__ == "__main__":
    compute_adaptive_edit_scores()
