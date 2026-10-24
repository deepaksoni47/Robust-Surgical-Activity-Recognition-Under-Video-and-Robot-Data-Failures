import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.reproducibility import save_json


TASKS = ["Suturing", "Knot_Tying", "Needle_Passing"]


def parse_meta_file(meta_path: Path, task_name: str) -> List[Dict[str, Any]]:
    """Parse a JIGSAWS task meta_file_*.txt.
    
    Format per line:
    filename \t\t skill_level \t total_score \t\t s1 \t s2 \t s3 \t s4 \t s5 \t s6
    """
    records = []
    if not meta_path.is_file():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = re.split(r"\s+", line)
            if len(parts) < 3:
                continue
            
            trial_id = parts[0]
            skill_level = parts[1]
            try:
                total_grs = int(parts[2])
            except ValueError:
                total_grs = None
                
            grs_elements = []
            for item in parts[3:]:
                try:
                    grs_elements.append(int(item))
                except ValueError:
                    pass

            # Extract user_id and trial_num from trial_id (e.g., Suturing_B001 -> user 'B', trial 1)
            match = re.search(r"([A-Za-z]+)_([A-I])(\d+)", trial_id)
            if match:
                user_id = match.group(2)
                trial_num = int(match.group(3))
            else:
                user_id = None
                trial_num = None

            records.append({
                "trial_id": trial_id,
                "task": task_name,
                "user_id": user_id,
                "trial_num": trial_num,
                "skill_level": skill_level,
                "grs_total": total_grs,
                "grs_elements": grs_elements,
            })

    return records


def build_manifest(dataset_root: str | Path = "dataset", tasks: Optional[List[str]] = None) -> pd.DataFrame:
    """Build unified multi-task metadata manifest for JIGSAWS dataset.
    
    Args:
        dataset_root: Root path to JIGSAWS dataset directory.
        tasks: List of task folder names to index. Defaults to all 3 canonical tasks.
        
    Returns:
        DataFrame containing 103 validated trial records with modality file paths.
    """
    root = Path(dataset_root)
    if tasks is None:
        tasks = TASKS

    all_records = []

    for task in tasks:
        task_dir = root / task
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")

        meta_file = task_dir / f"meta_file_{task}.txt"
        meta_records = parse_meta_file(meta_file, task)

        kinematics_dir = task_dir / "kinematics" / "AllGestures"
        transcriptions_dir = task_dir / "transcriptions"
        video_dir = task_dir / "video"

        for rec in meta_records:
            trial_id = rec["trial_id"]

            kin_path = kinematics_dir / f"{trial_id}.txt"
            trans_path = transcriptions_dir / f"{trial_id}.txt"
            vid_cap1 = video_dir / f"{trial_id}_capture1.avi"
            vid_cap2 = video_dir / f"{trial_id}_capture2.avi"

            rec["kinematics_path"] = str(kin_path).replace("\\", "/")
            rec["transcription_path"] = str(trans_path).replace("\\", "/")
            rec["video_capture1_path"] = str(vid_cap1).replace("\\", "/")
            rec["video_capture2_path"] = str(vid_cap2).replace("\\", "/")

            rec["kinematics_exists"] = kin_path.is_file()
            rec["transcription_exists"] = trans_path.is_file()
            rec["video_capture1_exists"] = vid_cap1.is_file()
            rec["video_capture2_exists"] = vid_cap2.is_file()

            all_records.append(rec)

    df = pd.DataFrame(all_records)
    return df


def export_manifest(df: pd.DataFrame, output_dir: str | Path = "artifacts") -> Path:
    """Export manifest to CSV and JSON formats."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "manifest.csv"
    json_path = out_dir / "manifest.json"

    df.to_csv(csv_path, index=False)
    save_json(df.to_dict(orient="records"), json_path)

    return csv_path
