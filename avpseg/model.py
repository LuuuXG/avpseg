"""Files and settings for the final five-fold AVP model."""

from __future__ import annotations

import os
from pathlib import Path
import shutil


MODEL_ROOT = Path(__file__).resolve().parent.parent / "models" / "v1"
DATASET = "705"
CONFIGURATION = "3d_fullres"
TRAINER_NAME = "nnUNetTrainer_100epochs_RobustIntensityFast"
CHECKPOINT = "checkpoint_best.pth"
FOLDS = (0, 1, 2, 3, 4)
MODEL_FILES = ("dataset.json", "plans.json", *(f"fold{fold}.pth" for fold in FOLDS))


def prepare_model_results(work_dir: Path) -> Path:
    """Expose the flat model files in nnU-Net's inference layout."""
    missing = [name for name in MODEL_FILES if not (MODEL_ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model files in {MODEL_ROOT}: {', '.join(missing)}")

    results = work_dir / "nnUNet_results"
    trainer_dir = results / "Dataset705_AVP_seg_v2" / (
        f"{TRAINER_NAME}__nnUNetPlans__{CONFIGURATION}"
    )
    files = {"dataset.json": "dataset.json", "plans.json": "plans.json"}
    files.update({f"fold{fold}.pth": f"fold_{fold}/{CHECKPOINT}" for fold in FOLDS})
    for source_name, destination_name in files.items():
        source = (MODEL_ROOT / source_name).resolve()
        destination = trainer_dir / destination_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            try:
                destination.symlink_to(source)
            except OSError:
                shutil.copy2(source, destination)
    return results.resolve()
