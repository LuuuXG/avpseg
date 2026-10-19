from __future__ import annotations

from .model import TRAINER_NAME


def find_packaged_trainer(class_name: str):
    if class_name != TRAINER_NAME:
        return None
    from .nnunet_trainers.nnUNetTrainer_100epochs_RobustIntensityFast import (
        nnUNetTrainer_100epochs_RobustIntensityFast,
    )

    return nnUNetTrainer_100epochs_RobustIntensityFast


def main() -> int:
    from nnunetv2.inference import predict_from_raw_data

    original_finder = predict_from_raw_data.recursive_find_python_class

    def find_trainer(folder: str, class_name: str, current_module: str):
        trainer = original_finder(folder, class_name, current_module)
        if trainer is not None:
            return trainer
        return find_packaged_trainer(class_name)

    predict_from_raw_data.recursive_find_python_class = find_trainer
    predict_from_raw_data.predict_entry_point()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
