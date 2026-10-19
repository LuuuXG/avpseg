from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import avpseg

from .nnunet_predict import find_packaged_trainer
from .model import MODEL_FILES, MODEL_ROOT, TRAINER_NAME


REQUIRED_MODULES = (
    "nibabel", "numpy", "pandas", "scipy", "skimage", "matplotlib",
    "openpyxl", "nnunetv2", "torch", "acvl_utils", "blosc2",
)
def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    print(f"Python: {sys.executable}")
    print(f"avpseg: {Path(avpseg.__file__).resolve()}")

    print("\nModules:")
    modules_ok = True
    for name in REQUIRED_MODULES:
        found = module_available(name)
        modules_ok = modules_ok and found
        print(f"  {'OK  ' if found else 'MISS'} {name}")

    print("\nCustom trainer:")
    try:
        trainer_ok = find_packaged_trainer(TRAINER_NAME) is not None
        detail = TRAINER_NAME
    except (ImportError, OSError) as exc:
        trainer_ok = False
        detail = f"{TRAINER_NAME}: {exc}"
    print(f"  {'OK  ' if trainer_ok else 'MISS'} {detail}")

    root = MODEL_ROOT
    print(f"\nModel files: {root}")
    models_ok = True
    for name in MODEL_FILES:
        found = (root / name).is_file()
        models_ok = models_ok and found
        print(f"  {'OK  ' if found else 'MISS'} {name}")

    if modules_ok and trainer_ok and models_ok:
        print("\nAVP Segmentation is ready.")
        return 0
    if not modules_ok or not trainer_ok:
        print("\nInstall the inference runtime: bash install.sh")
    if not models_ok:
        print(f"\nPlace the missing model files in {root}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
