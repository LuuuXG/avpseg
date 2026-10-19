#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m pip install -e "$project_dir"
# Install the NIfTI runtime from requirements.txt; skip nnU-Net's DICOM extras.
python -m pip install --no-deps nnunetv2==2.5.2
printf '\nInstallation complete. Run avp_seg_doctor to check the environment and model files.\n'
