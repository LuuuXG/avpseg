#!/usr/bin/env python
"""Run the final AVP segmentation and optic nerve analysis pipeline for one case.

Inputs:
    1. T1w NIfTI
    2. SynthSeg NIfTI in the same anatomical space
    3. Output directory (default: avp_seg_output)

Outputs:
    - avp_mask.nii.gz
    - avp_prob.nii.gz
    - avp_5label.nii.gz
    - optic_nerve_metrics.xlsx
    - qc.html
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import ListedColormap, to_rgba
import nibabel as nib
from nibabel.processing import resample_from_to, resample_to_output
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
from .morphometry import (
    bridge_mask_components,
    centerline_length_mm,
    centerline_positions_mm,
    crop_to_mask,
    surface_area_mm2,
    skeletonize_3d,
)
from .voi import ANCHOR_LABELS, crop_bounds
from .restore import match_probability_to_prediction
from .subregions import LABELS as FIVE_LABEL_NAMES, split_mask
from .model import (
    CHECKPOINT, CONFIGURATION, DATASET, FOLDS, MODEL_ROOT, TRAINER_NAME,
    prepare_model_results,
)


DEFAULT_CROP_SIZE = np.asarray((160, 160, 110), dtype=int)
DEFAULT_CENTER_OFFSET = np.asarray((0.0, 41.0, 7.0), dtype=float)
DEFAULT_TARGET_SPACING = np.asarray((1.0, 1.0, 1.0), dtype=float)
DEFAULT_OUTPUT_NAMES = {
    "mask": "avp_mask.nii.gz",
    "prob": "avp_prob.nii.gz",
    "five_label": "avp_5label.nii.gz",
    "metrics": "optic_nerve_metrics.xlsx",
    "qc": "qc.html",
}


@dataclass
class WorkImages:
    original_t1w: nib.Nifti1Image
    original_synthseg: nib.Nifti1Image
    work_t1w: nib.Nifti1Image
    work_synthseg: nib.Nifti1Image
    resampled_to_training_spacing: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-case AVP/optic nerve pipeline.")
    parser.add_argument("paths", nargs="*", help="T1w.nii.gz synthseg.nii.gz [output_dir]")
    parser.add_argument("--t1w", type=Path, default=None, help="Input T1w NIfTI.")
    parser.add_argument("--synthseg", type=Path, default=None, help="SynthSeg NIfTI in T1w space.")
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="Output folder (default: avp_seg_output).")
    parser.add_argument("--target-spacing", type=float, nargs=3, default=tuple(DEFAULT_TARGET_SPACING))
    parser.add_argument("--crop-size", type=int, nargs=3, default=tuple(DEFAULT_CROP_SIZE))
    parser.add_argument("--center-offset", type=float, nargs=3, default=tuple(DEFAULT_CENTER_OFFSET))
    parser.add_argument("--prob-threshold", type=float, default=0.5)
    parser.add_argument("--mask-name", default=DEFAULT_OUTPUT_NAMES["mask"], help="Output AVP mask file name.")
    parser.add_argument("--prob-name", default=DEFAULT_OUTPUT_NAMES["prob"], help="Output AVP probability file name.")
    parser.add_argument("--five-label-name", default=DEFAULT_OUTPUT_NAMES["five_label"], help="Output 5-label segmentation file name.")
    parser.add_argument("--metrics-name", default=DEFAULT_OUTPUT_NAMES["metrics"], help="Output optic nerve metrics file name.")
    parser.add_argument("--qc-name", default=DEFAULT_OUTPUT_NAMES["qc"], help="Output QC HTML file name.")
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--write-provenance", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.paths:
        if len(args.paths) not in (2, 3):
            parser.error("Usage: avp_seg T1w synthseg [output_dir]")
        if args.t1w is None:
            args.t1w = Path(args.paths[0])
        if args.synthseg is None:
            args.synthseg = Path(args.paths[1])
        if args.output_dir is None and len(args.paths) == 3:
            args.output_dir = Path(args.paths[2])
    if args.output_dir is None:
        args.output_dir = Path("avp_seg_output")
    if args.t1w is None or args.synthseg is None:
        parser.error("Provide T1w and SynthSeg paths.")
    for attr in ("mask_name", "prob_name", "five_label_name", "metrics_name", "qc_name"):
        validate_output_filename(parser, getattr(args, attr), attr.replace("_", "-"))
    output_names = [args.mask_name, args.prob_name, args.five_label_name, args.metrics_name, args.qc_name]
    if len(set(output_names)) != len(output_names):
        parser.error("Output file names must be unique.")
    return args


def validate_output_filename(parser: argparse.ArgumentParser, value: str, option_name: str) -> None:
    path = Path(value)
    if path.name != value or path.is_absolute() or value in {"", ".", ".."}:
        parser.error(f"--{option_name} must be a file name only, not a path: {value!r}")
    if re.search(r'[<>:"/\\\\|?*]', value):
        parser.error(f"--{option_name} contains characters that are not portable in file names: {value!r}")


def voxel_sizes(img: nib.Nifti1Image) -> np.ndarray:
    return np.asarray(img.header.get_zooms()[:3], dtype=float)


def close_spacing(a: np.ndarray, b: np.ndarray, tol: float = 1e-3) -> bool:
    return bool(np.allclose(a, b, atol=tol, rtol=0))


def check_t1w_synthseg_compatibility(t1w: nib.Nifti1Image, synthseg: nib.Nifti1Image) -> None:
    if t1w.shape[:3] != synthseg.shape[:3]:
        raise ValueError(f"T1w and SynthSeg shape mismatch: {t1w.shape[:3]} vs {synthseg.shape[:3]}")
    if not close_spacing(voxel_sizes(t1w), voxel_sizes(synthseg), tol=1e-3):
        raise ValueError(f"T1w and SynthSeg spacing mismatch: {voxel_sizes(t1w)} vs {voxel_sizes(synthseg)}")
    if not np.allclose(t1w.affine, synthseg.affine, atol=1e-2):
        raise ValueError("T1w and SynthSeg affine matrices are not sufficiently aligned.")


def make_work_images(t1w_path: Path, synthseg_path: Path, target_spacing: np.ndarray) -> WorkImages:
    t1w = nib.load(str(t1w_path))
    synthseg = nib.load(str(synthseg_path))
    check_t1w_synthseg_compatibility(t1w, synthseg)

    need_resample = not close_spacing(voxel_sizes(t1w), target_spacing, tol=1e-3)
    if need_resample:
        work_t1w = resample_to_output(t1w, voxel_sizes=tuple(target_spacing), order=3)
        work_synthseg = resample_from_to(synthseg, (work_t1w.shape[:3], work_t1w.affine), order=0)
    else:
        work_t1w = t1w
        work_synthseg = synthseg
    return WorkImages(t1w, synthseg, work_t1w, work_synthseg, need_resample)


def save_like(data: np.ndarray, ref: nib.Nifti1Image, path: Path, dtype: np.dtype) -> None:
    header = ref.header.copy()
    header.set_data_dtype(dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(dtype), ref.affine, header), str(path))


def crop_bounds_allow_padding(center: np.ndarray, crop_size: np.ndarray, shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start = np.rint(center - crop_size / 2).astype(int)
    for axis in range(3):
        if shape[axis] >= crop_size[axis]:
            start[axis] = min(max(start[axis], 0), shape[axis] - crop_size[axis])
        else:
            start[axis] = int(np.floor((shape[axis] - crop_size[axis]) / 2))
    return start, start + crop_size


def crop_img_allow_padding(
    img: nib.Nifti1Image,
    start: np.ndarray,
    stop: np.ndarray,
    pad_value: float = 0,
) -> tuple[nib.Nifti1Image, dict[str, np.ndarray]]:
    shape = np.asarray(img.shape[:3], dtype=int)
    crop_size = stop - start
    src_start = np.maximum(start, 0)
    src_stop = np.minimum(stop, shape)
    dst_start = src_start - start
    dst_stop = dst_start + (src_stop - src_start)

    data = np.asanyarray(img.dataobj)
    cropped = np.full(tuple(crop_size), pad_value, dtype=data.dtype)
    cropped[
        dst_start[0] : dst_stop[0],
        dst_start[1] : dst_stop[1],
        dst_start[2] : dst_stop[2],
    ] = data[
        src_start[0] : src_stop[0],
        src_start[1] : src_stop[1],
        src_start[2] : src_stop[2],
    ]

    affine = img.affine.copy()
    affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ start.astype(float)
    meta = {
        "source_start": src_start,
        "source_stop": src_stop,
        "insert_start": dst_start,
        "insert_stop": dst_stop,
    }
    return nib.Nifti1Image(cropped, affine, img.header.copy()), meta


def prepare_voi(work: WorkImages, work_dir: Path, crop_size: np.ndarray, center_offset: np.ndarray) -> dict[str, object]:
    images_ts = work_dir / "imagesTs"
    images_ts.mkdir(parents=True, exist_ok=True)

    synth = np.asanyarray(work.work_synthseg.dataobj).astype(np.int32)
    anchor_idx = np.argwhere(np.isin(synth, ANCHOR_LABELS))
    if anchor_idx.size == 0:
        raise ValueError(f"SynthSeg has no dynamic VOI anchor labels: {ANCHOR_LABELS}")

    anchor_center = anchor_idx.mean(axis=0)
    center = anchor_center + center_offset
    shape = np.asarray(work.work_t1w.shape[:3], dtype=int)
    if np.any(shape < crop_size):
        start, stop = crop_bounds_allow_padding(center, crop_size, shape)
    else:
        start, stop = crop_bounds(center, crop_size, shape)
    crop, crop_meta = crop_img_allow_padding(work.work_t1w, start, stop, pad_value=0)
    input_path = images_ts / "case_0000.nii.gz"
    nib.save(crop, str(input_path))

    return {
        "imagesTs": images_ts,
        "input_path": input_path,
        "crop_start": start,
        "crop_stop": stop,
        "crop_source_start": crop_meta["source_start"],
        "crop_source_stop": crop_meta["source_stop"],
        "crop_insert_start": crop_meta["insert_start"],
        "crop_insert_stop": crop_meta["insert_stop"],
        "anchor_center": anchor_center,
        "crop_center": center,
    }


def predict_voi(args: argparse.Namespace, images_ts: Path, pred_dir: Path, work_dir: Path) -> None:
    if importlib.util.find_spec("nnunetv2") is None:
        raise RuntimeError("Install the inference runtime with: bash install.sh")
    pred_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["nnUNet_raw"] = str(work_dir / "nnUNet_raw")
    env["nnUNet_preprocessed"] = str(work_dir / "nnUNet_preprocessed")
    env["nnUNet_results"] = str(prepare_model_results(work_dir))
    cmd = [
        sys.executable, "-m", "avpseg.nnunet_predict",
        "-i", str(images_ts), "-o", str(pred_dir),
        "-d", DATASET, "-c", CONFIGURATION,
        "-f", *(str(fold) for fold in FOLDS),
        "-tr", TRAINER_NAME, "-chk", CHECKPOINT, "--save_probabilities",
    ]
    subprocess.run(cmd, env=env, check=True)


def restore_to_work_space(work: WorkImages, crop_info: dict[str, object], pred_dir: Path) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    pred_path = pred_dir / "case.nii.gz"
    prob_path = pred_dir / "case.npz"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not prob_path.exists():
        raise FileNotFoundError(prob_path)

    pred_img = nib.load(str(pred_path))
    pred = np.asanyarray(pred_img.dataobj).astype(np.uint8)
    probs = np.load(prob_path)["probabilities"]
    fg_prob = match_probability_to_prediction(probs[1].astype(np.float32), pred)

    start = np.asarray(crop_info["crop_start"], dtype=int)
    stop = np.asarray(crop_info["crop_stop"], dtype=int)
    src_start = np.asarray(crop_info.get("crop_source_start", np.maximum(start, 0)), dtype=int)
    src_stop = np.asarray(crop_info.get("crop_source_stop", np.minimum(stop, work.work_t1w.shape[:3])), dtype=int)
    ins_start = np.asarray(crop_info.get("crop_insert_start", src_start - start), dtype=int)
    ins_stop = np.asarray(crop_info.get("crop_insert_stop", ins_start + (src_stop - src_start)), dtype=int)
    full_prob = np.zeros(work.work_t1w.shape[:3], dtype=np.float32)
    full_prob[
        src_start[0] : src_stop[0],
        src_start[1] : src_stop[1],
        src_start[2] : src_stop[2],
    ] = fg_prob[
        ins_start[0] : ins_stop[0],
        ins_start[1] : ins_stop[1],
        ins_start[2] : ins_stop[2],
    ]
    full_mask = (full_prob >= 0.5).astype(np.uint8)
    return (
        nib.Nifti1Image(full_mask, work.work_t1w.affine, work.work_t1w.header.copy()),
        nib.Nifti1Image(full_prob, work.work_t1w.affine, work.work_t1w.header.copy()),
    )


def resample_outputs_to_original(
    work: WorkImages,
    work_mask: nib.Nifti1Image,
    work_prob: nib.Nifti1Image,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if work.resampled_to_training_spacing:
        target = (work.original_t1w.shape[:3], work.original_t1w.affine)
        prob_img = resample_from_to(work_prob, target, order=1)
        prob = np.clip(np.asanyarray(prob_img.dataobj).astype(np.float32), 0.0, 1.0)
    else:
        prob = np.asanyarray(work_prob.dataobj).astype(np.float32)
    mask = (prob >= threshold).astype(np.uint8)
    return mask, prob


def enforce_left_right_by_world_x(seg: np.ndarray, affine: np.ndarray) -> np.ndarray:
    out = seg.copy()
    for left_label, right_label in ((1, 2), (4, 5)):
        pts_left = np.argwhere(seg == left_label)
        pts_right = np.argwhere(seg == right_label)
        if pts_left.size == 0 or pts_right.size == 0:
            continue
        left_x = np.mean(nib.affines.apply_affine(affine, pts_left)[:, 0])
        right_x = np.mean(nib.affines.apply_affine(affine, pts_right)[:, 0])
        if left_x > right_x:
            out[seg == left_label] = right_label
            out[seg == right_label] = left_label
    return out


def make_five_label(mask: np.ndarray, ref: nib.Nifti1Image) -> np.ndarray:
    spacing = voxel_sizes(ref)
    seg, _, _, _ = split_mask(
        mask > 0,
        spacing=spacing,
        min_component_voxels=10,
        chiasm_path_mm=6.0,
        compact_chiasm_core=True,
        chiasm_core_radii_mm=np.asarray((6.0, 6.0, 4.0), dtype=float),
        separate_pairs=True,
        separator_iterations=2,
    )
    return enforce_left_right_by_world_x(seg.astype(np.uint8), ref.affine)


def skeleton_profile(mask: np.ndarray, spacing: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    crop, start = crop_to_mask(mask, margin=2)
    skel = skeletonize_3d(crop) > 0
    edt = ndi.distance_transform_edt(crop, sampling=tuple(spacing))
    coords = np.argwhere(skel & crop)
    if coords.size == 0:
        return np.asarray([]), np.asarray([]), np.empty((0, 3), dtype=int)
    radii = edt[tuple(coords.T)]
    positions_lookup = centerline_positions_mm(skel, spacing)
    positions = np.asarray([positions_lookup.get(tuple(c.tolist()), np.nan) for c in coords], dtype=float)
    finite = np.isfinite(positions)
    return 2.0 * radii[finite].astype(float), positions[finite], coords[finite] + start


def nerve_metrics(seg: np.ndarray, ref: nib.Nifti1Image) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    spacing = voxel_sizes(ref)
    voxel_volume = float(np.prod(spacing))
    rows: list[dict[str, float | str]] = []
    profiles: dict[str, pd.DataFrame] = {}

    for label in (1, 2):
        name = FIVE_LABEL_NAMES[label]
        mask = seg == label
        metric_mask, bridge_voxels = bridge_mask_components(mask, spacing)
        volume = float(np.count_nonzero(mask) * voxel_volume)
        length, skel_voxels, endpoints = centerline_length_mm(metric_mask, spacing)
        diameters, positions, coords = skeleton_profile(metric_mask, spacing)
        surface = surface_area_mm2(mask, spacing)
        diameter_mean = float(np.mean(diameters)) if diameters.size else np.nan
        diameter_std = float(np.std(diameters)) if diameters.size else np.nan
        mean_area = float(volume / length) if length > 0 else np.nan
        equiv_diameter = math.sqrt(4.0 * mean_area / math.pi) if mean_area > 0 else np.nan
        surface_to_volume = float(surface / volume) if volume > 0 else np.nan
        tortuosity = np.nan
        finite_pos = np.isfinite(positions)
        if np.count_nonzero(finite_pos) >= 2 and length > 0:
            a = coords[finite_pos][np.argmin(positions[finite_pos])]
            b = coords[finite_pos][np.argmax(positions[finite_pos])]
            end_distance = float(np.linalg.norm((a - b) * spacing))
            tortuosity = float(length / end_distance) if end_distance > 0 else np.nan

        row = {
            "region": name,
            "label": label,
            "voxels": int(np.count_nonzero(mask)),
            "volume_mm3": volume,
            "centerline_length_mm": float(length),
            "mean_diameter_mm": diameter_mean,
            "equiv_diameter_mm": equiv_diameter,
            "mean_cross_section_area_mm2": mean_area,
            "diameter_cv": float(diameter_std / diameter_mean) if diameter_mean and np.isfinite(diameter_mean) else np.nan,
            "tortuosity": tortuosity,
            "surface_to_volume_ratio": surface_to_volume,
            "diameter_samples": int(diameters.size),
            "skeleton_voxels": int(skel_voxels),
            "skeleton_endpoints": int(endpoints),
            "metric_bridge_voxels": int(bridge_voxels),
        }
        rows.append(row)
        if diameters.size:
            finite_position = np.isfinite(positions)
            max_pos = float(np.max(positions[finite_position])) if np.any(finite_position) else 0.0
            position_norm = np.zeros_like(positions, dtype=float)
            if max_pos > 0:
                position_norm[finite_position] = positions[finite_position] / max_pos
            profiles[name] = pd.DataFrame(
                {
                    "position_mm": positions,
                    "position_norm": position_norm,
                    "diameter_mm": diameters,
                    "i": coords[:, 0],
                    "j": coords[:, 1],
                    "k": coords[:, 2],
                }
            )
        else:
            profiles[name] = pd.DataFrame(columns=["position_mm", "position_norm", "diameter_mm", "i", "j", "k"])

    return pd.DataFrame(rows), profiles


def write_metrics_xlsx(path: Path, metrics: pd.DataFrame) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        metrics.to_excel(writer, index=False, sheet_name="optic_nerve_metrics")


def normalize_t1(t1: np.ndarray, brain: np.ndarray | None = None) -> np.ndarray:
    sample = t1[brain] if brain is not None and np.any(brain) else t1[np.isfinite(t1)]
    lo, hi = np.percentile(sample, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((t1.astype(np.float32) - lo) / (hi - lo), 0, 1)


def bbox(mask: np.ndarray, margin: int = 8) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return np.zeros(3, dtype=int), np.asarray(mask.shape, dtype=int)
    start = np.maximum(pts.min(axis=0) - margin, 0)
    stop = np.minimum(pts.max(axis=0) + margin + 1, mask.shape)
    return start, stop


def crop(arr: np.ndarray, start: np.ndarray, stop: np.ndarray) -> np.ndarray:
    return arr[start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]]


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def draw_overlay(ax, bg: np.ndarray, label: np.ndarray, colors: dict[int, str], title: str) -> None:
    ax.imshow(bg.T, cmap="gray", origin="lower", interpolation="nearest")
    for lab, color in colors.items():
        mask = (label == lab).T
        if np.any(mask):
            ax.contour(mask, levels=[0.5], colors=[color], linewidths=1.0, origin="lower")
            ax.imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap([(0, 0, 0, 0), color]), alpha=0.32, origin="lower")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def mesh_from_binary(mask: np.ndarray, step_size: int = 1) -> tuple[np.ndarray, np.ndarray] | None:
    if not np.any(mask):
        return None
    padded = np.pad(mask.astype(np.float32), 1, mode="constant")
    try:
        verts, faces, _, _ = marching_cubes(padded, level=0.5, step_size=step_size)
    except ValueError:
        return None
    return verts - 1.0, faces


def add_mesh_to_axis(ax, mesh: tuple[np.ndarray, np.ndarray] | None, color: str, alpha: float, linewidth: float = 0.0) -> None:
    if mesh is None:
        return
    verts, faces = mesh
    rgba = to_rgba(color, alpha)
    collection = Poly3DCollection(
        verts[faces],
        facecolors=[rgba],
        edgecolors="none" if linewidth == 0 else [to_rgba(color, min(alpha + 0.15, 1.0))],
        linewidths=linewidth,
    )
    collection.set_alpha(alpha)
    ax.add_collection3d(collection)


def build_glass_meshes(
    brain: np.ndarray,
    label: np.ndarray,
    colors: dict[int, str],
) -> tuple[tuple[np.ndarray, np.ndarray] | None, list[tuple[tuple[np.ndarray, np.ndarray] | None, str]]]:
    brain_mesh = mesh_from_binary(brain, step_size=7)
    roi_meshes = [(mesh_from_binary(label == lab, step_size=1), color) for lab, color in colors.items()]
    return brain_mesh, roi_meshes


def set_3d_axes(ax, shape: tuple[int, int, int], elev: float, azim: float) -> None:
    ax.set_xlim(0, shape[0])
    ax.set_ylim(0, shape[1])
    ax.set_zlim(0, shape[2])
    ax.set_box_aspect(shape)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def glass_brain_png(brain: np.ndarray, label: np.ndarray, colors: dict[int, str], title: str) -> str:
    brain_mesh, roi_meshes = build_glass_meshes(brain, label, colors)
    fig = plt.figure(figsize=(9, 3.2), constrained_layout=True)
    views = (("axial-ish", 72, -90), ("coronal-ish", 8, -90), ("sagittal-ish", 8, 0))
    for idx, (name, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        add_mesh_to_axis(ax, brain_mesh, "#d0d0d0", alpha=0.08)
        for roi_mesh, color in roi_meshes:
            add_mesh_to_axis(ax, roi_mesh, color, alpha=0.96)
        set_3d_axes(ax, brain.shape, elev=elev, azim=azim)
        ax.set_title(name, fontsize=9)
    fig.suptitle(title)
    return fig_to_base64(fig)


def overlay_montage(t1: np.ndarray, label: np.ndarray, colors: dict[int, str], title: str) -> str:
    start, stop = bbox(label > 0, margin=10)
    t1c = crop(t1, start, stop)
    labc = crop(label, start, stop)
    t1c = normalize_t1(t1c, labc > 0)
    pts = np.argwhere(labc > 0)
    fig, axes = plt.subplots(3, 5, figsize=(12, 7), constrained_layout=True)
    for row, (axis, name) in enumerate(((2, "axial"), (1, "coronal"), (0, "sagittal"))):
        qs = np.linspace(0.08, 0.92, 5)
        indices = np.unique(np.quantile(pts[:, axis], qs).astype(int))
        if indices.size < 5:
            indices = np.linspace(pts[:, axis].min(), pts[:, axis].max(), 5).astype(int)
        for col, idx in enumerate(indices[:5]):
            ax = axes[row, col]
            if axis == 2:
                bg, lab = t1c[:, :, idx], labc[:, :, idx]
            elif axis == 1:
                bg, lab = t1c[:, idx, :], labc[:, idx, :]
            else:
                bg, lab = t1c[idx, :, :], labc[idx, :, :]
            draw_overlay(ax, bg, lab, colors, f"{name} {idx}")
    fig.suptitle(title)
    return fig_to_base64(fig)


def diameter_curve_png(profiles: dict[str, pd.DataFrame]) -> str:
    fig, ax = plt.subplots(figsize=(6.2, 3.7), constrained_layout=True)
    colors = {"left_optic_nerve": "#2f80ed", "right_optic_nerve": "#eb5757"}
    for region, df in profiles.items():
        if df.empty:
            continue
        bins = np.linspace(0, 1, 16)
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sub = df[(df["position_norm"] >= lo) & (df["position_norm"] < hi if hi < 1 else df["position_norm"] <= hi)]
            if len(sub):
                xs.append((lo + hi) / 2)
                ys.append(float(sub["diameter_mm"].mean()))
        ax.plot(xs, ys, marker="o", linewidth=2, label=region.replace("_", " "), color=colors.get(region))
    ax.set_xlabel("Normalized position along optic nerve (chiasm to orbit)")
    ax.set_ylabel("Mean local diameter (mm)")
    ax.set_title("Optic nerve diameter QC profile")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return fig_to_base64(fig)


def render_qc_html(
    path: Path,
    t1: np.ndarray,
    synthseg: np.ndarray,
    avp_mask: np.ndarray,
    five_label: np.ndarray,
    profiles: dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
) -> None:
    brain = synthseg > 0
    start, stop = bbox(brain | (avp_mask > 0), margin=12)
    brain_c = crop(brain, start, stop)
    avp_c = crop(avp_mask, start, stop)
    five_c = crop(five_label, start, stop)

    avp_colors = {1: "#e31a1c"}
    five_class = np.zeros_like(five_c, dtype=np.uint8)
    five_class[np.isin(five_c, (1, 2))] = 1
    five_class[five_c == 3] = 2
    five_class[np.isin(five_c, (4, 5))] = 3
    five_colors = {1: "#2f80ed", 2: "#f2c94c", 3: "#27ae60"}
    on_colors = {1: "#2f80ed", 2: "#eb5757"}

    avp_proj = glass_brain_png(brain_c, (avp_c > 0).astype(np.uint8), avp_colors, "AVP whole-mask glass brain")
    avp_overlay = overlay_montage(t1, avp_mask.astype(np.uint8), avp_colors, "AVP 2D overlays")
    five_proj = glass_brain_png(brain_c, five_class, five_colors, "5-label glass brain: ON / chiasm / OT")
    on_proj = glass_brain_png(brain_c, np.where(np.isin(five_c, (1, 2)), five_c, 0), on_colors, "Left/right optic nerves")
    curve = diameter_curve_png(profiles)
    metrics_html = metrics.to_html(index=False, float_format=lambda x: f"{x:.4f}" if np.isfinite(x) else "")

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>AVP QC</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
h1, h2 {{ margin-bottom: 0.4em; }}
.grid {{ display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; }}
img {{ max-width: 100%; border: 1px solid #ddd; background: white; }}
table {{ border-collapse: collapse; font-size: 12px; }}
td, th {{ border: 1px solid #ddd; padding: 4px 6px; }}
</style>
</head>
<body>
<h1>AVP / optic nerve QC</h1>
<h2>Whole AVP glass brain</h2>
<img src="data:image/png;base64,{avp_proj}">
<h2>AVP 2D overlays</h2>
<img src="data:image/png;base64,{avp_overlay}">
<h2>Five-label glass brain</h2>
<img src="data:image/png;base64,{five_proj}">
<h2>Left/right optic nerves</h2>
<img src="data:image/png;base64,{on_proj}">
<h2>Optic nerve diameter curve</h2>
<img src="data:image/png;base64,{curve}">
<h2>Metrics</h2>
{metrics_html}
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {
        "mask": args.output_dir / args.mask_name,
        "prob": args.output_dir / args.prob_name,
        "five_label": args.output_dir / args.five_label_name,
        "metrics": args.output_dir / args.metrics_name,
        "qc": args.output_dir / args.qc_name,
    }
    if not args.overwrite:
        existing = [p for p in final_paths.values() if p.exists()]
        if existing:
            raise FileExistsError(f"Output exists; pass --overwrite: {existing}")

    with tempfile.TemporaryDirectory(prefix="avp_pipeline_", dir=str(args.output_dir)) as tmp:
        work_dir = Path(tmp)
        work = make_work_images(args.t1w, args.synthseg, np.asarray(args.target_spacing, dtype=float))
        crop_info = prepare_voi(work, work_dir, np.asarray(args.crop_size, dtype=int), np.asarray(args.center_offset))
        pred_dir = work_dir / "predictions"
        predict_voi(args, crop_info["imagesTs"], pred_dir, work_dir)
        work_mask_img, work_prob_img = restore_to_work_space(work, crop_info, pred_dir)
        avp_mask, avp_prob = resample_outputs_to_original(work, work_mask_img, work_prob_img, args.prob_threshold)

        five_label = make_five_label(avp_mask, work.original_t1w)
        metrics, profiles = nerve_metrics(five_label, work.original_t1w)

        save_like(avp_mask, work.original_t1w, final_paths["mask"], np.uint8)
        save_like(avp_prob, work.original_t1w, final_paths["prob"], np.float32)
        save_like(five_label, work.original_t1w, final_paths["five_label"], np.uint8)
        write_metrics_xlsx(final_paths["metrics"], metrics)

        t1 = np.asanyarray(work.original_t1w.dataobj).astype(np.float32)
        synth = np.asanyarray(work.original_synthseg.dataobj)
        render_qc_html(final_paths["qc"], t1, synth, avp_mask, five_label, profiles, metrics)

        if args.keep_work:
            keep_dir = args.output_dir / "_work"
            if keep_dir.exists():
                shutil.rmtree(keep_dir)
            shutil.copytree(work_dir, keep_dir)
        if args.write_provenance:
            provenance = {
                "t1w": str(args.t1w),
                "synthseg": str(args.synthseg),
                "resampled_to_training_spacing": work.resampled_to_training_spacing,
                "original_spacing": voxel_sizes(work.original_t1w).tolist(),
                "work_spacing": voxel_sizes(work.work_t1w).tolist(),
                "crop_start": np.asarray(crop_info["crop_start"]).astype(int).tolist(),
                "crop_stop": np.asarray(crop_info["crop_stop"]).astype(int).tolist(),
                "folds": list(FOLDS),
                "checkpoint": CHECKPOINT,
                "model_root": str(MODEL_ROOT),
                "outputs": {name: str(path) for name, path in final_paths.items()},
            }
            if args.keep_work:
                provenance["work_dir"] = str(args.output_dir / "_work")
            (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print("AVP pipeline complete.")
    for path in final_paths.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
