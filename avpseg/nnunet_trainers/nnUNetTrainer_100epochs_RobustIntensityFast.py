import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import (
    BGContrast,
    ComposeTransforms,
    ContrastTransform,
    GammaTransform,
    MultiplicativeBrightnessTransform,
    RandomTransform,
    nnUNetTrainer,
)


class nnUNetTrainer_100epochs_RobustIntensityFast(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100

    @staticmethod
    def get_training_transforms(
        patch_size,
        rotation_for_DA,
        deep_supervision_scales,
        mirror_axes,
        do_dummy_2d_data_aug,
        use_mask_for_norm=None,
        is_cascaded=False,
        foreground_labels=None,
        regions=None,
        ignore_label=None,
    ):
        default_transforms = nnUNetTrainer.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm,
            is_cascaded,
            foreground_labels,
            regions,
            ignore_label,
        )
        robust_intensity = [
            RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast((0.45, 1.8)),
                    synchronize_channels=False,
                    p_per_channel=1,
                ),
                apply_probability=0.3,
            ),
            RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast((0.45, 2.2)),
                    preserve_range=False,
                    synchronize_channels=False,
                    p_per_channel=1,
                ),
                apply_probability=0.3,
            ),
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.45, 2.2)),
                    p_invert_image=1,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                apply_probability=0.25,
            ),
        ]
        return ComposeTransforms([*robust_intensity, default_transforms])
