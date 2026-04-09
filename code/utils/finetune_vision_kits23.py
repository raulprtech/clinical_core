"""
Fine-tuning script: STU-Net pretrained → KiTS23 (kidney + tumor + cyst)
========================================================================

This script is INDEPENDENT of the experiment runner. It is run ONCE to
produce a fine-tuned STU-Net checkpoint that adds explicit tumor segmentation
to the baseline kidney-only segmentation.

WHEN TO RUN:
  Only after the kidney-only baseline (TotalSegmentator backend) is working
  end-to-end. The kidney-only baseline is sufficient for the first multimodal
  ablation; tumor segmentation is an upgrade that improves VISION-CONN's
  contribution to the final C-index.

REQUIREMENTS:
  - GPU with at least 8GB VRAM (12GB recommended)
  - KiTS23 dataset downloaded and preprocessed in nnUNetv2 format
  - nnUNetv2 installed
  - STU-Net repo cloned and STUNetTrainer files patched into nnunetv2
  - Pretrained STU-Net checkpoint (B size recommended for first attempt)

ENVIRONMENT VARIABLES:
  nnUNet_raw         path to raw KiTS23 data in nnUNet format
  nnUNet_preprocessed
  nnUNet_results

USAGE:
  python finetune_vision_kits23.py --pretrained /path/to/stunet_b.pth \\
                                    --dataset_id 220 \\
                                    --epochs 100

This is a SCAFFOLD. The actual training loop delegates to nnUNetv2's CLI
because reimplementing it here would duplicate code that already works
upstream. The role of this file is to:
  (1) Document the exact procedure
  (2) Validate that the environment is set up correctly before launching
  (3) Provide a reproducible entry point that the experiment_config can
      reference for traceability
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_environment() -> dict:
    """Verifies that nnUNetv2, STU-Net trainer, and env vars are configured."""
    issues = []
    info = {}

    try:
        import nnunetv2
        info['nnunetv2_version'] = getattr(nnunetv2, '__version__', 'unknown')
    except ImportError:
        issues.append("nnUNetv2 not installed. Run: pip install nnunetv2")

    for var in ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']:
        val = os.environ.get(var)
        if val is None:
            issues.append(f"Environment variable {var} not set")
        elif not Path(val).exists():
            issues.append(f"{var}={val} does not exist")
        else:
            info[var] = val

    try:
        from nnunetv2.training.nnUNetTrainer.variants.STUNetTrainer import (  # noqa
            STUNetTrainer_base,
        )
        info['stunet_trainer_patched'] = True
    except ImportError:
        issues.append(
            "STUNetTrainer not found in nnunetv2 installation. "
            "Patch the STU-Net repo files into nnunetv2/training/nnUNetTrainer/variants/ "
            "following the instructions at https://github.com/uni-medical/STU-Net"
        )
        info['stunet_trainer_patched'] = False

    try:
        import torch
        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
            if info['gpu_memory_gb'] < 8:
                issues.append(
                    f"GPU has only {info['gpu_memory_gb']}GB. STU-Net-B fine-tuning "
                    "needs at least 8GB. Consider STU-Net-S or use Google Colab Pro+."
                )
        else:
            issues.append("No CUDA GPU available. Fine-tuning requires GPU.")
    except ImportError:
        issues.append("PyTorch not installed")

    return {'info': info, 'issues': issues}


def launch_finetuning(
    pretrained_path: str,
    dataset_id: int,
    epochs: int,
    fold: int,
    trainer_size: str,
):
    """
    Launches the actual nnUNetv2 training command.
    """
    trainer_name = f"STUNetTrainer_{trainer_size}_ft"

    cmd = [
        sys.executable, "-m", "nnunetv2.run.run_training",
        f"Dataset{dataset_id:03d}",
        "3d_fullres",
        str(fold),
        "-tr", trainer_name,
        "-pretrained_weights", pretrained_path,
        "--c",
    ]

    env = os.environ.copy()
    env['nnUNet_n_epochs'] = str(epochs)

    print("Launching:", " ".join(cmd))
    print(f"Epochs override: {epochs}")
    result = subprocess.run(cmd, env=env)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune STU-Net pretrained on TotalSegmentator → KiTS23",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--pretrained',
        required=False,
        help="Path to STU-Net pretrained checkpoint (.pth)",
    )
    parser.add_argument(
        '--dataset_id',
        type=int,
        default=220,
        help="nnUNet dataset ID for KiTS23 (default: 220)",
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help="Number of fine-tuning epochs (default: 100)",
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=0,
        help="Cross-validation fold (default: 0)",
    )
    parser.add_argument(
        '--trainer_size',
        choices=['small', 'base', 'large', 'huge'],
        default='base',
        help="STU-Net size variant (default: base)",
    )
    parser.add_argument(
        '--check_only',
        action='store_true',
        help="Only verify environment, do not launch training",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("STU-Net fine-tuning environment check")
    print("=" * 60)

    check = check_environment()

    print("\nDetected:")
    for k, v in check['info'].items():
        print(f"  {k}: {v}")

    if check['issues']:
        print("\nIssues found:")
        for issue in check['issues']:
            print(f"  - {issue}")
        print("\nFix these before running training.")
        return 1

    print("\nEnvironment OK.")

    if args.check_only:
        return 0

    if not args.pretrained:
        print("\n--pretrained is required to launch training.")
        return 1

    if not Path(args.pretrained).exists():
        print(f"\nPretrained checkpoint not found: {args.pretrained}")
        return 1

    print(f"\nLaunching fine-tuning:")
    print(f"  Pretrained: {args.pretrained}")
    print(f"  Dataset:    {args.dataset_id}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Fold:       {args.fold}")
    print(f"  Trainer:    STUNetTrainer_{args.trainer_size}_ft")

    return launch_finetuning(
        pretrained_path=args.pretrained,
        dataset_id=args.dataset_id,
        epochs=args.epochs,
        fold=args.fold,
        trainer_size=args.trainer_size,
    )


if __name__ == "__main__":
    sys.exit(main())
