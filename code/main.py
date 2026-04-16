import sys
from pathlib import Path
import argparse

# 1. Add project root to path so 'core' and 'components' can be found
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

# 2. Add 'core' to path for backward compatibility if needed, 
# though 'core.registry' etc is preferred.
sys.path.append(str(root_dir / "core"))

# 3. Import the runner and main pipeline
from core.experiment_runner import run_experiment
from core.main import MultimodalPipeline

def main():
    parser = argparse.ArgumentParser(description="CLINICAL-CORE Modular Runner Shim")
    parser.add_argument('--config', type=str, default='experiments/experiment_config.yaml',
                        help='Path to experiment config YAML')
    parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep (tabular only)')
    parser.add_argument('--validate', action='store_true', help='Run top-3 validation (tabular only)')
    
    args = parser.parse_args()
    
    if args.sweep:
        print("Redirecting to sweep.py...")
        from components.tabular.utils.sweep import main as sweep_main
        # Override sys.argv for the sub-module
        sys.argv = [sys.argv[0], '--config', args.config]
        sweep_main()
    elif args.validate:
        print("Redirecting to validate_top3.py...")
        from components.tabular.utils.validate_top3 import main as validate_main
        sys.argv = [sys.argv[0], '--config', args.config]
        validate_main()
    else:
        print(f"Running experiment with config: {args.config}")
        run_experiment(args.config)

if __name__ == "__main__":
    main()
