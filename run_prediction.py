import os
import argparse
from datetime import datetime
import subprocess
from tqdm import tqdm

def main(yaml_dir, output_dir):
    # Create main timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join(output_dir, f"{timestamp}_predictions")
    os.makedirs(main_output_dir, exist_ok=True)

    # Get and sort YAML files
    yaml_files = sorted([
        f for f in os.listdir(yaml_dir)
        if f.endswith(".yaml")
    ])

    if not yaml_files:
        print("❌ No YAML files found in the provided directory.")
        return

    for yaml_file in tqdm(yaml_files, desc="Running predictions"):
        seq_number = os.path.splitext(yaml_file)[0]
        input_path = os.path.join(yaml_dir, yaml_file)

        # Subdirectory named after the seq_number (not timestamp)
        sub_output_dir = os.path.join(main_output_dir, seq_number)
        os.makedirs(sub_output_dir, exist_ok=True)

        print(f"🔮 Running boltz prediction for {yaml_file} → {sub_output_dir}")
        subprocess.run([
            "boltz", "predict", input_path, "--out_dir", sub_output_dir
        ])

    print(f"✅ All predictions saved in: {main_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run boltz prediction on YAML files.")
    parser.add_argument("yaml_dir", help="Directory containing YAML files")
    parser.add_argument("output_dir", help="Directory where predictions should be saved")
    args = parser.parse_args()

    main(args.yaml_dir, args.output_dir)
# example usage:
# python run_prediction.py /workspace/gfp_function_prediction/data/20250805_051302_subsample/subsample /workspace/gfp_function_prediction/data/outputs