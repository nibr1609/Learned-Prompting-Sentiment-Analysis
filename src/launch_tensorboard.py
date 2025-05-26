import os
import argparse
import subprocess
import sys

def find_model_folder(base_path, id_substring):
    for folder_name in os.listdir(base_path):
        if id_substring in folder_name and os.path.isdir(os.path.join(base_path, folder_name)):
            return os.path.join(base_path, folder_name)
    return None

def main():
    # This is used to launch tensorboard for experiment id (given that experiment created tensorboard logs)
    parser = argparse.ArgumentParser(description="Launch TensorBoard for a given model ID.")
    parser.add_argument("--id", required=True, help="ID substring to search for in model folder names.")
    args = parser.parse_args()

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    model_path = find_model_folder(base_path, args.id)

    if model_path is None:
        print(f"No folder found in {base_path} containing ID: {args.id}")
        sys.exit(1)

    logs_path = os.path.join(model_path, "logs")
    if not os.path.isdir(logs_path):
        print(f"No 'logs' subfolder found in {model_path}")
        sys.exit(1)

    print(f"Launching TensorBoard for logs in: {logs_path}")
    subprocess.run(["tensorboard", "--logdir", logs_path])

if __name__ == "__main__":
    main()
