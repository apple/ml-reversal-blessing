from huggingface_hub import hf_hub_download
import joblib

REPO_ID = "apple/ml-reversal-blessing"


for directory in [
    "ckpt/multiplication_simulation_l2r",
    "ckpt/multiplication_simulation_r2l",
    "ckpt/reverse_multiplication_simulation_l2r",
    "ckpt/reverse_multiplication_simulation_r2l",
]:
    print(f"Downloading {directory} from {REPO_ID}...")
    hf_hub_download(
        repo_id=REPO_ID, filename=f"{directory}/checkpoint.pt", local_dir="."
    )
