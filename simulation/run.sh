#!/bin/bash
model_type=$1
direction=$2
# Check if HF_TOKEN environment variable is set
if [ -z "${HF_TOKEN}" ]; then
    echo "Error: HF_TOKEN environment variable is not set."
    echo "Please set it with: export HF_TOKEN=your_huggingface_token"
    exit 1
fi

pip install --upgrade torch torchvision torchaudio

# Install dependencies: lm-evaluation-harness
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
pushd lm-evaluation-harness; git checkout v0.4.3; popd
mkdir lm-evaluation-harness/lm_eval/tasks/multiplication
cp simulation/multiply.yaml lm-evaluation-harness/lm_eval/tasks/multiplication/multiply.yaml
python simulation/generate_yaml.py
pip install -e lm-evaluation-harness

# Install dependencies: torchtitan (for downloading tokenizer)
git clone https://github.com/pytorch/torchtitan.git
pushd torchtitan
python scripts/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3-8B --tokenizer_path "original" --hf_token=${HF_TOKEN}
popd

python simulation/generate_json.py --model-type ${model_type}
python -m simulation.run_evaluate --model-type ${model_type} --direction ${direction}
