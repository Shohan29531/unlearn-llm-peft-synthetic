# Unlearn LLM with PEFT and Synthetic Data

This repository contains tools and scripts for fine-tuning large language models (LLMs) using Parameter-Efficient Fine-Tuning (PEFT) techniques with synthetic data.
The repo fine-tunes Mistral-7B-v0.1 and generates synthetic samples using Llama-3.3-70B-Instruct model. 

## Prerequisites

- CUDA-compatible GPUs
- [Hugging Face](https://huggingface.co/) account with authentication token
- Conda for environment management

## Installation

```bash
# Clone the repository
git clone https://github.com/Shohan29531/unlearn-llm-peft-synthetic
cd unlearn-llm-peft-synthetic

# Create and activate conda environment
conda create -n lora-env python=3.10 -y
conda activate lora-env

# Install dependencies
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Verify CUDA Installation

```bash
python cuda_test.py
```

## Serving the Base Model

You need to initialize a local LLM server using vLLM. Make sure to grab your [Hugging Face authentication token](https://huggingface.co/settings/tokens) first.

### Multi-GPU Setup (4 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

### Single-GPU Setup

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8000
```

## Synthetic Data Generation

Use the synthetic-data-kit to generate training data:

```bash
# Ingest personal information
synthetic-data-kit -c configs/config.yaml ingest data/txt/personal_info.txt 

# Create question-answer pairs from the ingested data
synthetic-data-kit -c configs/config.yaml create data/output/personal_info.txt --type qa
```

## Shutting Down the Server

After data generation is complete:

1. Shut down the LLama-70B model server using `Ctrl + C` in the terminal
2. Close that terminal
3. Verify the server has been shut down by checking GPU memory usage:

```bash
nvidia-smi
```

## Data Preparation

Extract training data from the synthetic dataset:

```bash
python extract_training_data.py
```

## Fine-Tuning and Verification

The default setup fine-tunes the `mistralai/Mistral-7B-v0.1` model. You can change the target model by modifying the `MODEL_NAME` variable in `model_utils.py`.

```bash
# Run fine-tuning
python train_secret.py

# Verify the fine-tuned model
python verify_secret.py
```

## Notes

- Ensure the LLama-70B server is shut down before running fine-tuning to avoid out-of-memory issues
- The fine-tuning process uses PEFT techniques to efficiently update only a small subset of the model parameters
- I thank the authors of [Meta synthetic-data-kit](https://github.com/meta-llama/synthetic-data-kit).


