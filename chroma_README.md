# Chroma Training Script

This training script is designed for fine-tuning the Chroma model, an 8.9B parameter model based on FLUX.1-schnell with architectural modifications. The script incorporates the key improvements outlined in the Chroma model's documentation:

1. **Reduced Model Size (8.9B)**: The original 12B FLUX model is reduced to 8.9B by replacing the large modulation layer with a more efficient FFN.
2. **MMDiT Masking**: Masks T5 padding tokens to enhance fidelity and increase training stability.
3. **Custom Timestep Distribution**: Uses a custom distribution to prevent loss spikes during training.
4. **Minibatch Optimal Transport**: Implements advanced transport problem math to accelerate training by reducing path ambiguity.

## Prerequisites

Before running the training script, you'll need the following components:

1. The base Chroma model checkpoint for fine-tuning: `/home/kade/toolkit/diffusion/comfy/models/checkpoints/chroma-unlocked-v18.safetensors`
2. T5-XXL text encoder (choose one):
   - [T5 XXL FP16](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors)
   - [T5 XXL FP8](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors)
3. [CLIP-L](https://huggingface.co/openai/clip-vit-large-patch14/tree/main)
4. [FLUX VAE](https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors)
5. A dataset with captions or metadata for training

## Installation

1. Make sure you have all the dependencies installed:

```bash
pip install torch accelerate transformers diffusers safetensors bitsandbytes
```

2. Clone the repository if you haven't already:

```bash
git clone https://github.com/your-repo/sd-scripts.git
cd sd-scripts
```

## Usage

### Basic Command

Here's a basic command to start training with the Chroma model:

```bash
python chroma_train.py \
  --pretrained_model_name_or_path="/home/kade/toolkit/diffusion/comfy/models/checkpoints/chroma-unlocked-v18.safetensors" \
  --clip_l="path/to/clip_l.safetensors" \
  --t5xxl="path/to/t5xxl_fp16.safetensors" \
  --ae="path/to/flux_vae.safetensors" \
  --train_data_dir="path/to/training/images" \
  --in_json="path/to/metadata.json" \
  --output_dir="path/to/output" \
  --learning_rate=1e-5 \
  --max_train_steps=5000 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  --cache_latents \
  --cache_text_encoder_outputs \
  --apply_t5_attn_mask
```

### Advanced Options

The script includes several Chroma-specific options:

- `--use_mmdit_masking`: Use MMDiT masking for padding tokens (enabled by default)
- `--use_custom_timestep`: Use custom timestep distribution (enabled by default)
- `--use_optimal_transport`: Use minibatch optimal transport (enabled by default)

Other useful options:

- `--blocks_to_swap`: Number of blocks to swap between CPU and GPU to reduce memory usage
- `--save_every_n_steps`: Save a checkpoint every N steps
- `--sample_every_n_steps`: Generate sample images every N steps
- `--t5xxl_max_token_length`: Max token length for T5-XXL tokenizer

### Sample Training Command with 8bit Optimization

For systems with limited VRAM, you can use 8-bit optimizations:

```bash
python chroma_train.py \
  --pretrained_model_name_or_path="/home/kade/toolkit/diffusion/comfy/models/checkpoints/chroma-unlocked-v18.safetensors" \
  --clip_l="path/to/clip_l.safetensors" \
  --t5xxl="path/to/t5xxl_fp8_e4m3fn.safetensors" \
  --ae="path/to/flux_vae.safetensors" \
  --train_data_dir="path/to/training/images" \
  --in_json="path/to/metadata.json" \
  --output_dir="path/to/output" \
  --learning_rate=1e-5 \
  --max_train_steps=5000 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  --blocks_to_swap=18 \
  --cache_latents \
  --cache_text_encoder_outputs \
  --apply_t5_attn_mask \
  --optimizer_type="adamw8bit"
```

### Training with DreamBooth Method

You can also train using the DreamBooth method:

```bash
python chroma_train.py \
  --pretrained_model_name_or_path="/home/kade/toolkit/diffusion/comfy/models/checkpoints/chroma-unlocked-v18.safetensors" \
  --clip_l="path/to/clip_l.safetensors" \
  --t5xxl="path/to/t5xxl_fp16.safetensors" \
  --ae="path/to/flux_vae.safetensors" \
  --train_data_dir="path/to/instance/images" \
  --reg_data_dir="path/to/class/images" \
  --output_dir="path/to/output" \
  --learning_rate=1e-6 \
  --max_train_steps=800 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  --cache_latents \
  --cache_text_encoder_outputs \
  --apply_t5_attn_mask
```

## Notes

- The Chroma model is still being developed, so expect updates and improvements to this training script.
- For best results, use a good quality dataset with clean, descriptive captions.
- Training may require a significant amount of VRAM. Using techniques like gradient checkpointing and block swapping can help reduce memory requirements.
- The model has a reduced size of 8.9B parameters (vs original 12B) due to optimized modulation layers.
