export SAMPLE_DIR="lora/samples/pic"
export OUTPUT_DIR="lora/lora_ckpt/lora"

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export LORA_RANK=16

accelerate launch lora/train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$SAMPLE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="UAV" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=2e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=200 \
  --lora_rank=$LORA_RANK \
  --seed="0"
