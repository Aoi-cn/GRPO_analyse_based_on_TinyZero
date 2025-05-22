export N_GPUS=1
export BASE_MODEL=/data/zhouminghao/LLaMA-Factory/models/Qwen2.5-0.5B
export DATA_DIR=/data/zhouminghao/TinyZero/data/countdown
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-0.5b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/grpo_tiny_zero.sh