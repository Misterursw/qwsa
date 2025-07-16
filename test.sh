#!/bin/bash

# 你的微调后的 QWSA 模型检查点路径
MODEL_CHECKPOINT="/home/ubuntu/.cache/huggingface/hub/models--omlab--Qwen2.5VL-3B-VLM-R1-REC-500steps/snapshots/b9a4c5aa915faddc7e129c3f61d630ceef73a911"

# SAM 预训练权重路径
SAM_WEIGHTS="/data/lzl/ckpt/medsam_vit_b.pth"

# 测试图片路径
TEST_IMAGE="/data/lzl/data/dataset/reason_seg/ReasonSeg/train/3588328_892066223b_o.jpg"

TEST_PROMPT="You are an expert visual assistant. Your task is to respond to user queries about an image. 
You must follow a strict format. First, provide a step-by-step reasoning process enclosed in </think> tags. 
Second, provide a concise final answer enclosed in <answer></answer> tags. 
The answer must ALWAYS contain the special token <|extra_100|> immediately after the identified object to trigger the segmentation based on your reasoning.
Example: <think>your reasoning</think> <answer>car<|extra_100|></answer>."
# 输出目录
OUTPUT_DIR="./test_output_final"

echo "--- 开始 QWSA 模型推理测试 ---"

python test.py \
    --model-path "$MODEL_CHECKPOINT" \
    --image-path "$TEST_IMAGE" \
    --prompt "$TEST_PROMPT" \
    --sam-checkpoint "$SAM_WEIGHTS" \
    --out-dir "$OUTPUT_DIR" \
    --precision "bf16"

echo "--- 推理完成 ---"
echo "结果已保存到: $OUTPUT_DIR"