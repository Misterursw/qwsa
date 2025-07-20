#!/bin/bash

# 你的微调后的 QWSA 模型检查点路径
MODEL_CHECKPOINT="/home/ubuntu/.cache/huggingface/hub/models--omlab--Qwen2.5VL-3B-VLM-R1-REC-500steps/snapshots/b9a4c5aa915faddc7e129c3f61d630ceef73a911"

# SAM 预训练权重路径
SAM_WEIGHTS="/data/lzl/ckpt/medsam_vit_b.pth"

# 测试图片路径
TEST_IMAGE="/data/lzl/data/dataset/reason_seg/ReasonSeg/train/3588328_892066223b_o.jpg"

TEST_PROMPT="You are a professional visual segmentation assistant. Your expertise is identifying objects in images and triggering precise segmentation.

TASK REQUIREMENTS:
1. Analyze the image thoroughly
2. Identify the most prominent or relevant object for segmentation
3. Provide semantic context about the object
4. Activate segmentation using the required trigger word

RESPONSE FORMAT:
<think>
Image contains: [list key objects you observe]
Primary segmentation target: [chosen object with justification]
Object characteristics: [size, color, position, type, etc.]
Segmentation strategy: [why this object is suitable for segmentation]
</think>

<answer>
I identify a [detailed object description] in the image. This [object category] shows [key visual features]. 
To perform accurate segmentation of this [object], I activate the segmentation process: seg
</answer>

CRITICAL SUCCESS FACTOR: The word "seg" is essential and must appear in your answer to trigger the segmentation algorithm."
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