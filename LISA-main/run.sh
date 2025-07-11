#!/bin/bash

# =========================================================================
#                           运行配置区
# =========================================================================

# --- 1. GPU 使用配置 ---
# 请在此处修改您希望使用的GPU ID，用逗号隔开。
# 示例:
#   - 使用 GPU 0 和 GPU 2: GPUS_TO_USE="0,2"
#   - 只使用 GPU 3:       GPUS_TO_USE="3"
GPUS_TO_USE="0,1,2,3"


# --- 2. 日志文件设置 ---
LOG_DIR="/home/ubuntu/LISA-main/log"


# --- 3. 训练参数设置 ---
MODEL_PATH="/data/lzl/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
DATASET_DIR="/home/ubuntu/liuzelin/dataset"
VISION_PRETRAINED="/home/ubuntu/LISA-main/sam_vit_h_4b8939.pth"
EXP_NAME="qwsa-3b"

# --- 4. Conda 环境配置 ---
DEEPSPEED_PATH="/home/ubuntu/miniconda3/envs/lisa/bin/deepspeed"


# =========================================================================
#                           脚本执行区 (一般无需修改)
# =========================================================================

# --- 步骤 1: 检查和设置 ---
echo "[状态] 正在检查 DeepSpeed 路径并准备日志文件..."
if [ ! -f "$DEEPSPEED_PATH" ]; then
    echo "[错误] DeepSpeed 可执行文件未在指定路径找到: $DEEPSPEED_PATH"
    echo "[提示] 请检查您的 Conda 环境名称和路径是否正确。"
    exit 1
fi
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${EXP_NAME}_$(date +%m%d%H%M).log"


# --- 步骤 2: 构造 DeepSpeed --include 参数 ---
# 这是最关键的修复：我们不再使用 CUDA_VISIBLE_DEVICES
# 而是为 deepspeed 命令构造 --include 参数, e.g., --include localhost:0,1
DEEPSPEED_INCLUDE_STR="localhost:$GPUS_TO_USE"


# --- 步骤 3: 打印运行配置 ---
echo "================================================="
echo "         LISA Qwen 训练脚本已启动          "
echo "================================================="
echo "[配置] 将使用 DeepSpeed: $DEEPSPEED_PATH"
echo "[配置] DeepSpeed 将包含 GPUs: $DEEPSPEED_INCLUDE_STR"
echo "[配置] 模型路径: $MODEL_PATH"
echo "[配置] 日志文件将保存至: $LOG_FILE"
echo "-------------------------------------------------"
echo "[提示] 这是一个后台任务，启动后您可以关闭终端。"
echo "[提示] 请使用以下命令实时查看训练日志:"
echo "         tail -f $LOG_FILE"
echo "================================================="


# --- 步骤 4: 启动训练 ---
# 最终修复: 使用 --include 标志来明确指定GPU，而不是依赖环境变量
nohup $DEEPSPEED_PATH --include "$DEEPSPEED_INCLUDE_STR" --master_port=24998 train_qs.py \
  --version="$MODEL_PATH" \
  --dataset_dir="$DATASET_DIR" \
  --vision_pretrained="$VISION_PRETRAINED" \
  --dataset="reason_seg" \
  --sample_rates="1" \
  --exp_name="$EXP_NAME" \
  --conv_type="qwen_vl" > "$LOG_FILE" 2>&1 &

# --- 步骤 5: 启动确认 ---
echo "[状态] 正在等待 DeepSpeed 进程启动..."
sleep 5 # 等待几秒钟让进程有时间初始化
if ps -ef | grep "train_qs.py" | grep -v grep > /dev/null
then
    echo "[成功] DeepSpeed 进程已在后台成功启动。"
    echo "[提示] 您现在可以安全地关闭此终端。"
else
    echo "[失败] DeepSpeed 进程可能未能成功启动，请立即检查日志文件: $LOG_FILE"
fi
echo "================================================="

