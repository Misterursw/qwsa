import argparse
import os
import shutil
import sys
import time
import logging
from functools import partial

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig

from model.QWSA import QWSAForCausalLM
from model.qwen import conversation as conversation_lib
from utils.dataset import ValDataset, collate_fn, ReasonSegDataset
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU, IGNORE_INDEX)

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler("train_qs.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Qwen Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--version", default="Qwen/Qwen2.5-VL-3B-Instruct", type=str)
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"], help="precision for inference")
    parser.add_argument("--image_size", default=256, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--dataset", default="reason_seg", type=str)
    parser.add_argument("--sample_rates", default="1", type=str)
    parser.add_argument("--sem_seg_data", default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary", type=str)
    parser.add_argument("--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str)
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="qwsa-3b", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=1, type=int)
    parser.add_argument("--batch_size", default=8, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=1.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--conv_type", default="qwen_vl", type=str, choices=["llava_v1", "llava_llama_2", "qwen_vl"])
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    args.distributed = world_size > 1

    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    logging.info("==========================================================")
    logging.info("               LISA Qwen 训练脚本启动               ")
    logging.info("==========================================================")
    logging.info(f"所有日志将保存在: {args.log_dir}")
    logging.info(f"检测到 world size = {world_size}。分布式训练: {'启用' if args.distributed else '禁用'}")

    logging.info(f"[步骤 1/8] 正在从 '{args.version}' 加载配置...")
    config = AutoConfig.from_pretrained(args.version, trust_remote_code=True, local_files_only=True)
    config.image_grid_pinpoints = [[448, 448]]
    logging.info("配置加载并修正完毕。")

    logging.info("[步骤 2/8] 准备使用 DeepSpeed Stage 3 加载模型...")
    torch_dtype = torch.bfloat16 if args.precision == "bf16" else (torch.half if args.precision == "fp16" else torch.float32)
    model = QWSAForCausalLM.from_pretrained(
        args.version, config=config, torch_dtype=torch_dtype, local_files_only=True, trust_remote_code=True,
        seg_token_idx=0, vision_pretrained=args.vision_pretrained, ce_loss_weight=args.ce_loss_weight,
        dice_loss_weight=args.dice_loss_weight, bce_loss_weight=args.bce_loss_weight,
        train_mask_decoder=args.train_mask_decoder, out_dim=args.out_dim)
    logging.info("模型加载完毕。")

    logging.info(f"[步骤 3/8] 正在加载 Tokenizer 和 Processor...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, cache_dir=None, model_max_length=args.model_max_length, padding_side="right",
        use_fast=False, trust_remote_code=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token 
    # ==================== 修改开始 ====================
    # 定义所有你需要的特殊token
    # 我们不仅需要[SEG]，还需要<think>, </think>, <answer>, </answer>
    new_special_tokens = {
        "additional_special_tokens": [
            "[SEG]", 
            "<think>", 
            "</think>", 
            "<answer>", 
            "</answer>"
        ]
    }

    # 将新token添加到分词器中
    num_added_tokens = tokenizer.add_special_tokens(new_special_tokens)

    if num_added_tokens > 0:
        # 如果成功添加了新token，必须调整模型词嵌入层的大小以匹配
        model.resize_token_embeddings(len(tokenizer))
        
        # 初始化新token的词向量（非常重要！）
        # 你可以简单地使用现有词向量的平均值来初始化它们
        with torch.no_grad():
            input_embeds = model.get_input_embeddings().weight.data
            output_embeds = model.get_output_embeddings().weight.data
            
            avg_embed = input_embeds[:-num_added_tokens].mean(dim=0, keepdim=True)
            input_embeds[-num_added_tokens:] = avg_embed
            output_embeds[-num_added_tokens:] = avg_embed

    # 找到 [SEG] token的ID并保存，这部分逻辑保持不变
    args.seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")
    model.seg_token_idx = args.seg_token_idx

    # ==================== 修改结束 ====================
    processor = transformers.AutoProcessor.from_pretrained(
        args.version, trust_remote_code=True, use_fast=True, local_files_only=True)
    logging.info("Tokenizer 和 Processor 加载完毕。")
    
    logging.info(f"[步骤 4/8] 正在调整词嵌入大小以适应新 Token...")
    model.resize_token_embeddings(len(tokenizer))
    model.config.vocab_size = len(tokenizer)
    logging.info(f"词表大小调整为: {len(tokenizer)}")

    if args.lora_r > 0:
        logging.info(f"[步骤 5/8] 正在配置 LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
        # ... LoRA 配置逻辑 ...
    else:
        logging.info("[步骤 5/8] 未启用 LoRA。")

    if args.gradient_checkpointing:
        logging.info("启用梯度检查点 (Gradient Checkpointing)。")
        model.gradient_checkpointing_enable()
    
    logging.info(f"[步骤 6/8] 正在创建和加载数据集...")
    train_dataset = ReasonSegDataset(
        base_image_dir=args.dataset_dir, tokenizer=tokenizer, vision_tower=None, processor=processor,
        samples_per_epoch=(args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size),
        precision=args.precision, image_size=args.image_size, num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val, reason_seg_data=args.reason_seg_data, explanatory=args.explanatory,)
    val_dataset = None
    if not args.no_eval:
        val_dataset = ValDataset(args.dataset_dir, tokenizer, None, processor, args.val_dataset, image_size=args.image_size)
    logging.info("数据集加载完毕。")

    logging.info(f"[步骤 7/8] 正在配置和初始化 DeepSpeed...")
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {"type": "AdamW", "params": {"lr": args.lr, "weight_decay": 0.0, "betas": (args.beta1, args.beta2)}},
        "scheduler": {"type": "WarmupDecayLR", "params": {"total_num_steps": args.epochs * args.steps_per_epoch, "warmup_min_lr": 0, "warmup_max_lr": args.lr, "warmup_num_steps": 100, "warmup_type": "linear"}},
        "fp16": {"enabled": args.precision == "fp16"}, "bf16": {"enabled": args.precision == "bf16"},
        "gradient_clipping": 1.0,
        "zero_optimization": {"stage": 3, "offload_optimizer": {"device": "cpu", "pin_memory": True},
                              "offload_param": {"device": "cpu", "pin_memory": True}, "contiguous_gradients": True,
                              "overlap_comm": True, "reduce_scatter": True, "reduce_bucket_size": 5e8,
                              "allgather_bucket_size": 5e8},
    }
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model, model_parameters=[p for p in model.parameters() if p.requires_grad],
        training_data=train_dataset, config=ds_config,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, conv_type=args.conv_type,
                           use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank, processor=processor),)
    logging.info("DeepSpeed 初始化成功。")

    logging.info(f"[步骤 8/8] 检查是否需要恢复训练...")
    # ... 恢复训练逻辑 ...

    val_loader = None
    if val_dataset is not None:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=False, sampler=val_sampler,
                                                 collate_fn=partial(collate_fn, tokenizer=tokenizer,
                                                                    conv_type=args.conv_type,
                                                                    use_mm_start_end=args.use_mm_start_end,
                                                                    local_rank=args.local_rank, processor=processor),)
    best_score = 0.0
    logging.info("==========================================================")
    logging.info("               所有准备工作完成，开始训练               ")
    logging.info("==========================================================")
    
    train_iter = iter(train_loader)
    for epoch in range(args.start_epoch, args.epochs):
        train_iter = train(train_loader, model_engine, epoch, scheduler, writer, train_iter, args)
        
        if not args.no_eval and val_loader is not None:
            giou, ciou = validate(val_loader, model_engine, epoch, writer, args)
            is_best = giou > best_score
            best_score = max(giou, best_score)
            
            if args.local_rank == 0:
                save_dir_epoch = os.path.join(args.log_dir, f"checkpoint_epoch_{epoch+1}")
                logging.info(f"正在保存 Epoch {epoch+1} 的检查点到: {save_dir_epoch}")
                model_engine.save_checkpoint(save_dir_epoch)
                logging.info(f"Epoch {epoch+1} 的检查点保存完毕。")

                if is_best:
                    save_dir_best = os.path.join(args.log_dir, "checkpoint_best")
                    logging.info(f"*** 新的最佳模型！gIoU: {best_score:.4f}。正在保存到: {save_dir_best} ***")
                    if os.path.exists(save_dir_best):
                        shutil.rmtree(save_dir_best)
                    model_engine.save_checkpoint(save_dir_best)
                    logging.info("最佳模型检查点保存完毕。")


def train(train_loader, model, epoch, scheduler, writer, train_iter, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [batch_time, losses, ce_losses, mask_losses, mask_bce_losses, mask_dice_losses],
        prefix="Epoch: [{}]".format(epoch + 1),
    )
    
    model.train()
    end = time.time()
    
    logging.info(f"--- 开始 Epoch {epoch+1}/{args.epochs} ---")
    
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)
            
            if not input_dict or not input_dict.get("images").numel():
                logging.warning("警告: 跳过一个空的或无效的批次。")
                continue

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)
            
            model_inputs = {k: v for k, v in input_dict.items() if k in [
                "images", "pixel_values", "input_ids", "labels", "attention_mask", 
                "offset", "masks_list", "label_list", "resize_list", "inference", "image_grid_thw"
            ] and v is not None}
            
            if args.precision == "fp16":
                model_inputs["images"] = model_inputs["images"].half()
                if "pixel_values" in model_inputs:
                    model_inputs["pixel_values"] = model_inputs["pixel_values"].half()
            elif args.precision == "bf16":
                model_inputs["images"] = model_inputs["images"].bfloat16()
                if "pixel_values" in model_inputs:
                    model_inputs["pixel_values"] = model_inputs["pixel_values"].bfloat16()
            
            output_dict = model(**model_inputs)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]

            losses.update(loss.item(), model_inputs["images"].size(0))
            ce_losses.update(ce_loss.item(), model_inputs["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), model_inputs["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), model_inputs["images"].size(0))
            mask_losses.update(mask_loss.item(), model_inputs["images"].size(0))
            model.backward(loss)
            model.step()
        
        batch_time.update(time.time() - end)
        end = time.time()

        if (global_step + 1) % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                lr = scheduler.get_last_lr()[0]
                writer.add_scalar("train/loss", losses.avg, epoch * args.steps_per_epoch + global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, epoch * args.steps_per_epoch + global_step)
                writer.add_scalar("train/mask_bce_loss", mask_bce_losses.avg, epoch * args.steps_per_epoch + global_step)
                writer.add_scalar("train/mask_dice_loss", mask_dice_losses.avg, epoch * args.steps_per_epoch + global_step)
                writer.add_scalar("train/mask_loss", mask_losses.avg, epoch * args.steps_per_epoch + global_step)
                writer.add_scalar("train/lr", lr, epoch * args.steps_per_epoch + global_step)
                logging.info(f"Epoch [{epoch+1}] Step [{global_step+1}/{args.steps_per_epoch}]: 模型权重已更新。当前Loss: {losses.val:.4f} (Avg: {losses.avg:.4f})")

            batch_time.reset(); data_time.reset(); losses.reset(); ce_losses.reset(); mask_bce_losses.reset(); mask_dice_losses.reset()
            
    logging.info(f"--- Epoch {epoch+1} 训练完成 ---")
    return train_iter


def validate(val_loader, model_engine, epoch, writer, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":.0f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":.4f", Summary.AVERAGE)
    
    model_engine.eval()
    
    if args.local_rank == 0:
        logging.info(f"--- Epoch {epoch+1} 验证开始 ---")

    for input_dict in tqdm.tqdm(val_loader, disable=(args.local_rank != 0), desc=f"Validating Epoch {epoch+1}"):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        
        model_inputs = {k: v for k, v in input_dict.items() if v is not None}
        
        if args.precision == "fp16":
            model_inputs["images"] = model_inputs["images"].half()
            if "pixel_values" in model_inputs:
                model_inputs["pixel_values"] = model_inputs["pixel_values"].half()
        elif args.precision == "bf16":
            model_inputs["images"] = model_inputs["images"].bfloat16()
            if "pixel_values" in model_inputs:
                model_inputs["pixel_values"] = model_inputs["pixel_values"].bfloat16()

        with torch.no_grad():
            output_dict = model_engine(**model_inputs)
            pred_masks = output_dict["pred_masks"]

        masks_list = input_dict["masks_list"][0].int()
        
        if not pred_masks:
            logging.warning("警告：模型返回了空的掩码列表，跳过此批次。")
            continue
            
        output_list = (pred_masks[0] > 0).int()

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255)
            intersection += intersection_i
            union += union_i
            acc_iou_i = intersection_i / (union_i + 1e-5)
            acc_iou_i[union_i == 0] = 1.0
            acc_iou += acc_iou_i
        
        intersection_meter.update(intersection.cpu().numpy())
        union_meter.update(union.cpu().numpy())
        acc_iou_meter.update(acc_iou.cpu().numpy() / masks_list.shape[0], n=1)

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1] if hasattr(iou_class, '__len__') and len(iou_class) > 1 else 0.0
    giou = acc_iou_meter.avg[1] if hasattr(acc_iou_meter.avg, '__len__') and len(acc_iou_meter.avg) > 1 else (acc_iou_meter.avg if not hasattr(acc_iou_meter.avg, '__len__') else 0.0)

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch + 1)
        writer.add_scalar("val/ciou", ciou, epoch + 1)
        logging.info(f"--- Epoch {epoch+1} 验证完成: gIoU: {giou:.4f}, cIoU: {ciou:.4f} ---")
    
    return giou, ciou


if __name__ == "__main__":
    main(sys.argv[1:])
