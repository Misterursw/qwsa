# 文件路径: model/QWSA.py
import os
from typing import List
import logging # ### MODIFICATION START ###

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration
from model.segment_anything import build_sam_vit_h

def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float, scale=1000, eps=1e-6):
    inputs = inputs.sigmoid().flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    return loss.sum() / (num_masks + 1e-8)

def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)

class QWSAForCausalLM(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        
        # 确保关键配置存在
        if not hasattr(config, "image_grid_pinpoints") or config.image_grid_pinpoints is None:
            config.image_grid_pinpoints = [[448, 448]]

        # 1. 构建SAM视觉模型
        self.visual_model = build_sam_vit_h()

        # 2. 加载预训练SAM权重
        vision_pretrained = kwargs.pop("vision_pretrained", None)
        if vision_pretrained and os.path.exists(vision_pretrained):
            sam_checkpoint = torch.load(vision_pretrained, map_location="cpu")
            self.visual_model.load_state_dict(sam_checkpoint, strict=False)
        
        # 3. 冻结参数
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if kwargs.get("train_mask_decoder", False):
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # 4. 构建文本特征投影层
        in_dim = config.hidden_size
        out_dim = kwargs.get("out_dim", 256)
        text_fc = [
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        # 5. 保存其他参数
        self.seg_token_idx = kwargs.pop("seg_token_idx", -1)
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", 1.0)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", 0.5)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", 2.0)

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings = self.visual_model.image_encoder(pixel_values)
        return image_embeddings

    def forward(self, **kwargs):
        # ------------------- 核心逻辑修改 -------------------
        # 不再有独立的evaluate_segmentation，所有逻辑都在forward中
        # 通过 self.training 标志来区分是训练还是评估
        
        model_kwargs = {k: v for k, v in kwargs.items() if k in [
            "pixel_values", "input_ids", "labels", "attention_mask", "image_grid_thw"
        ]}
        
        images = kwargs['images']
        offset = kwargs['offset']
        resize_list = kwargs['resize_list']
        
        # 执行一次前向传播，获取 hidden_states
        model_kwargs['output_hidden_states'] = True
        outputs = super().forward(**model_kwargs)
        hidden_states = outputs.hidden_states[-1]
        input_ids = model_kwargs.get("input_ids")

        # --- 通用逻辑: 提取分割嵌入 ---
        seg_token_mask = (input_ids == self.seg_token_idx)

        # ### MODIFICATION START ###
        # 在推理时添加诊断日志
        if not self.training:
            logging.info(f"[QWSA.forward-DEBUG] Is training: {self.training}")
            logging.info(f"[QWSA.forward-DEBUG] seg_token_idx: {self.seg_token_idx}")
            logging.info(f"[QWSA.forward-DEBUG] input_ids shape: {input_ids.shape}")
            logging.info(f"[QWSA.forward-DEBUG] Number of SEG tokens found in input_ids: {torch.sum(input_ids == self.seg_token_idx)}")
        # ### MODIFICATION END ###

        if hidden_states.shape[1] > seg_token_mask.shape[1]:
             diff = hidden_states.shape[1] - seg_token_mask.shape[1]
             padding = torch.zeros((seg_token_mask.shape[0], diff), dtype=torch.bool, device=seg_token_mask.device)
             seg_token_mask = torch.cat([padding, seg_token_mask], dim=1)
        
        pred_embeddings = self.text_hidden_fcs[0](hidden_states[seg_token_mask])

        # ### MODIFICATION START ###
        # 在推理时添加诊断日志
        if not self.training:
            logging.info(f"[QWSA.forward-DEBUG] pred_embeddings shape: {pred_embeddings.shape}")
        # ### MODIFICATION END ###

        seg_token_counts = seg_token_mask.int().sum(-1)
        seg_token_offset = torch.cat([torch.zeros(1, device=seg_token_counts.device).long(), seg_token_counts.cumsum(-1)])
        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i+1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_
        
        # --- 通用逻辑: 生成预测掩码 ---
        image_embeddings_sam = self.get_visual_embs(images)
        pred_masks = []
        for i in range(len(pred_embeddings)):
            if pred_embeddings[i].shape[0] == 0: continue
            sparse_embeddings, dense_embeddings = self.visual_model.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, _ = self.visual_model.mask_decoder(
                image_embeddings=image_embeddings_sam[i].unsqueeze(0),
                image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings, multimask_output=False,
            )
            # 在评估时也需要真实标签的尺寸来还原掩码
            original_size = kwargs['label_list'][i].shape
            pred_mask = self.visual_model.postprocess_masks(
                low_res_masks, input_size=resize_list[i], original_size=original_size,
            )
            pred_masks.append(pred_mask[:, 0])

        # --- 根据模型状态（训练/评估）返回不同内容 ---
        if self.training:
            # 训练模式：计算并返回损失
            ce_loss = outputs.loss * self.ce_loss_weight
            device = ce_loss.device
            mask_bce_loss = torch.tensor(0.0, device=device)
            mask_dice_loss = torch.tensor(0.0, device=device)
            num_masks = 0
            
            masks_list = kwargs['masks_list']
            for batch_idx in range(len(pred_masks)):
                gt_mask = masks_list[batch_idx]; pred_mask = pred_masks[batch_idx]
                if gt_mask.shape[0] == 0: continue
                num_masks += gt_mask.shape[0]
                mask_bce_loss += (sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0])
                mask_dice_loss += (dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0])

            mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
            mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
            mask_loss = mask_bce_loss + mask_dice_loss

            # ### MODIFICATION START ###
            # 在训练时，额外返回预测的掩码和真实掩码
            return {
                "loss": ce_loss + mask_loss, "ce_loss": ce_loss,
                "mask_bce_loss": mask_bce_loss, "mask_dice_loss": mask_dice_loss, "mask_loss": mask_loss,
                "pred_masks_train": pred_masks,  # 返回预测掩码
                "gt_masks_train": masks_list     # 返回真实掩码
            }
            # ### MODIFICATION END ###

        else:
            # 评估模式：直接返回预测的掩码
            return {"pred_masks": pred_masks}