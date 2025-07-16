# 文件路径: model/QWSA.py
import os
from typing import List
import logging # ### MODIFICATION START ###
from model.segment_anything import build_sam_vit_b # 确保导入
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration
from model.segment_anything import build_sam_vit_b
from model.segment_anything.utils.transforms import ResizeLongestSide

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
def preprocess_image_for_sam(image: torch.Tensor, image_size: int = 1024) -> torch.Tensor:
    """
    为SAM模型预处理图像：调整大小、归一化、填充。
    假设输入为 [batch_size, channels, height, width] 的张量。
    """
    # 反归一化 pixel_values 到 [0, 255]
    image = image * torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image.device).view(1, -1, 1, 1) + \
            torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image.device).view(1, -1, 1, 1)
    image = image * 255  # 恢复到 [0, 255]
    
    pixel_mean = torch.tensor([0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255], device=image.device).view(-1, 1, 1)
    pixel_std = torch.tensor([0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255], device=image.device).view(-1, 1, 1)
    
    transform = ResizeLongestSide(image_size)
    batch_size = image.shape[0]
    image_sam_list = []
    
    for i in range(batch_size):
        img_np = image[i].permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]
        img_resized = transform.apply_image(img_np)  # 调整大小
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).to(image.device)  # [H, W, C] -> [C, H, W]
        image_sam_list.append(img_tensor)
    
    image_sam = torch.stack(image_sam_list, dim=0)  # [batch_size, C, H, W]
    image_sam = (image_sam - pixel_mean) / pixel_std  # 归一化
    
    # 填充为方形
    _, _, h, w = image_sam.shape
    padh = image_size - h
    padw = image_size - w
    image_sam = F.pad(image_sam, (0, padw, 0, padh))
    
    return image_sam

class QWSAForCausalLM(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        
        if not hasattr(config, "image_grid_pinpoints") or config.image_grid_pinpoints is None:
            config.image_grid_pinpoints = [[448, 448]]

        self.visual_model = build_sam_vit_b()
        vision_pretrained = kwargs.pop("vision_pretrained", None)
        if vision_pretrained and os.path.exists(vision_pretrained):
            sam_checkpoint = torch.load(vision_pretrained, map_location="cpu")
            # 使用 assign=True 强制赋值 meta 参数
            self.visual_model.load_state_dict(sam_checkpoint, strict=False, assign=True)
            logging.info("SAM 权重加载完成")
            logging.info(f"SAM pos_embed 示例值: {self.visual_model.image_encoder.pos_embed[:5]}")
        
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if kwargs.get("train_mask_decoder", False):
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

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

        self.seg_token_idx = kwargs.pop("seg_token_idx", -1)
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", 1.0)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", 0.5)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", 2.0)
        self.image_size = kwargs.pop("image_size", 1024)
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
            with torch.no_grad():
                image_embeddings = self.visual_model.image_encoder(pixel_values)
            return image_embeddings
    def forward(self, **kwargs):
        model_kwargs = {k: v for k, v in kwargs.items() if k in [
            "pixel_values", "input_ids", "labels", "attention_mask", "image_grid_thw"
        ]}
        
        pixel_values = kwargs.get('pixel_values')
        logging.info(f"pixel_values in forward形状: {pixel_values.shape}")
        logging.info(f"pixel_values in forward范围: min={pixel_values.min()}, max={pixel_values.max()}")
        if pixel_values is None:
            raise ValueError("Expected 'pixel_values' in kwargs for image processing")
        
        # 调试：检查 pixel_values 范围和形状
        if not self.training:
            logging.info(f"pixel_values 形状: {pixel_values.shape}")
            logging.info(f"pixel_values 范围: min={pixel_values.min()}, max={pixel_values.max()}")
        
        images = preprocess_image_for_sam(pixel_values, self.image_size).to(pixel_values.dtype)
        logging.info(f"SAM images 形状: {images.shape}")
        logging.info(f"SAM images 范围: min={images.min()}, max={images.max()}")        
        # 调试：比较 Qwen2.5-VL 和 SAM 的图像嵌入
        if not self.training:
            qwen_visual_emb = self.visual(pixel_values)
            sam_visual_emb = self.get_visual_embs(images)
            logging.info(f"Qwen visual embedding norm: {qwen_visual_emb.norm()}")
            logging.info(f"SAM visual embedding norm: {sam_visual_emb.norm()}")
        
        offset = kwargs.get('offset', [0])
        resize_list = kwargs.get('resize_list', [(images.shape[-2], images.shape[-1])] * images.shape[0])
        
        model_kwargs['output_hidden_states'] = True
        outputs = super().forward(**model_kwargs)
        hidden_states = outputs.hidden_states[-1]
        input_ids = model_kwargs.get("input_ids")

        seg_token_mask = (input_ids == self.seg_token_idx)
        if not self.training:
            logging.info(f"[QWSA.forward-DEBUG] Is training: {self.training}")
            logging.info(f"[QWSA.forward-DEBUG] seg_token_idx: {self.seg_token_idx}")
            logging.info(f"[QWSA.forward-DEBUG] input_ids shape: {input_ids.shape}")
            logging.info(f"[QWSA.forward-DEBUG] Number of SEG tokens found: {torch.sum(seg_token_mask)}")

        if hidden_states.shape[1] > seg_token_mask.shape[1]:
            diff = hidden_states.shape[1] - seg_token_mask.shape[1]
            padding = torch.zeros((seg_token_mask.shape[0], diff), dtype=torch.bool, device=seg_token_mask.device)
            seg_token_mask = torch.cat([padding, seg_token_mask], dim=1)
        
        pred_embeddings = self.text_hidden_fcs[0](hidden_states[seg_token_mask])

        if not self.training:
            logging.info(f"[QWSA.forward-DEBUG] pred_embeddings shape: {pred_embeddings.shape}")

        seg_token_counts = seg_token_mask.int().sum(-1)
        seg_token_offset = torch.cat([torch.zeros(1, device=seg_token_counts.device).long(), seg_token_counts.cumsum(-1)])
        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i+1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_
        logging.info(f"[SEG] token 嵌入范数: {pred_embeddings.norm(dim=-1)}")
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
            original_size = kwargs.get('label_list', [(images.shape[-2], images.shape[-1])] * len(pred_embeddings))[i]
            pred_mask = self.visual_model.postprocess_masks(
                low_res_masks, input_size=resize_list[i], original_size=original_size,
            )
            pred_masks.append(pred_mask[:, 0])

        if self.training:
            ce_loss = outputs.loss * self.ce_loss_weight
            device = ce_loss.device
            mask_bce_loss = torch.tensor(0.0, device=device)
            mask_dice_loss = torch.tensor(0.0, device=device)
            num_masks = 0
            
            masks_list = kwargs.get('masks_list', [])
            for batch_idx in range(len(pred_masks)):
                gt_mask = masks_list[batch_idx] if masks_list else torch.zeros_like(pred_masks[batch_idx])
                if gt_mask.shape[0] == 0: continue
                num_masks += gt_mask.shape[0]
                mask_bce_loss += (sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0])
                mask_dice_loss += (dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0])

            mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
            mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
            mask_loss = mask_bce_loss + mask_dice_loss

            return {
                "loss": ce_loss + mask_loss, "ce_loss": ce_loss,
                "mask_bce_loss": mask_bce_loss, "mask_dice_loss": mask_dice_loss, "mask_loss": mask_loss,
                "pred_masks_train": pred_masks, "gt_masks_train": masks_list
            }
        else:
            return {"pred_masks": pred_masks}