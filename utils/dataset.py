import glob
import os
import random
from PIL import Image

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask

from model.qwen import conversation as conversation_lib
from model.qwen.constants import IGNORE_INDEX
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .refer import REFER
from .utils import ANSWER_LIST  # <--- 在这里添加这一行
from .reason_seg_dataset import ReasonSegDataset


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1, processor=None
):
    """
    数据整理函数，用于处理和打包一个批次的数据。
    增加了对无效样本和空批次的处理。
    """
    # 1. 过滤掉数据集中可能存在的无效样本（在__getitem__中返回None的）
    batch = [item for item in batch if item is not None]
    
    # 2. 如果过滤后整个批次为空，则返回一个结构完整但内容为空的字典
    if not batch:
        # 这个返回结构确保了训练循环在尝试访问这些键时不会出错
        return {
            "images": torch.tensor([], dtype=torch.float32),
            "pixel_values": torch.tensor([], dtype=torch.float32),
            "input_ids": torch.tensor([], dtype=torch.long),
            "labels": torch.tensor([], dtype=torch.long),
            "attention_mask": torch.tensor([], dtype=torch.long),
            "masks_list": [],
            "label_list": [],
            "resize_list": [],
            "offset": torch.LongTensor([]),
            "inference": [],
            "image_paths": [],
            "questions_list": [],
            "sampled_classes_list": [],
            "image_grid_thw": None,
        }

    assert processor is not None, "Qwen requires the processor to be passed to collate_fn"

    # 初始化收集器
    image_path_list, images_list, pil_images, all_messages_structured = [], [], [], []
    masks_list, label_list, resize_list, questions_list, sampled_classes_list = [], [], [], [], []
    offset_list = [0]
    cnt = 0
    inferences = []

    # 3. 从有效批次中收集原始数据
    for (
        image_path, images, pil_image, messages, masks, label, resize,
        questions, sampled_classes, inference
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        pil_images.append(pil_image)
        all_messages_structured.append(messages)
        masks_list.append(masks.float())
        label_list.append(label)
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += 1
        offset_list.append(cnt)
        inferences.append(inference)

    # 4. 使用 Qwen processor 处理文本和图像
    texts_for_processing = [
        processor.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=(not conversation[-1]['content'])
        )
        for conversation in all_messages_structured
    ]
    # ===================== 新增修复逻辑 开始 =====================
    # 为验证/推理样本在prompt末尾添加[SEG] token，以触发掩码预测
    for i in range(len(texts_for_processing)):
        if inferences[i]:  # 检查是否为验证样本 (inference=True)
            original_text = texts_for_processing[i] # ### MODIFICATION START ###
            texts_for_processing[i] += "[SEG]"
            # 添加日志，确认[SEG]已添加
            logging.info(f"[collate_fn-DEBUG] Appended [SEG] for validation. Original text tail: '...{original_text[-50:]}'. New text tail: '...{texts_for_processing[i][-50:]}'") # ### MODIFICATION END ###

    # ===================== 新增修复逻辑 结束 =====================

    inputs = processor(
        text=texts_for_processing,
        images=pil_images,
        return_tensors="pt",
        padding=True
    )

    # 5. 提取所有需要的输出
    input_ids = inputs['input_ids']
    attention_masks = inputs['attention_mask']
    images_clip = inputs['pixel_values']
    image_grid_thw = inputs.get('image_grid_thw')

    # 6. 为 'labels' 创建目标张量，并屏蔽padding部分
    targets = input_ids.clone()
    targets[targets == tokenizer.pad_token_id] = IGNORE_INDEX
    
    # 7. 精确屏蔽掉用户提问部分，只保留助手回答部分的损失
    for i, conversation_text in enumerate(texts_for_processing):
        # Qwen的模板在助手部分前有固定的分隔符
        sep = "<|im_start|>assistant"
        parts = conversation_text.split(sep)
        if len(parts) > 1:
            # 计算需要被屏蔽的token长度
            instruction_len = len(tokenizer(parts[0] + sep, add_special_tokens=False).input_ids)
            targets[i, :instruction_len] = IGNORE_INDEX
    
    # 8. 构建并返回最终的批处理字典
    return_dict = {
        "images": torch.stack(images_list, dim=0),
        "pixel_values": images_clip,
        "input_ids": input_ids,
        "labels": targets,
        "attention_mask": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "inference": inferences[0] if inferences else False, # 处理空列表情况
        # 以下为辅助信息，在模型调用前需过滤
        "image_paths": image_path_list,
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
    }
    
    if image_grid_thw is not None:
        return_dict['image_grid_thw'] = image_grid_thw
    
    return return_dict


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower, # This argument is no longer used but kept for API consistency
        processor,
        val_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.data_type = ""
        self.data_list = []
        # ==================== 修改开始 ====================
        self.val_data = [] # 用于存储从JSON加载的复杂问答数据

        splits = val_dataset.split("|")
        # 强制使用 ReasonSeg 类型的复杂验证集
        # 例如: --val_dataset "ReasonSeg|val_explanatory"
        if len(splits) == 2 and splits[0] == "ReasonSeg":
            ds, split = splits
            # 假设您的复杂验证集JSON位于 .../explanatory/val.json
            val_json_path = os.path.join(self.base_image_dir, "reason_seg", ds, "explanatory", f"{split}.json")
            if os.path.exists(val_json_path):
                print(f"找到复杂的验证集文件: {val_json_path}")
                with open(val_json_path, 'r') as f:
                    self.val_data = json.load(f)
                self.data_type = "reason_seg_explanatory"
            else:
                raise FileNotFoundError(f"未找到指定的复杂验证集文件: {val_json_path}")
        else:
            raise ValueError("验证集配置不正确。请使用类似 'ReasonSeg|val' 的格式，并确保对应的 .../explanatory/val.json 文件存在。")
        # ==================== 修改结束 ====================

    def __len__(self):
        return len(self.val_data)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        while True:
            try:
                # ==================== 修改开始 ====================
                # 直接从加载的 val_data 中获取数据
                item = self.val_data[idx]
                image_name = item["image"]
                json_name = item["json"]
                
                # 构建完整路径
                # 假设图片和json文件在同一个val文件夹下
                image_path = os.path.join(self.base_image_dir, "reason_seg", "ReasonSeg", "val", image_name)
                json_path = os.path.join(self.base_image_dir, "reason_seg", "ReasonSeg", "val", json_name)

                if not os.path.exists(image_path) or not os.path.exists(json_path):
                    print(f"警告: 验证集文件缺失，跳过 {image_path} 或 {json_path}")
                    idx = random.randint(0, len(self.data_list) - 1) # 随机换一个索引重试
                    continue

                # 从JSON获取query和output
                full_prompt_text = item["query"]
                answer_text = item["outputs"] # 这个答案用于构建对话，但在验证时模型会自己生成

                # 从掩码JSON文件获取真实掩码
                img_for_json = cv2.imread(image_path)
                if img_for_json is None:
                    idx = random.randint(0, len(self.data_list) - 1)
                    continue
                
                masks, _, _ = get_mask_from_json(json_path, img_for_json)
                masks = [masks] # 包装成列表

                # --- 统一的数据处理流程 ---
                image_cv = cv2.cvtColor(img_for_json, cv2.COLOR_BGR2RGB)
                # ==================== 修改结束 ====================
                
                pil_image = Image.fromarray(image_cv)
                image_for_sam = self.transform.apply_image(image_cv)
                resize = image_for_sam.shape[:2]
                image_for_sam_tensor = self.preprocess(torch.from_numpy(image_for_sam).permute(2, 0, 1).contiguous())
                
                # 构建对话消息
                # 注意：对于验证，助手的回答内容是空的，因为这是模型需要生成的部分。
                # 但我们仍然需要完整的结构来应用模板。
                messages = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": full_prompt_text}]},
                    {"role": "assistant", "content": ""} 
                ]

                final_masks = np.stack(masks, axis=0)
                final_masks = torch.from_numpy(final_masks)
                label = torch.ones(final_masks.shape[1], final_masks.shape[2]) * self.ignore_label
                
                # 如果所有步骤都成功，返回数据并跳出循环
                return (
                    image_path,
                    image_for_sam_tensor,
                    pil_image,
                    messages,
                    final_masks,
                    label,
                    resize,
                    [full_prompt_text],
                    [], # sents 在这里可以为空，因为我们用的是完整的query
                    True, # inference=True
                )
            
            except Exception as e:
                print(f"警告: 在处理验证集样本索引 {idx} 时发生错误: {e}。正在尝试下一个样本...")
                idx = random.randint(0, len(self.val_data) - 1)


