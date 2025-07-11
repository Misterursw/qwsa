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

        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            self.data_list = glob.glob(os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg"))
            self.data_type = "reason_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            refer_api = REFER(self.base_image_dir, ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            self.data_list = refer_api.loadImgs(image_ids=images_ids_val)
            
            self.refer_api = refer_api
            self.data_type = "refer_seg"

    def __len__(self):
        return len(self.data_list)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        # 修复：使用 while True 循环来确保总能返回一个有效样本
        while True:
            try:
                if self.data_type == "refer_seg":
                    image_info = self.data_list[idx]
                    image_path = image_info["file_name"]
                    if not os.path.exists(image_path):
                         image_path = os.path.join(self.base_image_dir, "images/mscoco/images/train2014", image_info["file_name"].split("/")[-1])
                    
                    refs = self.refer_api.imgToRefs[image_info["id"]]
                    sents = [sent['sent'] for ref in refs for sent in ref['sentences']]
                    if not sents:
                        # 如果没有句子，跳过此样本
                        idx = random.randint(0, len(self.data_list) - 1)
                        continue

                    full_prompt_text = "Please segment " + sents[0].strip().lower()
                    
                    ann_ids = [ref["ann_id"] for ref in refs]
                    anns = self.refer_api.loadAnns(ann_ids=ann_ids[0])
                    masks = [self.refer_api.getMask(ann)['mask'] for ann in anns]

                else: # reason_seg (val)
                    image_path = self.data_list[idx]
                    json_path = image_path.replace(".jpg", ".json")

                    if not os.path.exists(json_path):
                        # 如果JSON不存在，跳过
                        idx = random.randint(0, len(self.data_list) - 1)
                        continue
                    
                    img_for_json = cv2.imread(image_path)
                    if img_for_json is None:
                        # 如果图片损坏，跳过
                        idx = random.randint(0, len(self.data_list) - 1)
                        continue

                    mask_json, sents, is_sentence = get_mask_from_json(json_path, img_for_json)
                    
                    if not sents:
                        # 如果没有句子，跳过
                        idx = random.randint(0, len(self.data_list) - 1)
                        continue

                    full_prompt_text = sents[0].strip()
                    if not is_sentence:
                        full_prompt_text = f"What is {full_prompt_text} in this image? Please output segmentation mask."
                    else:
                        full_prompt_text = f"{full_prompt_text} Please output segmentation mask."
                    masks = [mask_json]

                # --- 统一的数据处理流程 ---
                image_cv = cv2.imread(image_path)
                if image_cv is None:
                    idx = random.randint(0, len(self.data_list) - 1)
                    continue

                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                
                pil_image = Image.fromarray(image_cv)

                image_for_sam = self.transform.apply_image(image_cv)
                resize = image_for_sam.shape[:2]
                image_for_sam_tensor = self.preprocess(torch.from_numpy(image_for_sam).permute(2, 0, 1).contiguous())
                # ===================== 在这里添加调试代码 =====================
                print(f"\n[DEBUG] Loading Val Sample: {image_path}")
                print(f"[DEBUG]   - Loaded sents: {sents}")
                print(f"[DEBUG]   - Generated prompt: {full_prompt_text}\n")
                # =============================================================

                # messages = [
                #     {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": full_prompt_text}]},
                #     {"role": "assistant", "content": ""} # 对于推理，将助手内容设置为空字符串
                # ]
                answer = random.choice(ANSWER_LIST) # e.g., "Sure, it is [SEG]."
                messages = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": full_prompt_text}]},
                    {"role": "assistant", "content": answer} 
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
                    sents,
                    True, # inference=True
                )
            
            except Exception as e:
                # 捕获任何异常，打印错误，然后随机选择下一个样本重试
                print(f"警告: 在处理验证集样本索引 {idx} (路径: {self.data_list[idx]}) 时发生错误: {e}。正在尝试下一个样本...")
                idx = random.randint(0, len(self.data_list) - 1)

