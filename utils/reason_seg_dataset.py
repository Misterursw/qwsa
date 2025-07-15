# 6.10代码/utils/reason_seg_dataset.py

import glob
import json
import os
import random
from PIL import Image
import cv2
import logging
import numpy as np
import torch
import torch.nn.functional as F
# 移除旧的CLIP处理器导入
# from transformers import CLIPImageProcessor 

from model.qwen import conversation as conversation_lib # 确保使用Qwen的对话模板
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
# 确保从qwen的常量中导入，而不是llava的
from model.qwen.constants import DEFAULT_IMAGE_TOKEN 
from .utils import (ANSWER_LIST,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)


class ReasonSegDataset(torch.utils.data.Dataset):
    """
    用于处理推理分割任务的数据集类。
    """
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower, # 这个参数现在可以考虑移除，因为processor会处理
        processor,    # <-- 必须传入Qwen的processor
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.processor = processor # <-- 保存processor
        self.transform = ResizeLongestSide(image_size)

        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        reason_seg_data_path, splits = reason_seg_data.split("|")
        splits = splits.split("_")
        images = []
        for split in splits:
            images_split = glob.glob(
                os.path.join(
                    base_image_dir, "reason_seg", reason_seg_data_path, split, "*.jpg"
                )
            )
            images.extend(images_split)
        jsons = [path.replace(".jpg", ".json") for path in images]
        self.reason_seg_data = (images, jsons)

        print("number of reason_seg samples: ", len(images))
        logging.info(f"Initialized ReasonSegDataset with {len(images)} samples.")


        if explanatory != -1:
            self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
            self.img_to_explanation = {}
            try:
                with open(
                    os.path.join(
                        base_image_dir,
                        "reason_seg",
                        reason_seg_data_path,
                        "explanatory",
                        "train.json",
                    )
                ) as f:
                    items = json.load(f)
                for item in items:
                    img_name = item["image"]
                    self.img_to_explanation[img_name] = {
                        "query": item["query"],
                        "outputs": item["outputs"],
                    }
                print("len(self.img_to_explanation): ", len(self.img_to_explanation))
            except FileNotFoundError:
                print("Explanatory JSON not found, skipping.")


    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """规范化像素值并填充为方形输入。"""
        # 规范化颜色
        x = (x - self.pixel_mean) / self.pixel_std
        # 填充
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        while True:
            try:
                images, jsons = self.reason_seg_data
                idx = random.randint(0, len(images) - 1)
                image_path = jsons[idx].replace(".json", ".jpg")
                json_path = jsons[idx]
                
                # 从图片路径中提取图片名，用于查询explanation
                img_name = os.path.basename(image_path)

                # ==================== 修改开始 ====================
                # 核心修改：检查这张图片是否存在于解释性数据中
                if self.explanatory != -1 and img_name in self.img_to_explanation:
                    # 如果存在，则直接使用解释性数据
                    explanation_data = self.img_to_explanation[img_name]
                    question = explanation_data["query"]
                    answer = explanation_data["outputs"] # 假设JSON中也有答案字段，如果没有则需要调整
                else:
                    # 如果图片没有对应的复杂CoT数据，则跳过这个样本
                    # print(f"警告: 样本 {img_name} 没有对应的解释性数据，跳过。")
                    continue # 直接进入下一次循环，寻找下一个有效的样本
                # ==================== 修改结束 ====================

                if not os.path.exists(image_path):
                    continue
                image_cv = cv2.imread(image_path)
                if image_cv is None:
                    continue
                
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                ori_size = image_cv.shape[:2]

                image_for_sam = self.transform.apply_image(image_cv)
                resize = image_for_sam.shape[:2]
                
                mask_from_json, sents, is_sentence = get_mask_from_json(json_path, image_cv)
                
                if sents is None or len(sents) == 0:
                    continue

                # 注意：这里的 sents 仍然需要，因为我们需要它来确定要分割哪个mask
                # 但 question 和 answer 已经从我们的CoT数据中获取了
                if self.num_classes_per_sample < len(sents):
                    sampled_inds = np.random.choice(
                        list(range(len(sents))),
                        size=self.num_classes_per_sample,
                        replace=False,
                    )
                else:
                    sampled_inds = list(range(len(sents)))
                
                sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
                sampled_masks = [mask_from_json for _ in sampled_inds]

                # 准备结构化的对话消息，这里的answer需要包含[SEG] token
                # 我们假设CoT的答案也需要一个地方插入[SEG]
                # 这里我们简单地在末尾添加，您可能需要根据需要调整
                final_answer = answer + " [SEG]"
                messages = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
                    {"role": "assistant", "content": final_answer}
                ]

                image_for_sam_tensor = self.preprocess(torch.from_numpy(image_for_sam).permute(2, 0, 1).contiguous())
                masks_tensor = torch.from_numpy(np.stack(sampled_masks, axis=0))
                label_tensor = torch.ones(masks_tensor.shape[1], masks_tensor.shape[2]) * self.ignore_label
                pil_image = Image.fromarray(image_cv)
                
                return (
                    image_path,
                    image_for_sam_tensor,
                    pil_image,
                    messages,
                    masks_tensor,
                    label_tensor,
                    resize,
                    [question],
                    sampled_sents,
                    False,
                )

            except Exception as e:
                print(f"在处理索引 {idx} (路径: {image_path}) 时发生错误: {e}，正在尝试下一个样本...")
                import time
                time.sleep(0.1)