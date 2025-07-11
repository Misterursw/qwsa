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
        # 循环直到找到一个有效的样本
        while True:
            try:
                images, jsons = self.reason_seg_data
                # 从数据集中随机选择一个索引
                idx = random.randint(0, len(images) - 1)
                image_path = jsons[idx].replace(".json", ".jpg")
                json_path = jsons[idx]

                # 检查图像文件是否存在
                if not os.path.exists(image_path):
                    print(f"警告: 图像文件不存在于 {image_path}，跳过此样本。")
                    # 继续下一次循环，尝试获取新样本
                    continue

                image_cv = cv2.imread(image_path)
                # 检查图像是否成功读取
                if image_cv is None:
                    print(f"警告: 无法读取或图像文件已损坏: {image_path}，跳过此样本。")
                    continue
                
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                ori_size = image_cv.shape[:2]

                # --- SAM的图像预处理 ---
                image_for_sam = self.transform.apply_image(image_cv)
                resize = image_for_sam.shape[:2]
                
                # 从JSON获取掩码和句子
                mask_from_json, sents, is_sentence = get_mask_from_json(json_path, image_cv)
                
                # 检查句子列表是否有效
                if sents is None or len(sents) == 0:
                    print(f"警告: 在 {json_path} 中未找到有效句子，跳过此样本。")
                    continue

                # --- 数据采样 ---
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

                # --- 准备Qwen的输入 ---
                # 随机选择一个解说性问题或普通问题
                if self.explanatory != -1 and random.random() < self.explanatory:
                     # 确保解说性问题列表不为空
                    if self.explanatory_question_list:
                        question = random.choice(self.explanatory_question_list)
                    else:
                        # 如果为空，则退回使用普通问题
                        question = random.choice(self.long_question_list).format(sent=sampled_sents[0])
                else:
                    question_template = random.choice(self.long_question_list)
                    question = question_template.format(sent=sampled_sents[0])

                # 准备结构化的对话消息
                messages = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
                    {"role": "assistant", "content": random.choice(self.answer_list)}
                ]

                # --- 准备返回的张量 ---
                image_for_sam_tensor = self.preprocess(torch.from_numpy(image_for_sam).permute(2, 0, 1).contiguous())

                masks_tensor = torch.from_numpy(np.stack(sampled_masks, axis=0))
                
                label_tensor = torch.ones(masks_tensor.shape[1], masks_tensor.shape[2]) * self.ignore_label
                
                pil_image = Image.fromarray(image_cv)
                
                # 如果所有步骤都成功，则返回数据并跳出循环
                return (
                    image_path,
                    image_for_sam_tensor,
                    pil_image,
                    messages,
                    masks_tensor,
                    label_tensor,
                    resize,
                    [question], # 保持questions_list的格式
                    sampled_sents,
                    False, # inference
                )

            except Exception as e:
                # 捕获任何异常，打印错误信息，然后继续下一次循环
                print(f"在处理索引 {idx} (路径: {image_path}) 时发生错误: {e}，正在尝试下一个样本...")
                # 等待一小段时间，避免在连续错误时刷屏
                import time
                time.sleep(0.1)