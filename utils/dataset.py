import glob
import os
import random
from PIL import Image
import logging
import json
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
from model.qwen import conversation as conversation_lib
from model.qwen.constants import IGNORE_INDEX
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .refer import REFER
from .utils import ANSWER_LIST
from .reason_seg_dataset import ReasonSegDataset

def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1, processor=None
):
    """
    Data collator function that handles and packages a batch of data.
    Adds handling for invalid samples and empty batches.
    """
    # 1. Filter out invalid samples (those returning None in __getitem__)
    batch = [item for item in batch if item is not None]
    
    # 2. If the entire batch is empty after filtering, return a complete but empty structure
    if not batch:
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

    # Initialize collectors
    image_path_list, images_list, pil_images, all_messages_structured = [], [], [], []
    masks_list, label_list, resize_list, questions_list, sampled_classes_list = [], [], [], [], []
    offset_list = [0]
    cnt = 0
    inferences = []

    # 3. Collect raw data from the valid batch
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

    # 4. Use Qwen processor to handle text and images
    # For training, the conversation includes the answer. For validation, the assistant's part is empty.
    texts_for_processing = [
        processor.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=(not conversation[-1]['content'])
        )
        for conversation in all_messages_structured
    ]
    
    # ==================== 修复：移除硬编码逻辑 ====================
    # The problematic logic of force-appending "[SEG]" for validation samples has been removed.
    # The model is now expected to generate the "[SEG]" token on its own during validation,
    # which allows for a true end-to-end evaluation.
    # ==========================================================

    inputs = processor(
        text=texts_for_processing,
        images=pil_images,
        return_tensors="pt",
        padding=True
    )

    # 5. Extract all required outputs
    input_ids = inputs['input_ids']
    attention_masks = inputs['attention_mask']
    images_clip = inputs['pixel_values']
    image_grid_thw = inputs.get('image_grid_thw')

    # 6. Create target tensor for 'labels' and mask padding
    targets = input_ids.clone()
    targets[targets == tokenizer.pad_token_id] = IGNORE_INDEX
    
    # 7. Precisely mask out the user's question part, only calculating loss on the assistant's answer
    # This logic now correctly handles both training (with full answer) and validation (with empty answer)
    for i, conversation in enumerate(all_messages_structured):
        # The assistant's turn is the last one
        assistant_content = conversation[-1]['content']
        
        # Find the starting point of the assistant's response in the tokenized sequence
        # Qwen template adds specific tokens around roles
        assistant_prompt_marker = processor.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": ""}], tokenize=False, add_generation_prompt=True
        ).replace(tokenizer.eos_token, '') # Get the prefix for the assistant's turn

        full_text = texts_for_processing[i]
        # Find the last occurrence, which should be the start of the final generation prompt
        assistant_start_pos = full_text.rfind(assistant_prompt_marker.strip())

        if assistant_start_pos != -1:
            # Tokenize everything up to the assistant's turn to find the length to ignore
            instruction_part = full_text[:assistant_start_pos]
            instruction_len = len(tokenizer(instruction_part, add_special_tokens=False).input_ids)
            targets[i, :instruction_len] = IGNORE_INDEX
        
        # During validation, the assistant's content is empty, so we don't need to mask it.
        # During training, the label for the assistant's part is already in `targets`.

    # 8. Build and return the final batch dictionary
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
        "inference": inferences[0] if inferences else False, # Handle empty list case
        # Auxiliary info, needs to be filtered before model call
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
        self.val_data = []

        # ==================== 修改：加载验证集逻辑 ====================
        splits = val_dataset.split("|")
        if len(splits) == 2 and splits[0] == "ReasonSeg":
            ds, split = splits
            val_json_path = os.path.join(self.base_image_dir, "reason_seg", ds, "explanatory", f"{split}.json")
            if os.path.exists(val_json_path):
                logging.info(f"Found validation set file: {val_json_path}")
                with open(val_json_path, 'r') as f:
                    # We load the json which contains image, query, and the ground truth mask json path
                    self.val_data = json.load(f)
                self.data_type = "reason_seg_explanatory"
            else:
                raise FileNotFoundError(f"Specified validation set file not found: {val_json_path}")
        else:
            raise ValueError("Validation set config is incorrect. Use format 'ReasonSeg|val' and ensure the corresponding .json file exists.")
        # ==========================================================

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
        # Loop to handle potential errors and skip problematic samples
        while True:
            try:
                item = self.val_data[idx]
                image_name = item["image"]
                json_name = item["json"]
                
                # Construct full paths
                # Assuming images and json files are in the same 'val' folder
                image_path = os.path.join(self.base_image_dir, "reason_seg", "ReasonSeg", "val", image_name)
                json_path = os.path.join(self.base_image_dir, "reason_seg", "ReasonSeg", "val", json_name)

                if not os.path.exists(image_path) or not os.path.exists(json_path):
                    logging.warning(f"Validation file missing, skipping {image_path} or {json_path}")
                    idx = random.randint(0, len(self.val_data) - 1) # Try a different random index
                    continue

                # Get query from the validation data
                full_prompt_text = item["query"]
                # The ground truth answer is also in the data, but we won't use it for input
                # It can be used for text-based evaluation if needed later.
                gt_answer_text = item["outputs"] 

                # Get ground truth mask from the mask json file
                img_for_json = cv2.imread(image_path)
                if img_for_json is None:
                    logging.warning(f"Could not read image {image_path}, skipping.")
                    idx = random.randint(0, len(self.val_data) - 1)
                    continue
                
                masks, _, _ = get_mask_from_json(json_path, img_for_json)
                masks = [masks] # Wrap in a list

                # --- Unified data processing flow ---
                image_cv = cv2.cvtColor(img_for_json, cv2.COLOR_BGR2RGB)
                
                pil_image = Image.fromarray(image_cv)
                image_for_sam = self.transform.apply_image(image_cv)
                resize = image_for_sam.shape[:2]
                image_for_sam_tensor = self.preprocess(torch.from_numpy(image_for_sam).permute(2, 0, 1).contiguous())
                
                # ==================== 修复：构建正确的验证对话 ====================
                # For validation, the assistant's response is empty.
                # The model is expected to generate the full response, including CoT and [SEG].
                messages = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": full_prompt_text}]},
                    {"role": "assistant", "content": ""} # Assistant content is empty for inference
                ]
                # ================================================================

                final_masks = np.stack(masks, axis=0)
                final_masks = torch.from_numpy(final_masks)
                # The label for validation is the ground truth mask's shape, filled with ignore_label.
                # It's used to determine the output size for postprocessing.
                label = torch.ones(final_masks.shape[1], final_masks.shape[2]) * self.ignore_label
                
                # If all steps succeed, return data and break the loop
                return (
                    image_path,
                    image_for_sam_tensor,
                    pil_image,
                    messages,
                    final_masks,
                    label,
                    resize,
                    [full_prompt_text],
                    [], # sents is not needed here as we use the full query
                    True, # inference=True
                )
            
            except Exception as e:
                logging.error(f"Error processing validation sample index {idx}: {e}. Trying next sample.")
                idx = (idx + 1) % len(self.val_data) # Move to the next sample to avoid infinite loops
