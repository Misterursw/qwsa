import argparse
import os
import re
import sys
from PIL import Image
import cv2
import torch
from typing import List
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor
from model.QWSA import QWSAForCausalLM
from model.segment_anything.utils.transforms import ResizeLongestSide
import logging
import numpy as np
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
import argparse
import os
import re
import sys
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor
from model.QWSA import QWSAForCausalLM
from model.segment_anything.utils.transforms import ResizeLongestSide
import logging
from transformers.generation.utils import StoppingCriteria
from typing import List
from transformers import StoppingCriteriaList
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: List[List[int]]):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if len(input_ids[0]) >= len(stop_ids):
                if torch.equal(input_ids[0][-len(stop_ids):], torch.tensor(stop_ids, device=input_ids.device)):
                    return True
        return False

def inference(model: QWSAForCausalLM, tokenizer, processor, image_path: str, prompt: str, args):
    """
    执行推理：文本生成、特征提取和掩码预测。
    """
    logging.info(f"加载图像: {image_path}")
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        logging.error(f"无法读取图像文件: {image_path}")
        return None, "错误: 无法读取图像文件。"
    
    image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_cv_rgb)
    
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
        {"role": "assistant", "content": ""}
    ]
    
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    qwen_inputs = processor(text=[text], images=[pil_image], return_tensors="pt", padding=True).to(model.device)
    
    if 'pixel_values' in qwen_inputs:
        qwen_inputs['pixel_values'] = qwen_inputs['pixel_values'].to(model.dtype)
        # 添加调试信息
        logging.info(f"Qwen pixel_values shape: {qwen_inputs['pixel_values'].shape}")
    
    logging.info("模型正在生成文本...")
    
    stop_strings = ["</answer>", "<|im_end|>"]
    stop_token_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings if s]
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
    
    with torch.no_grad():
        generate_outputs = model.generate(
            **qwen_inputs,
            max_new_tokens=args.model_max_length,
            do_sample=True, top_p=0.9, temperature=0.7,
            output_hidden_states=True, return_dict_in_generate=True,
            use_cache=True, stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
        )

    if generate_outputs.hidden_states is None:
        logging.error("模型未返回 hidden_states。")
        return None, "错误：模型未能获取隐藏状态。"

    output_ids = generate_outputs.sequences[0]
    generated_sequence = output_ids[qwen_inputs['input_ids'].shape[1]:]
    assistant_response = tokenizer.decode(generated_sequence, skip_special_tokens=False).strip()
    
    logging.info(f"模型生成回答 (完整解码):\n{assistant_response}")

    # ==================== 详细解码打印 ====================
    logging.info("------ 开始逐个Token解码分析 ------")
    for i, token_id in enumerate(generated_sequence):
        token_text = tokenizer.decode([token_id])
        logging.info(f"Token {i}: ID={token_id.item():<6} -> Decoded: '{token_text}'")
    logging.info("------ 逐个Token解码分析结束 ------")
    # ==========================================================

    predicted_mask = None
    
    # 寻找分割触发信号的代码保持不变
    SEG_TRIGGER_WORD = "seg"
    seg_word_id = tokenizer.convert_tokens_to_ids(SEG_TRIGGER_WORD)
    
    logging.info(f"正在寻找触发词 '{SEG_TRIGGER_WORD}', ID: {seg_word_id}")
    
    seg_indices = torch.where(generated_sequence == seg_word_id)[0]
    
    if len(seg_indices) == 0:
        logging.info("未找到独立的'seg' token，尝试在生成文本中查找...")
        if SEG_TRIGGER_WORD in assistant_response.lower():
            logging.info(f"在生成文本中找到'{SEG_TRIGGER_WORD}'，尝试定位...")
            for i in range(len(generated_sequence) - 5, len(generated_sequence)):
                if i >= 0:
                    seg_indices = torch.tensor([i])
                    break
    
    if len(seg_indices) == 0:
        logging.info("尝试备选方案：查找<|seg_mask|> token...")
        SEG_TOKEN = "<|seg_mask|>"
        seg_token_id = tokenizer.convert_tokens_to_ids(SEG_TOKEN)
        seg_indices = torch.where(generated_sequence == seg_token_id)[0]
        
        if len(seg_indices) > 0:
            logging.info(f"找到 <|seg_mask|> token，ID: {seg_token_id}")
    
    if len(seg_indices) > 0:
        seg_index = seg_indices[0].item()
        logging.info(f"检测到分割触发信号，位置: {seg_index}。开始预测掩码...")
        
        try:
            seg_step_hidden_states = generate_outputs.hidden_states[seg_index]
            last_layer_hidden_states = seg_step_hidden_states[-1]
            seg_embedding = last_layer_hidden_states[:, -1, :]
        except IndexError as e:
            logging.error(f"索引 hidden_states 出错: {e}")
            return None, "错误: 提取 token 嵌入时索引越界。"
        
        projected_seg_embedding = model.text_hidden_fcs[0](seg_embedding).to(dtype=model.dtype)
        image_tensor_for_sam = qwen_inputs['pixel_values']
        
        # 添加调试信息
        logging.info(f"传入SAM的图像张量形状: {image_tensor_for_sam.shape}")

        with torch.no_grad():
            sam_image_embedding = model.get_sam_image_embeddings(image_tensor_for_sam)
            sparse_embeddings, dense_embeddings = model.visual_model.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=projected_seg_embedding.unsqueeze(1))
            sparse_embeddings = sparse_embeddings.to(projected_seg_embedding.dtype)
            low_res_masks, _ = model.visual_model.mask_decoder(
                image_embeddings=sam_image_embedding,
                image_pe=model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False)
            pred_mask_processed = model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=pil_image.size[::-1],
                original_size=image_cv_rgb.shape[:2])
            
            if len(pred_mask_processed.shape) >= 2:
                predicted_mask = (pred_mask_processed[0, 0] > model.visual_model.mask_threshold).cpu().numpy()
                logging.info(f"掩码预测完成，形状: {predicted_mask.shape}")
            else:
                logging.error(f"预测掩码形状异常: {pred_mask_processed.shape}")
    else:
        logging.warning("模型未生成分割触发信号，跳过掩码预测。")

    return predicted_mask, assistant_response

def main():
    parser = argparse.ArgumentParser(description="QWSA Inference Script")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--sam-checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="./test_output")
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=1024, type=int)
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    torch_dtype = torch.bfloat16 if args.precision == "bf16" else (torch.half if args.precision == "fp16" else torch.float32)

    logging.info(f"正在从 {args.model_path} 加载Tokenizer和Processor...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ==================== 简化的Token添加逻辑 ====================
    logging.info("检查和添加必要的特殊Token...")
    
    # 检查'seg'是否已在词汇表中
    seg_word = "seg"
    seg_token_id = tokenizer.convert_tokens_to_ids(seg_word)
    
    if seg_token_id != tokenizer.unk_token_id:
        logging.info(f"触发词 '{seg_word}' 已在词汇表中，ID: {seg_token_id}")
    else:
        logging.warning(f"触发词 '{seg_word}' 不在词汇表中，将尝试其他方法")
    
    # 添加其他可能需要的token
    custom_tokens_to_add = ["<|seg_mask|>", "<think>", "</think>", "<answer>", "</answer>"]
    tokens_to_add = []
    
    for token in custom_tokens_to_add:
        if tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id:
            tokens_to_add.append(token)
    
    if tokens_to_add:
        num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        logging.info(f"成功添加 {num_added_tokens} 个新token: {tokens_to_add}")
    else:
        logging.info("所有必要的token已存在")
        num_added_tokens = 0
    
    # 最终验证
    final_seg_token_id = tokenizer.convert_tokens_to_ids("<|seg_mask|>")
    logging.info(f"备选 <|seg_mask|> token ID: {final_seg_token_id}")
    # ==========================================================

    logging.info("正在加载QWSA模型...")
    model = QWSAForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        vision_pretrained=args.sam_checkpoint,
        image_size=args.image_size,
        seg_token_idx=final_seg_token_id,  # 使用<|seg_mask|>的ID作为备选
        train_mask_decoder=True,
        out_dim=256
    ).cuda().eval()
    
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
        logging.info(f"模型词嵌入层大小已调整为 {len(tokenizer)}")
    
    model.config.output_hidden_states = True
    logging.info("模型加载完成。")

    predicted_mask, text_response = inference(model, tokenizer, processor, args.image_path, args.prompt, args)

    if text_response:
        base_filename = os.path.splitext(os.path.basename(args.image_path))[0]
        text_save_path = os.path.join(args.out_dir, f"{base_filename}_response.txt")
        with open(text_save_path, "w", encoding="utf-8") as f:
            f.write(text_response)
        logging.info(f"文本回答已保存到: {text_save_path}")

        if predicted_mask is not None:
            mask_save_path = os.path.join(args.out_dir, f"{base_filename}_mask.png")
            cv2.imwrite(mask_save_path, predicted_mask.astype(np.uint8) * 255)
            logging.info(f"分割掩码已保存到: {mask_save_path}")

            overlay_save_path = os.path.join(args.out_dir, f"{base_filename}_overlay.jpg")
            original_image_bgr = cv2.imread(args.image_path)
            if predicted_mask.shape != original_image_bgr.shape[:2]:
                predicted_mask = cv2.resize(predicted_mask.astype(np.uint8), (original_image_bgr.shape[1], original_image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            overlay_color = np.array([0, 0, 255], dtype=np.uint8)
            overlay = original_image_bgr.copy()
            bool_mask = predicted_mask.astype(bool)
            overlay[bool_mask] = cv2.addWeighted(overlay[bool_mask], 0.5, overlay_color, 0.5, 0)
            cv2.imwrite(overlay_save_path, overlay)
            logging.info(f"覆盖图像已保存到: {overlay_save_path}")
        else:
            logging.info("由于未生成掩码，跳过掩码和覆盖图像的保存。")

if __name__ == "__main__":
    main()