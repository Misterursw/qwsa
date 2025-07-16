# file: test.py

import argparse
import os
import sys
from PIL import Image
import cv2
import torch
import numpy as np
from transformers import AutoTokenizer, AutoProcessor
from model.QWSA import QWSAForCausalLM
from torchvision.transforms import ToTensor
import logging
import warnings

# 忽略所有 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    stream=sys.stdout
)

def inference(model: QWSAForCausalLM, tokenizer, processor, image_path: str, prompt: str, args):
    """
    执行解耦后的三步推理：1. 文本生成, 2. 特征提取, 3. 掩码预测
    """
    # --- 步骤 1: 加载图像并准备Qwen输入 ---
    logging.info(f"加载图像: {image_path}")
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        logging.error(f"无法读取图像文件: {image_path}")
        return None, "错误: 无法读取图像文件。"
    
    image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_cv_rgb)
    
    # 直接使用processor，它会自动处理图像占位符
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ],
        }
    ]
    
    # 应用chat模板
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    
    # 处理输入
    qwen_inputs = processor(
        text=[text],
        images=[pil_image],
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    # 确保数据类型正确
    if 'pixel_values' in qwen_inputs:
        qwen_inputs['pixel_values'] = qwen_inputs['pixel_values'].to(model.dtype)
    
# --- 步骤 2: 生成文本和隐藏状态 ---
    logging.info("模型正在生成文本和 CoT...")
    with torch.no_grad():
        # 添加停止词（保持原样，因为 <answer> 等非特殊 token）
        stop_strings = ["</answer>", "<|im_end|>"]
        stopping_criteria = []
        # 如果 tokenizer 支持停止词
        if hasattr(tokenizer, "encode"):
            stopping_criteria = []
            for stop_str in stop_strings:
                stop_ids = tokenizer.encode(stop_str, add_special_tokens=False)
                if stop_ids:
                    from transformers import StoppingCriteria, StoppingCriteriaList
                    
                    class StopOnTokens(StoppingCriteria):
                        def __init__(self, stop_id_seq: List[int]):
                            self.stop_id_seq = stop_id_seq
                        
                        def __call__(self, input_ids, scores, **kwargs):
                            if len(self.stop_id_seq) > 0 and input_ids[0][-len(self.stop_id_seq):].tolist() == self.stop_id_seq:
                                return True
                            return False
                    
                    stopping_criteria.append(StopOnTokens(stop_ids))  # 直接传 stop_ids (list[int])
        
        generate_outputs = model.generate(
            **qwen_inputs,
            max_new_tokens=args.model_max_length,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            output_hidden_states=True,
            return_dict_in_generate=True,
            use_cache=True,
            stopping_criteria=StoppingCriteriaList(stopping_criteria) if stopping_criteria else None,  # 添加条件，避免空列表
        )
    if generate_outputs.hidden_states is None:
        logging.error("模型未返回 hidden_states。请确认 config 或 generate() 的参数。")
    logging.info(f"Hidden states length: {len(generate_outputs.hidden_states)}")
    # 解码文本
    output_ids = generate_outputs.sequences[0]
    assistant_response = tokenizer.decode(output_ids[qwen_inputs['input_ids'].shape[1]:], skip_special_tokens=False).strip()
    
    # 新增：替换 <|extra_0|> 为 "[SEG]" 以显示直观（其他标签不变）
    seg_token_str = "<|extra_0|>"  # 对应 [SEG] 的 extra token
    assistant_response = assistant_response.replace(seg_token_str, "[SEG]")
    
    logging.info(f"模型生成回答 (包含特殊tokens):\n{assistant_response}")
    if "<answer>" in assistant_response and "</answer>" not in assistant_response:
        # 找到 [SEG] 或文本结束的位置（用显示字符串检查）
        if "[SEG]" in assistant_response:
            assistant_response = assistant_response.replace("[SEG]", "</answer>\n[SEG]")
        else:
            assistant_response += "\n</answer>"
    # 调试：打印生成的 token IDs
    generated_sequence = output_ids[qwen_inputs['input_ids'].shape[1]:]
    logging.info(f"Generated token IDs: {generated_sequence.tolist()}")
    
    # 调试：解码每个 token
    for i, token_id in enumerate(generated_sequence):
        token = tokenizer.decode([token_id], skip_special_tokens=False)
        logging.info(f"Token {i}: ID={token_id}, Text='{token}'")
    
    # --- 步骤 3: 提取[SEG]嵌入并预测掩码 ---
    predicted_mask = None
    seg_token_id = model.seg_token_idx  # 已设置为 <|extra_0|> 的 ID
    logging.info(f"Looking for SEG token with ID: {seg_token_id}")
    
    seg_token_mask = (generated_sequence == seg_token_id)
    logging.info(f"SEG token mask: {seg_token_mask}")
    logging.info(f"Number of SEG tokens found: {seg_token_mask.sum().item()}")
    
    # 如果找不到 [SEG] token，尝试通过文本查找（用显示字符串）
    if not torch.any(seg_token_mask):
        if "[SEG]" in assistant_response:
            logging.warning("[SEG] found in text but not as a single token. Tokenizer may be splitting it.")
            # 尝试重新 tokenize 来找到正确的位置
            # 注意：这里用替换后的 assistant_response，但实际 token 是 extra，所以如果替换正确，此处应已匹配 seg_token_id
            temp_tokens = tokenizer(assistant_response, return_tensors="pt")['input_ids'][0]
            logging.info(f"Re-tokenized response: {temp_tokens.tolist()}")
    logging.info(f"Input shape: {qwen_inputs['input_ids'].shape}")
    if torch.any(seg_token_mask):
        logging.info("检测到 [SEG] token, 开始预测掩码...")
        
        # 找到第一个 [SEG] token 的位置
        seg_index = torch.where(seg_token_mask)[0][0].item()

        # seg_index 是相对于生成序列的位置，需要加1因为hidden_states包含了每个生成步骤
        seg_step_index = seg_index + 1  # +1 因为第0步是输入
        if seg_step_index < len(generate_outputs.hidden_states):
            try:
                seg_step_hidden_states = generate_outputs.hidden_states[seg_step_index]  # 不要用[-1]
            except IndexError as e:
                logging.error(f"Index error accessing hidden states: {e}")
                logging.error(f"Hidden states length: {len(generate_outputs.hidden_states)}, seg_index: {seg_index}")
                raise
            seg_embedding = seg_step_hidden_states[:, -1, :]
        else:
            # 如果索引超出范围，使用最后一个隐藏状态
            seg_step_hidden_states = generate_outputs.hidden_states[-1][-1]
            # 需要找到对应的token位置
            seg_position = qwen_inputs['input_ids'].shape[1] + seg_index
            seg_embedding = seg_step_hidden_states[:, seg_position, :]
        # 投影 [SEG] 嵌入
        seg_embedding = seg_embedding.to(dtype=projected_seg_embedding.dtype)

        projected_seg_embedding = model.text_hidden_fcs[0](seg_embedding)
        logging.info(f"SEG token position: {seg_index}")
        from torchvision.transforms import ToTensor
        # 或者直接使用
        image_tensor_for_sam = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
        image_tensor_for_sam = image_tensor_for_sam.unsqueeze(0).to(model.device)
        
        with torch.no_grad():
            sam_image_embedding = model.get_sam_image_embeddings(image_tensor_for_sam)
            
            # 使用SAM解码器进行预测
            sparse_embeddings, dense_embeddings = model.visual_model.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=projected_seg_embedding.unsqueeze(1)
            )
            sparse_embeddings = sparse_embeddings.to(projected_seg_embedding.dtype)

            low_res_masks, _ = model.visual_model.mask_decoder(
                image_embeddings=sam_image_embedding,
                image_pe=model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            # 后处理掩码
            pred_mask_processed = model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=pil_image.size[::-1], # (H, W)
                original_size=image_cv_rgb.shape[:2] # (H, W)
            )
            
            print(f"pred_mask_processed shape: {pred_mask_processed.shape}")
            if len(pred_mask_processed.shape) >= 2:
                predicted_mask = (pred_mask_processed[0, 0] > 0).cpu().numpy()
            else:
                logging.error(f"Unexpected mask shape: {pred_mask_processed.shape}")
            logging.info(f"掩码预测完成，形状: {predicted_mask.shape}")
    else:
        logging.warning("模型未生成 [SEG] token，跳过掩码预测。")

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

    # 先加载 tokenizer 和 processor
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    
    # 添加特殊tokens - 确保它们被正确添加
    # special_tokens = ["[SEG]", "<think>", "</think>", "<answer>", "</answer>"]
    
    # # 检查这些 tokens 是否已经存在
    # existing_special_tokens = tokenizer.special_tokens_map.get('additional_special_tokens', [])
    # tokens_to_add = [token for token in special_tokens if token not in existing_special_tokens and token not in tokenizer.get_vocab()]
    
    # if tokens_to_add:
    #     num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
    #     logging.info(f"Added {num_added_tokens} special tokens: {tokens_to_add}")
    
    # # 验证 tokens 是否被正确添加
    # seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    # logging.info(f"[SEG] token ID: {seg_token_id}")
    # logging.info(f"[SEG] in vocab: {'[SEG]' in tokenizer.get_vocab()}")
    
    # # 如果 seg_token_id 是 tokenizer.unk_token_id，说明没有正确添加
    # if seg_token_id == tokenizer.unk_token_id:
    #     logging.error("[SEG] token was not properly added to vocabulary!")
    # # 确保在加载模型时传递所有必要的自定义参数

    # 验证 [SEG] 用 extra token
    seg_token_str = "<|extra_0|>"  # 可根据需要调整 N，确保在 vocab 中
    seg_token_id = tokenizer.convert_tokens_to_ids(seg_token_str)
    logging.info(f"[SEG] token str: {seg_token_str}, ID: {seg_token_id}")
    logging.info(f"[SEG] tokenized: {tokenizer.tokenize(seg_token_str)}")  # 应为单一 token
    
    if seg_token_id == tokenizer.unk_token_id:
        logging.error("[SEG] token was not properly added to vocabulary! Check extra token availability.")
    model = QWSAForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        vision_pretrained=args.sam_checkpoint,
        image_size=args.image_size,
        seg_token_idx = tokenizer.convert_tokens_to_ids("<|extra_0|>"),
        train_mask_decoder=True, # 假设在训练时是True
        out_dim=256 # 与训练时保持一致
    ).cuda().eval()
    model.config.output_hidden_states = True
    # if tokens_to_add:
    #     model.resize_token_embeddings(len(tokenizer))
    #     logging.info(f"Resized model embeddings to {len(tokenizer)}")
    
    logging.info("模型加载完成。")
    predicted_mask, text_response = inference(model, tokenizer, processor, args.image_path, args.prompt, args)

    # --- 结果保存 ---
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

            # 创建覆盖图像
            overlay_save_path = os.path.join(args.out_dir, f"{base_filename}_overlay.jpg")
            original_image_bgr = cv2.imread(args.image_path)
            if predicted_mask.shape != original_image_bgr.shape[:2]:
                predicted_mask = cv2.resize(predicted_mask.astype(np.uint8), (original_image_bgr.shape[1], original_image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            overlay_color = np.array([0, 0, 255], dtype=np.uint8) # 红色 (BGR)
            overlay = original_image_bgr.copy()
            # 将掩码转为布尔型，用于索引
            bool_mask = predicted_mask.astype(bool)
            overlay[bool_mask] = cv2.addWeighted(overlay[bool_mask], 0.5, overlay_color, 0.5, 0)
            cv2.imwrite(overlay_save_path, overlay)
            logging.info(f"覆盖图像已保存到: {overlay_save_path}")
        else:
            logging.info("由于未生成掩码，跳过掩码和覆盖图像的保存。")

if __name__ == "__main__":
    main()