import argparse
import os
import re
import sys
from PIL import Image
import cv2
import torch
import json
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor
from model.QWSA import QWSAForCausalLM
from model.segment_anything.utils.transforms import ResizeLongestSide
import logging
import numpy as np
import sys
# --- 配置日志 ---
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
from model.QWSA import QWSAForCausalLM, preprocess_image_for_sam
from model.segment_anything.utils.transforms import ResizeLongestSide
import logging


# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)




def inference(model, tokenizer, processor, image_path, prompt, args):
    """
    执行完整的端到端推理，包括文本生成和掩码预测。
    """
    # 1. 加载和预处理图像
    logging.info(f"正在加载图像: {image_path}")
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        logging.error(f"无法读取图像文件: {image_path}")
        return None, "错误: 无法读取图像文件。"
        
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    original_size = image_cv.shape[:2]
    pil_image = Image.fromarray(image_cv)
    logging.info(f"image_cv 形状: {image_cv.shape}")
    logging.info(f"original_size: {original_size}")
    logging.info(f"pil_image 模式: {pil_image.mode}, 大小: {pil_image.size}")
    
    # 2. 准备结构化的对话 prompt，显式包含 <|placeholder|> 占位符
    messages = [
        {
            "role": "user",
            "content": f"{prompt} <|placeholder|>"
        },
        {
            "role": "assistant",
            "content": ""
        }
    ]
    
    # 3. 将结构化消息应用模板，转换为字符串
    text_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 4. 验证 text_prompt 是否为字符串
    if not isinstance(text_prompt, str):
        logging.error(f"apply_chat_template 未生成有效字符串: {text_prompt}")
        return None, "错误: 无法生成有效的文本 prompt。"

    # 5. 调用处理器，生成 pixel_values
    inputs = processor(text=text_prompt, images=[pil_image], return_tensors="pt").to(model.device)
    inputs['pixel_values'] = inputs['pixel_values'].to(model.dtype)
    logging.info(f"processor 输出 keys: {list(inputs.keys())}")
    logging.info(f"pixel_values in inference形状: {inputs['pixel_values'].shape}")
    logging.info(f"pixel_values in inference范围: min={inputs['pixel_values'].min()}, max={inputs['pixel_values'].max()}")

    # 6. 保存分割所需的参数
    seg_params = {
        'offset': [0],
        'resize_list': [original_size],
        'label_list': [original_size]
    }
    model_inputs = {
        'input_ids': inputs['input_ids'],
        'pixel_values': inputs['pixel_values'],
        'attention_mask': inputs.get('attention_mask')
    }
    logging.info(f"model.generate 输入 pixel_values 形状: {model_inputs['pixel_values'].shape}")
    # 7. 模型生成文本和隐藏状态
    logging.info("模型正在生成文本和 CoT...")
    with torch.no_grad():
        generate_ids = model.generate(
            input_ids=inputs['input_ids'],
            pixel_values=inputs['pixel_values'],
            attention_mask=inputs.get('attention_mask'),
            max_new_tokens=args.model_max_length,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            output_hidden_states=True,
            return_dict_in_generate=True,
            use_cache=True,
        )

    # 8. 解码生成的文本
    output_ids = generate_ids.sequences[0]
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=False)
    
    assistant_response_parts = generated_text.split("<|im_start|>assistant")
    if len(assistant_response_parts) > 1:
        assistant_response = assistant_response_parts[1].split("<|im_end|>")[0].strip()
    else:
        assistant_response = "模型未能生成有效的回答格式。"
        
    logging.info(f"模型生成的回答:\n{assistant_response}")

    # 9. 查找 [SEG] token 并提取其隐藏状态
    seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    generated_sequence = generate_ids.sequences[0, inputs['input_ids'].shape[1]:]
    seg_token_mask = (generated_sequence == seg_token_id)

    if not torch.any(seg_token_mask):
        logging.warning("模型没有生成 [SEG] token。无法进行分割。")
        return None, assistant_response

    seg_token_indices_in_generated = torch.where(seg_token_mask)[0]
    first_seg_token_index_in_generated = seg_token_indices_in_generated[0].item()

    seg_step_hidden_states = generate_ids.hidden_states[first_seg_token_index_in_generated][-1]
    seg_embedding = seg_step_hidden_states[0, -1, :]

    # 10. 投影并预测掩码
    logging.info("正在使用 [SEG] token embedding 预测掩码...")
    with torch.no_grad():
        inputs_for_seg = {
            'pixel_values': inputs['pixel_values'],
            'input_ids': inputs['input_ids'],
            'offset': seg_params['offset'],
            'resize_list': seg_params['resize_list'],
            'label_list': seg_params['label_list']
        }
        if args.precision == "fp16":
            inputs_for_seg['pixel_values'] = inputs_for_seg['pixel_values'].half()
        elif args.precision == "bf16":
            inputs_for_seg['pixel_values'] = inputs_for_seg['pixel_values'].bfloat16()
        
        output_dict = model(**inputs_for_seg)
        pred_mask = output_dict['pred_masks'][0]  # 假设单张图像
        pred_mask = (pred_mask > 0).cpu().numpy().squeeze()

    return pred_mask, assistant_response

def main():
    parser = argparse.ArgumentParser(description="QWSA Inference Script")
    parser.add_argument("--model-path", type=str, required=True, help="Path to your trained model checkpoint.")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--prompt", type=str, required=True, help="The question or instruction for the model.")
    parser.add_argument("--sam-checkpoint", type=str, required=True, help="Path to the SAM pretrained weights.")
    parser.add_argument("--out-dir", type=str, default="./test_output", help="Directory to save the output.")
    parser.add_argument("--image_size", default=1024, type=int, help="Image size for SAM.")
    parser.add_argument("--model_max_length", default=1024, type=int, help="Max length for text generation.")
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"], help="Computation precision.")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    logging.info(f"正在从 '{args.model_path}' 加载模型、Tokenizer 和 Processor...")
    torch_dtype = torch.bfloat16 if args.precision == "bf16" else (torch.half if args.precision == "fp16" else torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    print(processor.image_processor)
    model = QWSAForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        vision_pretrained=args.sam_checkpoint,
        image_size=args.image_size  # 传递 image_size
    ).cuda().eval()
    with open(os.path.join(args.model_path, "config.json"), "r") as f:
        config = json.load(f)
    logging.info(f"模型配置: {config}")
    logging.info(f"模型属性: {dir(model)}")
    logging.info(f"是否有 vision_tower: {hasattr(model, 'vision_tower')}")
    logging.info(f"是否有 visual: {hasattr(model, 'visual')}")
    logging.info(f"是否有 vision_encoder: {hasattr(model, 'vision_encoder')}")
    logging.info(f"Processor image_mean: {processor.image_processor.image_mean}")
    logging.info(f"Processor image_std: {processor.image_processor.image_std}")
    special_tokens = ["[SEG]", "<think>", "</think>", "<answer>", "</answer>"]
    num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            input_embeds = model.get_input_embeddings().weight.data
            avg_embed = input_embeds[:-num_added_tokens].mean(dim=0, keepdim=True)
            input_embeds[-num_added_tokens:] = avg_embed
            if hasattr(model, 'get_output_embeddings'):
                output_embeds = model.get_output_embeddings().weight.data
                output_embeds[-num_added_tokens:] = avg_embed

    logging.info("模型加载完毕。")

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
            original_image = cv2.imread(args.image_path)
            if predicted_mask.shape != original_image.shape[:2]:
                predicted_mask = cv2.resize(predicted_mask.astype(np.uint8), (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            overlay_color = np.array([255, 0, 0], dtype=np.uint8)
            overlay = original_image.copy()
            overlay[predicted_mask.astype(bool)] = cv2.addWeighted(
                overlay[predicted_mask.astype(bool)], 0.5,
                overlay_color, 0.5, 0
            )
            cv2.imwrite(overlay_save_path, overlay)
            logging.info(f"覆盖图像已保存到: {overlay_save_path}")
        else:
            logging.info("由于未生成掩码，跳过掩码和覆盖图像的保存。")

if __name__ == "__main__":
    main()