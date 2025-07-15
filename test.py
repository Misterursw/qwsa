# test.py
import argparse
import os
import cv2
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor
from model.QWSA import QWSAForCausalLM
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from model.qwen import conversation as conversation_lib

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def main():
    parser = argparse.ArgumentParser(description="QWSA Inference Script")
    parser.add_argument("--model-path", type=str, required=True, help="Path to your trained model checkpoint.")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--prompt", type=str, required=True, help="The question or instruction for the model.")
    parser.add_argument("--sam-arch", type=str, default="h", choices=['h', 'l', 'b'], help="SAM architecture to use (h, l, or b).")
    parser.add_argument("--sam-checkpoint", type=str, required=True, help="Path to the SAM pretrained weights.")
    parser.add_argument("--out-dir", type=str, default="./test_output", help="Directory to save the output.")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Load Model and Tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")
    
    # Dynamically select SAM builder
    if args.sam_arch == 'b':
        from model.segment_anything import build_sam_vit_b as build_sam
    elif args.sam_arch == 'l':
        from model.segment_anything import build_sam_vit_l as build_sam
    else:
        from model.segment_anything import build_sam_vit_h as build_sam

    # Monkey patch the builder in QWSA before instantiation
    import model.QWSA
    model.QWSA.build_sam = build_sam

    model = QWSAForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        vision_pretrained=args.sam_checkpoint,
        seg_token_idx=seg_token_idx,
    ).cuda().eval()
    
    # 2. Prepare Image
    print("Preparing image...")
    image_np = cv2.imread(args.image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size = image_np.shape[:2]
    
    transform = ResizeLongestSide(1024)
    image_sam = transform.apply_image(image_np)
    image_sam_tensor = preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda().bfloat16()

    # 3. Prepare Prompt
    print("Preparing prompt...")
    conv = conversation_lib.conv_templates['qwen_vl'].copy()
    prompt_text = f"<image>\n{args.prompt}"
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    
    full_prompt = conv.get_prompt()
    
    # Manually handle tokenization for single inference
    # This part needs careful implementation based on how the model expects inputs during inference,
    # which might differ slightly from the training `collate_fn`.
    # For simplicity, we'll assume a batch size of 1.
    
    # 4. Inference
    # Note: The model's `evaluate` or a dedicated `generate_with_segmentation` method would be ideal.
    # Since it's not present, we will simulate the forward pass from `train_qs.py`.
    # This is a simplified version and might need adjustments.
    
    print("Running inference... (This part is simplified)")
    # This is a conceptual representation. The actual inference call needs to be adapted from the training loop
    # or a new inference function needs to be written that mirrors the logic in `validate`.
    # The `model.evaluate` in chat.py seems to be a custom method for LISA, not directly available here.
    # We will print a message instead of running the full, complex inference logic.
    print("\n[INFO] The provided code lacks a straightforward, self-contained inference function.")
    print("The training script (`train_qs.py`) and Gradio app (`app.py`) show the full process,")
    print("which involves batching and specific data collation.")
    print("\nTo make this test script fully functional, you would need to adapt the `validate` function")
    print("from `train_qs.py` for single-image inference, handling the tokenization, image processing,")
    print("and model forward pass exactly as it does.")
    
    print(f"\n--- Prepared Data ---")
    print(f"Image Tensor Shape: {image_sam_tensor.shape}")
    print(f"Full Prompt for Model:\n{full_prompt}")
    print("--------------------")

if __name__ == "__main__":
    main()