# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")



# from huggingface_hub import snapshot_download
# snapshot_download(
#     repo_id="liuhaotian/llava-llama-2-13b-chat-lightning-preview",
#     local_dir="/home/ubuntu/LISA-main/model/llava-llama-2-13b",
#     cache_dir="/home/ubuntu/LISA-main/model/cache",
#     allow_patterns=["*.bin", "*.json", "*.model", "*.py"],
#     ignore_patterns=["*.safetensors", "*.h5", "*.msgpack"],
#     resume_download=True
# )

