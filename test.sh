python test.py \
    --model-path "/path/to/your/finetuned/qwsa-3b/checkpoint_best" \
    --image-path "/path/to/your/test_image.jpg" \
    --prompt "Please segment the person in the foreground." \
    --sam-arch "b" \
    --sam-checkpoint "/path/to/your/sam_vit_b_01ec64.pth" \
    --out-dir "./test_output"