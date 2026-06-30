# comfyui-hello-world
* need to run it with GPU or MPS, the CPU is too slow
* image size matters, the big image need more memory, we need to resize the image and upscale it with other models
  * `--max-size 0` means original size, it might have OOM issue
  * `--max-size 512` will give you slightly bad result

# How to run
* run `bash bash scripts/setup.sh`
* run the following command in a terminal
```
python ComfyUI/main.py --use-pytorch-cross-attention --listen 127.0.0.1 --port 8188
```

* run the following command in another terminal

## change image style
* it seems good while changing the image into other style
```
python scripts/run_inference.py \
    --input-dir input_image \
    --output-dir output_images \
    --prompt 'change the image into ghibli style' \
    --nsfw --max-size 0
```

## difficult text removal
* it does not work
```
python scripts/run_inference.py \
    --input-dir input_text_removal \
    --output-dir output_images \
    --prompt 'remove the text 245.10 from the image' --nsfw --max-size 0
```

## combine image
* it does not have good resuls, we need 
  * a better prompt, and 
  * same size of the multiple images (for better results)
  * if the `--max-size` is smaller, we can reduce the size of memory usage
```
python scripts/run_inference.py \
    --images input_cloth_change/a.png input_cloth_change/b.png \
    --output-dir output_images \
    --prompt 'Take the clothing and pants from Picture 1 and dress the person in Picture 2 with them, only change the cloth and pants, other parts (face, background, positioning, resolution, ...) are the same as original Picture 2' \
    --nsfw --max-size 0 --upscale-to-input
```


