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
* The cloth plays an important role here
  * Even we asked to change the person, not just cloth, the cloth still play a big role
```
python scripts/run_inference.py \
    --images input_cloth_change/a.png input_cloth_change/b.png \
    --output-dir output_images \
    --prompt 'Take the clothing and pants from Picture 1 and dress the person in Picture 2 with them, only change the cloth and pants, other parts (face, background, positioning, resolution, ...) are the same as original Picture 2' \
    --nsfw --max-size 0 --upscale-to-input
```

# Qwen prompt tips for image editing
* facial 
```
Remove the double chin and excess fat under the jawline of the girl. Make the jawline look tight, smooth, and natural, perfectly blending with the original face shape. Do not change other facial features or the background.
```

* belly
```
Smooth out the lower abdomen area to make the stomach look flat and fit. Keep the original clothing, skin texture, lighting, and background completely unchanged. Only adjust the waistline for a natural, slim look.
```

* change cloth
```
Core Task: Transfer the exact outfit from Picture 1 onto the person in Picture 2.Clothing Details from Picture 1: [A dark brown cropped halter-style tank top featuring multi-strap criss-cross shoulder details and a fitted crop silhouette exposing the midriff.] from Picture 1.Preservation Constraints: Keep the person face, hair, body shape, posture, expression, and the entire background from Picture 2 completely identical and unchanged.Execution: Seamlessly replace the clothing in Picture 2 with the clothing from Picture 1, matching the lighting, shadows, and perspective of Picture 2.
```

* change person
```
Core Task: Seamlessly replace the girl in Picture 2 with the exact slim girl from Picture 1.Subject to Transfer (from Picture 1): Transfer the entire slim girl from Picture 1, including her exact face, facial features, dark hair with bangs, slim body shape, skin tone, and the clothing (dark brown multi-strap halter crop top).Environment to Keep (from Picture 2): Keep the entire background, lighting, environment, foreground elements, and setting from Picture 2 completely unchanged.Execution: Place the slim girl from Picture 1 into the scene of Picture 2. Blend the lighting, shadows, and perspective naturally so she looks like she was originally there in the environment of Picture 2.
```

* change the style of person only
```
Core Task: Transform ONLY the person in the image into a 2D Studio Ghibli anime style, while keeping the background completely unchanged.Subject Style: Redraw the woman in a hand-drawn Ghibli anime aesthetic. Her face, expression, hair with bangs, and clothing (dark brown multi-strap halter top) should be rendered with soft line art, clean anime shading, and gentle facial features typical of a Ghibli protagonist. Maintain her original pose, posture, and actions.Background Preservation: The entire background must remain exactly as they are in the original photo. Do not stylize or alter the environment.Rendering: Seamlessly blend the Ghibli-styled anime character into the real-world photo environment with natural lighting and shadows.
```

