from diffusers import DiffusionPipeline, StableDiffusionPipeline
import torch
import numpy as np
from PIL import Image
import os
import cv2

print("🚀 Quick Text-to-Video Generator")

prompt = os.getenv('PROMPT', 'A cute cat on the moon')
num_frames = 16

# Generate image
print("📸 Generating image...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float32
).to("cpu")

image = pipe(prompt, num_inference_steps=15).images[0]
image.save("output/base_image.png")

# Create simple animation
print("🎬 Creating animation...")
img_array = np.array(image)
frames = []

for i in range(num_frames):
    progress = i / num_frames
    # Simple zoom effect
    zoom = 1.0 + progress * 0.2
    h, w = img_array.shape[:2]
    new_h, new_w = int(h/zoom), int(w/zoom)
    left, top = (w - new_w) // 2, (h - new_h) // 2
    cropped = img_array[top:top+new_h, left:left+new_w]
    resized = cv2.resize(cropped, (w, h))
    frames.append(resized)

# Save video
print("💾 Saving video...")
height, width = frames[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/quick_video.mp4', fourcc, 8, (width, height))

for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

out.release()
print("✅ Done! Check output/ folder")
