import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
import os
import cv2
from pathlib import Path

print("🚀 Starting Simplified Text-to-Image-to-Video Pipeline...")

prompt = os.getenv('PROMPT', 'A cute teddy bear waving hello on the moon')
num_frames = int(os.getenv('NUM_FRAMES', '24'))

print(f"📝 Prompt: {prompt}")
print(f"🎞️ Target frames: {num_frames}")

# Step 1: Generate base image
print("\n📸 Generating base image...")
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        dtype=torch.float32,
        safety_checker=None
    ).to("cpu")
    
    base_image = pipe(prompt, num_inference_steps=10, height=256, width=256).images[0]
    base_image.save("base_image.png")
    print("✅ Base image saved")
    
except Exception as e:
    print(f"❌ Using fallback image: {e}")
    base_image = Image.new('RGB', (256, 256), color='lightblue')
    base_image.save("base_image.png")

# Step 2: Create simple zoom animation
print("\n🎬 Creating animation...")
Path("frames").mkdir(exist_ok=True)

base_array = np.array(base_image)
frames = []
height, width = base_array.shape[:2]

for i in range(num_frames):
    progress = i / max(1, num_frames - 1)
    
    # Simple zoom effect
    zoom = 1.0 + (progress * 0.2)  # 0-20% zoom
    
    # Calculate crop dimensions
    new_width = int(width / zoom)
    new_height = int(height / zoom)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    
    # Crop and resize
    cropped = base_array[top:top+new_height, left:left+new_width]
    resized = cv2.resize(cropped, (width, height))
    
    frames.append(resized)
    
    # Save frame
    Image.fromarray(resized).save(f"frames/frame_{i:03d}.png")
    
    if i % 10 == 0:
        print(f"📊 Frame {i+1}/{num_frames}")

# Step 3: Create video
print("\n🎥 Creating video...")
video_path = "generated_video.mp4"
fps = 15
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

out.release()
print(f"✅ Video saved: {video_path}")
print("🎉 Done!")
