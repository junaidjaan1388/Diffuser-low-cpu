import torch
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline, StableDiffusionPipeline
import os
import cv2
from pathlib import Path

print("🚀 Starting Text-to-Image-to-Video Pipeline...")

# Get parameters
prompt = os.getenv('PROMPT', 'A cute teddy bear waving hello on the moon')
num_frames = int(os.getenv('NUM_FRAMES', '24'))

print(f"📝 Prompt: {prompt}")
print(f"🎞️ Target frames: {num_frames}")

# Step 1: Text-to-Image
print("\n📸 Step 1: Generating base image from text...")
try:
    image_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        dtype=torch.float32,  # Fixed: use dtype instead of torch_dtype
        safety_checker=None,
        requires_safety_checker=False
    )
    image_pipe = image_pipe.to("cpu")
    
    # Generate base image
    base_image = image_pipe(
        prompt,
        num_inference_steps=20,
        height=256,
        width=256
    ).images[0]
    
    base_image.save("base_image.png")
    print("✅ Base image generated: base_image.png")
    
except Exception as e:
    print(f"❌ Error in text-to-image: {e}")
    # Fallback: Create a simple colored image
    base_image = Image.new('RGB', (256, 256), color='lightblue')
    base_image.save("base_image.png")
    print("🔄 Using fallback image")

# Step 2: Image-to-Video with variations
print("\n🎬 Step 2: Creating video from image variations...")

# Create output directory
Path("frames").mkdir(exist_ok=True)

# Generate frame variations
frames = []
base_array = np.array(base_image)

# Ensure base array is uint8
if base_array.dtype == np.float32:
    base_array = (base_array * 255).astype(np.uint8)

for i in range(num_frames):
    try:
        # Create slight variations for animation effect
        frame_array = base_array.copy().astype(np.float32)  # Work in float32 to avoid overflow
        
        # Add subtle transformations to create motion
        if num_frames > 1:
            # Calculate progress through animation (0 to 1 and back)
            progress = (i / (num_frames - 1)) * 2 * np.pi
            
            # Add wave-like transformations
            wave_factor = np.sin(progress) * 3  # Smaller displacement to avoid issues
            
            # Apply affine transformation for subtle movement
            rows, cols = frame_array.shape[:2]
            
            # Create transformation matrix for slight movement
            if i > 0:
                M = np.float32([[1, 0, wave_factor * 0.05], 
                               [0, 1, wave_factor * 0.05]])
                frame_array = cv2.warpAffine(frame_array.astype(np.uint8), M, (cols, rows)).astype(np.float32)
            
            # Add color variations - FIXED VERSION
            hue_shift = np.sin(progress) * 5  # Smaller range to avoid overflow
            
            # Convert to HSV for hue manipulation
            hsv = cv2.cvtColor(frame_array.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Apply hue shift safely
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180  # Hue is in [0, 179] range
            
            # Convert back to RGB
            frame_array = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Convert back to uint8 for saving
        frame_array = frame_array.astype(np.uint8)
        
        # Convert back to PIL and save
        frame_img = Image.fromarray(frame_array)
        frame_path = f"frames/frame_{i:03d}.png"
        frame_img.save(frame_path)
        frames.append(frame_array)
        
        if i % 10 == 0 or i == 0:
            print(f"📊 Generated frame {i+1}/{num_frames}")
            
    except Exception as e:
        print(f"❌ Error generating frame {i}: {e}")
        # Use base frame as fallback
        frame_img = Image.fromarray(base_array)
        frame_path = f"frames/frame_{i:03d}.png"
        frame_img.save(frame_path)
        frames.append(base_array)

# Step 3: Create video from frames
print("\n🎥 Step 3: Creating video file...")

# Get frame dimensions
if frames:
    height, width = frames[0].shape[:2]
    
    # Create video writer
    video_path = "generated_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = min(30, max(10, num_frames // 3))  # Adaptive FPS
    
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    
    print(f"✅ Video generated: {video_path}")
    print(f"📊 Video specs: {width}x{height}, {fps} FPS, {len(frames)} frames")
    print(f"⏱️ Duration: {len(frames)/fps:.2f} seconds")
else:
    print("❌ No frames were generated")

# Create a GIF as well
print("\n🔄 Creating GIF preview...")
try:
    gif_path = "preview.gif"
    
    # Use every Nth frame for GIF to keep file size reasonable
    skip_frames = max(1, len(frames) // 30)
    gif_frames = []
    
    for i in range(0, len(frames), skip_frames):
        img = Image.fromarray(frames[i])
        # Resize for smaller file size
        img_resized = img.resize((128, 128), Image.Resampling.LANCZOS)
        gif_frames.append(img_resized)
    
    # Save as GIF
    if gif_frames:
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=1000//fps,
            loop=0
        )
        print(f"✅ GIF preview: {gif_path}")
    else:
        print("❌ No frames available for GIF")
        
except Exception as e:
    print(f"❌ Error creating GIF: {e}")

print("🎉 Generation completed successfully!")
