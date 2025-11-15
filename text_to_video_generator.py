import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline
import os
import cv2
from pathlib import Path
import math
import imageio

print("🚀 Starting Text-to-Image-to-Video Generator...")

# Configuration
prompt = os.getenv('PROMPT', 'A cute cat walking on the moon with stars')
num_frames = int(os.getenv('NUM_FRAMES', '24'))
duration = int(os.getenv('DURATION', '3'))
fps = max(8, num_frames // duration)  # Calculate FPS based on duration

print(f"📝 Prompt: {prompt}")
print(f"🎞️ Frames: {num_frames}")
print(f"⏱️ Duration: {duration}s")
print(f"📊 FPS: {fps}")

# Create output directory
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
(output_dir / "frames").mkdir(exist_ok=True)

def generate_base_image(prompt, width=512, height=512):
    """Generate base image from text prompt"""
    print("\n📸 Step 1: Generating base image from text...")
    
    try:
        # Use a smaller model for faster generation
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to("cpu")
        
        # Generate image
        image = pipe(
            prompt,
            num_inference_steps=20,
            height=height,
            width=width,
            guidance_scale=7.5
        ).images[0]
        
        image_path = output_dir / "base_image.png"
        image.save(image_path)
        print(f"✅ Base image saved: {image_path}")
        return image
        
    except Exception as e:
        print(f"❌ Error generating image: {e}")
        print("🔄 Creating fallback image...")
        # Create a simple fallback image
        image = Image.new('RGB', (width, height), color='navy')
        draw = ImageDraw.Draw(image)
        
        # Draw moon and stars
        draw.ellipse([150, 150, 350, 350], fill='lightgray', outline='white')
        for i in range(50):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            size = np.random.randint(1, 3)
            draw.ellipse([x, y, x+size, y+size], fill='white')
        
        image_path = output_dir / "base_image.png"
        image.save(image_path)
        return image

def create_animated_video(base_image, num_frames, fps):
    """Create animated video from base image"""
    print("\n🎬 Step 2: Creating animated video...")
    
    width, height = base_image.size
    base_array = np.array(base_image)
    frames = []
    
    # Animation parameters
    for i in range(num_frames):
        progress = i / max(1, num_frames - 1)
        frame = base_array.copy().astype(np.float32)
        
        # Apply multiple animation effects
        
        # 1. Zoom effect
        zoom = 1.0 + math.sin(progress * math.pi) * 0.1
        
        # 2. Pan effect
        pan_x = math.sin(progress * 2 * math.pi) * 20
        pan_y = math.cos(progress * math.pi) * 10
        
        # 3. Color shift
        color_shift = math.sin(progress * 4 * math.pi) * 10
        
        # Apply transformations
        M = np.float32([[1, 0, pan_x], [0, 1, pan_y]])
        transformed = cv2.warpAffine(frame.astype(np.uint8), M, (width, height))
        
        # Apply zoom (crop and resize)
        if zoom != 1.0:
            new_width = int(width / zoom)
            new_height = int(height / zoom)
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            cropped = transformed[top:top+new_height, left:left+new_width]
            transformed = cv2.resize(cropped, (width, height))
        
        # Apply color shift
        hsv = cv2.cvtColor(transformed, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + color_shift) % 180
        transformed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Add frame counter and progress
        img_pil = Image.fromarray(transformed)
        draw = ImageDraw.Draw(img_pil)
        
        # Add info text
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        text = f"Frame {i+1}/{num_frames}"
        draw.text((10, 10), text, fill='white', font=font)
        draw.text((10, 30), prompt[:30] + "...", fill='white', font=font)
        
        # Progress bar
        bar_width = width - 20
        bar_height = 8
        draw.rectangle([10, height-20, 10 + bar_width, height-20 + bar_height], fill='#333333')
        draw.rectangle([10, height-20, 10 + int(bar_width * progress), height-20 + bar_height], fill='#00ff00')
        
        frame_array = np.array(img_pil)
        frames.append(frame_array)
        
        # Save individual frame
        frame_path = output_dir / "frames" / f"frame_{i:03d}.png"
        img_pil.save(frame_path)
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"📊 Generated frame {i+1}/{num_frames}")
    
    return frames

def create_mp4_video(frames, fps, output_path):
    """Create MP4 video from frames"""
    print(f"\n🎥 Creating MP4 video ({fps} FPS)...")
    
    if not frames:
        print("❌ No frames to create video")
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"✅ MP4 video saved: {output_path}")

def create_gif_preview(frames, output_path, fps=8):
    """Create GIF preview from frames"""
    print("\n🔄 Creating GIF preview...")
    
    if not frames:
        print("❌ No frames to create GIF")
        return
    
    # Use every Nth frame for reasonable file size
    skip_frames = max(1, len(frames) // 15)
    gif_frames = []
    
    for i in range(0, len(frames), skip_frames):
        img = Image.fromarray(frames[i])
        # Resize for smaller file size
        img_resized = img.resize((320, 240), Image.Resampling.LANCZOS)
        gif_frames.append(img_resized)
    
    if gif_frames:
        gif_frames[0].save(
            str(output_path),
            save_all=True,
            append_images=gif_frames[1:],
            duration=1000//fps,
            loop=0,
            optimize=True
        )
        print(f"✅ GIF preview saved: {output_path}")

def create_download_info():
    """Create a README file with download instructions"""
    readme_content = f"""
# Generated Video Assets

## Prompt: {prompt}
- Frames: {num_frames}
- Duration: {duration}s
- FPS: {fps}

## Generated Files:
1. `generated_video.mp4` - Main video file
2. `preview.gif` - Animated GIF preview
3. `base_image.png` - Source image used for animation
4. `frames/` - Directory containing all individual frames

## Download Instructions:
1. Click on the 'generated-video-assets' artifact above
2. Download the ZIP file
3. Extract to view your generated content

## Usage:
- Use the MP4 for high-quality video
- Use the GIF for quick previews or sharing
- Use individual frames for frame-by-frame analysis

Generated on: {os.environ.get('GITHUB_SERVER_URL', 'GitHub')}
    """.strip()
    
    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)

# Main execution
try:
    # Step 1: Generate base image
    base_image = generate_base_image(prompt)
    
    # Step 2: Create animated frames
    frames = create_animated_video(base_image, num_frames, fps)
    
    # Step 3: Create video files
    create_mp4_video(frames, fps, output_dir / "generated_video.mp4")
    create_gif_preview(frames, output_dir / "preview.gif")
    
    # Step 4: Create download info
    create_download_info()
    
    # Final summary
    print("\n🎉 Generation Completed Successfully!")
    print("📁 Generated Files:")
    print(f"   • generated_video.mp4 ({fps} FPS, {duration}s)")
    print(f"   • preview.gif (animated preview)")
    print(f"   • base_image.png (source image)")
    print(f"   • frames/ ({num_frames} individual frames)")
    print(f"   • README.md (usage instructions)")
    print("\n📥 Download from the Artifacts section above!")
    
except Exception as e:
    print(f"❌ Error in main execution: {e}")
    raise
