from diffusers import DiffusionPipeline
import torch
import os
from PIL import Image
import numpy as np

print("Loading pipeline...")
pipeline = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b",
    dtype=torch.float32,  # Changed from torch_dtype to dtype
    device_map="cpu"
)

# Generation parameters
prompt = "A cute cat is walking on the moon"
num_frames = 24
num_inference_steps = 10
height = 240
width = 320

print("Generating video...")
print(f"Prompt: {prompt}")
print(f"Frames: {num_frames}, Steps: {num_inference_steps}")

# Generate video frames
result = pipeline(
    prompt,
    num_frames=num_frames,
    num_inference_steps=num_inference_steps,
    height=height,
    width=width
)

# The frames are returned as numpy arrays, not PIL images
video_frames = result.frames

# Create output directory
os.makedirs("output", exist_ok=True)

# Convert numpy arrays to PIL Images and save
for i, frame in enumerate(video_frames):
    # Convert numpy array to PIL Image
    if isinstance(frame, np.ndarray):
        # Ensure the array is in the correct format (H, W, C)
        if frame.dtype == np.float32:
            # Convert from float [0,1] to uint8 [0,255]
            frame = (frame * 255).astype(np.uint8)
        pil_image = Image.fromarray(frame)
        pil_image.save(f"output/frame_{i:03d}.png")
    else:
        # If it's already a PIL Image (shouldn't happen with this pipeline)
        frame.save(f"output/frame_{i:03d}.png")

print(f"Saved {len(video_frames)} frames to output/ directory")

# Try to create video if imageio is available
try:
    import imageio
    print("Creating video file...")
    
    # Create video from frames
    with imageio.get_writer("output/generated_video.mp4", fps=8) as writer:
        for frame in video_frames:
            if isinstance(frame, np.ndarray):
                # Ensure frame is uint8
                if frame.dtype == np.float32:
                    frame = (frame * 255).astype(np.uint8)
                writer.append_data(frame)
            else:
                # Convert PIL to numpy
                writer.append_data(np.array(frame))
    
    print("Video saved as: output/generated_video.mp4")
    
except ImportError:
    print("imageio not available - skipping video creation")

print("Video generation completed!")
