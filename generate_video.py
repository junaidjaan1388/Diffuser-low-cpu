from diffusers import DiffusionPipeline
import torch
import os
from PIL import Image
import imageio

def create_video_from_frames(frames, output_path, fps=8):
    """Convert frames to video"""
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(np.array(frame))
    print(f"Video saved as: {output_path}")

# Force CPU usage
print("Loading pipeline...")
pipeline = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b",
    torch_dtype=torch.float32,
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
video_frames = pipeline(
    prompt,
    num_frames=num_frames,
    num_inference_steps=num_inference_steps,
    height=height,
    width=width
).frames

# Create output directory
os.makedirs("output", exist_ok=True)

# Save individual frames
for i, frame in enumerate(video_frames):
    frame.save(f"output/frame_{i:03d}.png")

print(f"Saved {len(video_frames)} frames to output/ directory")

# Create video file (optional)
try:
    import numpy as np
    create_video_from_frames(video_frames, "output/generated_video.mp4")
    print("Video file created successfully!")
except ImportError:
    print("Install imageio and imageio-ffmpeg to create video file")

print("Video generation completed!")
