
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from IPython.display import Video, display
import gc

print("🚀 Loading text-to-video model on CPU...")

# Use CPU
device = "cpu"
torch_dtype = torch.float32  # Use float32 for CPU

try:
    pipe = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b",
        torch_dtype=torch_dtype,
    )
    pipe = pipe.to(device)
    
    print("✅ Model loaded successfully on CPU!")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # Try without dtype specification
    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b")
    pipe = pipe.to(device)

# Generate video with CPU-friendly settings
prompt = "A cute teddy bear waving hello"
print(f"🎬 Generating video: {prompt}")

# Use fewer steps and frames for CPU
video_frames = pipe(
    prompt,
    num_inference_steps=20,  # Reduced for CPU
    num_frames=8,           # Fewer frames
    height=128,             # Lower resolution
    width=128,
).frames[0]

# Save video
video_path = "cpu_generated_video.mp4"
export_to_video(video_frames, video_path)
print(f"✅ Video saved: {video_path}")

# Display video
display(Video(video_path, embed=True))

print("🎉 Video generated successfully on CPU!")
