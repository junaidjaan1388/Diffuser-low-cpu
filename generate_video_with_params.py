import os
from diffusers import DiffusionPipeline
import torch

# Get parameters from environment
prompt = os.getenv('PROMPT', 'A cute cat is walking on the moon')
num_frames = int(os.getenv('FRAMES', '24'))
num_inference_steps = int(os.getenv('STEPS', '10'))

print(f"Generating video with parameters:")
print(f"Prompt: {prompt}")
print(f"Frames: {num_frames}")
print(f"Steps: {num_inference_steps}")

# Your video generation code here...
