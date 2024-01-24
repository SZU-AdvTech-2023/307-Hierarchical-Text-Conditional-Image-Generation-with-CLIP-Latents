import datetime

from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
import os

now = datetime.datetime.now()
time_str = now.strftime('%Y-%m-%d_%H-%M')
num_picture = 5
per_save_step = 1
embedding_folder = 'record/2023-12-29_01-05/dictionary/'
picture_per_step = num_picture // per_save_step
# emotion_list = ["sadness"]
emotion_list = ["amusement", "excitement", "awe", "contentment", "fear", "disgust", "anger", "sadness"]

pipeline = DiffusionPipeline.from_pretrained("/mnt/d/model/stable-diffusion-v1-5/", torch_dtype=torch.float16, use_safetensors=False).to(
    "cuda:4")

for emotion in emotion_list:
    files = sorted([x for x in os.listdir(embedding_folder) if x.startswith(f"{emotion}")], key=lambda x: int(x.split(".")[0].split("_")[1]))
    if len(files) == 0:
        continue
    num = int(files[-1].split(".")[0].split("_")[1])
    for file in files:
        pipeline.load_textual_inversion(f"{embedding_folder}/{file}")
    for i in range(0, num+1):
        for _ in range(per_save_step):
            images = []
            for _ in range(picture_per_step):
                with torch.autocast("cuda"):
                    image = pipeline(f"Professional high-quality art of a <{emotion}_{i}>. photorealistic, 4k, HQ", num_inference_steps=50).images[0]
                images.append(image)
            os.makedirs(f"img/{time_str}/{emotion}", exist_ok=True)
            for image in images:
                try:
                    files = sorted([x for x in os.listdir(f"img/{time_str}/{emotion}") if x.endswith(".jpg")],
                                   key=lambda x: int(x.split(".")[0].split('_')[-1]))
                    num = int(files[-1].split(".")[0].split('_')[-1])
                    image.save(f"img/{time_str}/{emotion}/{emotion}_{i}_{num + 1}.jpg")
                except:
                    image.save(f"img/{time_str}/{emotion}/{emotion}_{i}_0.jpg")

with open(f'img/{time_str}/config.txt', 'w') as file:
    file.write(f'embedding folder: {embedding_folder}\n')
    file.write(f'model: stable-diffusion-v1-5\n')