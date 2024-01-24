import sys

import torch
import os
import torch.nn as nn
from model import *
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, UniPCMultistepScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import argparse
import random
from accelerate.utils import set_seed
from torch.utils.data import Dataset
from torchvision import transforms
import pickle
import torch.nn.functional as F

@torch.no_grad()
def count_relate(img, model, processor):
    with open(f'./dataset_balance/all_attribute_object_scene.pkl', 'rb') as f:
        attribute_total = pickle.load(f)
    data_pro = processor(images=img, text=attribute_total, return_tensors="pt", padding=True).to(model.device)
    data_pro = model(**data_pro)
    score = data_pro.logits_per_image.squeeze(0)
    indice = torch.argmax(score, dim=0)
    relate_semantic = attribute_total[indice.item()]
    relate_score = score[indice.item()].item()
    # values, indices = torch.topk(score, 10)
    # for i in range(indices.shape[0]):
    #     y = indices[i].item()
    #     k_semantic = attribute_total[y]

    return relate_semantic, relate_score


@torch.no_grad()
def emo_cls(cur_dir, device, weight):
    classifier = emo_classifier_768().to(device)
    state = torch.load(weight, map_location=device)
    classifier.load_state_dict(state)
    classifier.eval()

    CLIPmodel = CLIPModel.from_pretrained("/mnt/d/model/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("/mnt/d/model/clip-vit-large-patch14")

    class EmoDataset(Dataset):
        def __init__(self, data_root, processor):
            self.emotion_list_8 = {"amusement": 0,
                                   "awe": 1,
                                   "contentment": 2,
                                   "excitement": 3,
                                   "anger": 4,
                                   "disgust": 5,
                                   "fear": 6,
                                   "sadness": 7}
            self.emotion_list_2 = {"amusement": 0,
                                   "awe": 0,
                                   "contentment": 0,
                                   "excitement": 0,
                                   "anger": 1,
                                   "disgust": 1,
                                   "fear": 1,
                                   "sadness": 1
                                   }
            self.tfm = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            self.image_paths = []
            self.processor = processor
            self.data_root = data_root
            for root, _, file_path in os.walk(self.data_root):
                for file in file_path:
                    if file.endswith("jpg"):
                        self.image_paths.append(os.path.join(root, file))
            self._length = len(self.image_paths)

        def __len__(self):
            return self._length

        def __getitem__(self, i):
            path = self.image_paths[i]
            example = {}
            image = Image.open(path).convert('RGB')
            data = self.processor(images=image, return_tensors="pt", padding=True)
            data['pixel_values'] = data['pixel_values'].squeeze(0)
            example['image'] = data
            # data = self.model.get_image_features(**data)
            example['emotion_8'] = self.emotion_list_8[path.split('/')[-2]]
            example['emotion_2'] = self.emotion_list_2[path.split('/')[-2]]
            return example

    val_dataset = EmoDataset(cur_dir, processor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)
    picture_num = len(val_dataset)
    val_loader = tqdm(val_loader, file=sys.stdout)
    score_8 = 0
    score_2 = 0
    acc_num_2 = 0
    acc_num_8 = 0

    def eightemotion(Emo, Emo_num, Emo_score, pre, label, correct):

        for i in range(label.shape[0]):
            emo_label = label[i][0].item()
            Emo[emo_label] += correct[i].item()
            Emo_num[emo_label] += 1
            Emo_score[emo_label] += pre[i][emo_label]
        return Emo, Emo_num, Emo_score

    Emo = [0] * 8
    Emo_num = [0] * 8
    Emo_score = [0] * 8
    Emotion = ["amusement", "awe", "contentment",
               "excitement",
               "anger",
               "disgust",
               "fear",
               "sadness"
               ]
    for step, data in enumerate(val_loader):
        images = data['image'].to(device)
        clip = CLIPmodel.get_image_features(**images)
        pred = classifier(clip.to(device))
        labels_8 = data['emotion_8'].to(device).unsqueeze(1)
        labels_2 = data['emotion_2'].to(device).unsqueeze(1)
        pred_emotion_8 = torch.argmax(pred, dim=1, keepdim=True)
        p_8 = F.softmax(pred)
        p_2 = p_8.reshape((p_8.shape[0], 2, 4))
        p_2 = torch.sum(p_2, dim=2)
        p_2 = p_2.reshape((p_8.shape[0], -1))

        pred_emotion_2 = torch.argmax(p_2, dim=1, keepdim=True)

        pred_score_8 = torch.gather(p_8, dim=1, index=labels_8)
        pred_score_2 = torch.gather(p_2, dim=1, index=labels_2)

        acc_num_2 += (labels_2 == pred_emotion_2).sum().item()
        score_2 += torch.sum(pred_score_2).item()
        acc_num_8 += (labels_8 == pred_emotion_8).sum().item()
        score_8 += torch.sum(pred_score_8).item()
        eightemotion(Emo, Emo_num, Emo_score, p_8, labels_8, (labels_8 == pred_emotion_8))
    acc_8 = (acc_num_8 / picture_num) * 100
    total_score_8 = score_8 / picture_num
    acc_2 = (acc_num_2 / picture_num) * 100
    total_score_2 = score_2 / picture_num
    # print(cur_dir + "\n")
    # print(f"{acc_8:.2f}" + "\n")
    with open(os.path.join(cur_dir, 'evaluation.txt'), "a") as f:
        f.write(f'emo_score (8 class): {total_score_8:.2f}' + '\n')
        f.write(f'accuracy (8 class): {acc_8:.2f}%' + '\n')
        f.write(f'emo_score (2 class): {total_score_2:.2f}' + '\n')
        f.write(f'accuracy (2 class): {acc_2:.2f}%' + '\n')
        for i in range(8):
            tmp = Emo[i] / Emo_num[i] * 100
            f.write(f'{Emotion[i]} accuracy:{tmp:.2f}% score:{(Emo_score[i]/Emo_num[i]):.2f} \n')

if __name__ == "__main__":
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    weight = "../Clip_emotion_classifier/weights/2023-11-12-best.pth"
    files = [
        "/home/ubuntu/code/emo_generation-unclip/img/2023-12-29_07-37"
             ]
    for file in files:
        emo_cls(file, device, weight)
