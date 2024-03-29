import os
import sys
import json
import pickle
import random
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
def train_one_epoch(model, optimizer, data_loader, device, epoch, lable_mode, CLIPmodel):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
   
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        labels = data[lable_mode].to(device,dtype=torch.long)
        sample_num += len(labels)
        images = data['image']
        clip = CLIPmodel.get_image_features(**images)
        pred = model(clip.to(device))
        acc = torch.eq(torch.max(pred,1)[1],labels).sum().item()
        accu_num += acc

        loss = loss_function(pred, labels)
        loss.backward()  # -1报错
        accu_loss += loss.detach()
        
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch+1,
                                        accu_loss.item() / (step + 1),
                                        accu_num.item() / sample_num)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
# 

@torch.no_grad()
def evaluate(model, data_loader, device, epoch, lable_mode, CLIPmodel):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        labels = data[lable_mode].to(device,dtype=torch.long)
        sample_num += len(labels)
        images = data['image']
        clip = CLIPmodel.get_image_features(**images)
        pred = model(clip.to(device))
        
        acc = (torch.max(pred,1)[1] == labels).sum().item()
        accu_num += acc

        loss = loss_function(pred, labels)

        accu_loss += loss

        data_loader.desc = "[test epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch+1,
                        accu_loss.item() / (step + 1),
                        accu_num.item() / sample_num)
    
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
