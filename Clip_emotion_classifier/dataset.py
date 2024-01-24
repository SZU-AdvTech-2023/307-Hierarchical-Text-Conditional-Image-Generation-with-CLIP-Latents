'''
Date: 2023-01-10 10:16:14 
Author: ttd & dingt6616@gmail.com
EmoSet 自定义Dataset——>单标签和多标签的形式
'''
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import os
import json


class EmoSet(Dataset):
    def __init__(self, data_root, mode, model, processor):
        self.data_root = data_root
        self.model = model
        self.processor = processor
        assert mode.lower() in ('train', 'val', 'test')
        self.mode = mode
        # 属性的字典
        self.attr_type = json.load(open(os.path.join(self.data_root, 'info.json'),'r'))
        # 选择的模式对应的图片数量
        data_store = json.load(open(os.path.join(data_root, f'{mode}.json')))
        self.data_store = [
            [
                os.path.join(data_root, item[2]),
                os.path.join(data_root, item[3])
            ]
            for item in data_store
        ]
    def get_criterion(self,annotation_path):
        # 'emotion', 'brightness', 'colorfulness', 'scene', 'object', 'facial_expression', 'human_action' 7个属性
        image_json = json.load(open(os.path.join(self.data_root, annotation_path)))
        list_label = {}
        for tp in self.attr_type:
            if tp in image_json:
                if tp == 'object':
                    label = []
                    for ob in image_json[tp]:
                        label2idx = self.attr_type[tp]['label2idx']
                        ll = label2idx[ob]
                        label.append(ll)
                    # if len(label)!=3:
                    #     label = label + [-1]*int(3-len(label))
                else:    
                    label2idx = self.attr_type[tp]['label2idx']
                    label = label2idx[str(image_json[tp])]
            else:
                label = -1
            list_label[tp] = label
        return list_label

    def get_image_by_path(self, path):
        image = Image.open(path).convert('RGB')
        data = self.processor(images=image, return_tensors="pt", padding=True).to(self.model.device)
        data['pixel_values'] = data['pixel_values'].squeeze(0)
        # data = self.model.get_image_features(**data)
        return data
    
    def split_object(self, dic):
        if dic['object']==-1:
            dic['object'] = torch.zeros(409) 
        else:
            id = dic['object']
            dic['object'] = torch.zeros(409)
            dic['object'][id] = 1
        return dic
            

    def __getitem__(self,item):
        image_path, annotation_path = self.data_store[item]
        dic = self.get_criterion(annotation_path)
        dic['image'] = self.get_image_by_path(image_path)
        dic = self.split_object(dic)
        
        return dic

    def __len__(self):
        return len(self.data_store)


# data_transform = {
#         "train": transforms.Compose([transforms.RandomResizedCrop(224),
#                                      transforms.RandomHorizontalFlip(),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
#         "val": transforms.Compose([transforms.Resize(256),
#                                    transforms.CenterCrop(224),
#                                    transforms.ToTensor(),
#                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

# if __name__ == '__main__':
#     train_dataset = EmoSet('/home/qirui/dataset/EmoSet_v5/', 'train', data_transform["train"])
#     train = DataLoader(train_dataset, batch_size=6, shuffle=True, pin_memory=True, num_workers=4,)
#     x = next(iter(train))
#     print(x)
