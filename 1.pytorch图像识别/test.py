from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
from torch.optim import lr_scheduler
from torch import optim
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import glob,os,shutil
import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(model_path,device):
    model_ft = models.resnet18()
    model_ft.fc = nn.Linear(model_ft.fc.in_features, 2)
    model_ft.load_state_dict(torch.load(model_path))
    model_ft.to(device)
    model_ft.eval()

    return model_ft
model_ft = load_model("best_model.pth",device)

image_transforms =  transforms.Compose([
                    transforms.Resize(size=256),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                ])

img_path = ""
frame = Image.open(img_path)
frame = image_transforms(frame).unsqueeze(0)
frame = frame.to(device)
output = model_ft(frame)
# print(output.data)
preds = torch.max(output.data, 1)[-1].item()
print(preds)

