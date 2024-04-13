import sys 
import os
from datasets import load_from_disk
from io import BytesIO
from PIL import Image
from traceback import print_exc
import torch
from torchvision.models import resnet18, ResNet18_Weights

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('dataset_path')

args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

dataset = load_from_disk(args.dataset_path)

aligned_key = "image"

arch = "resnet18"
weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()
model = resnet18()

model.to(device)

output_key = arch
output_status_key = "status_" + arch


def embedder(sample):
    try:     
        # Loading the aligned image
        img = preprocess(Image.open(BytesIO(sample[aligned_key])).convert('RGB'))

        print(img.shape)
        img = img.unsqueeze(0)

        img = img.to(device)
        feats = model(img)

        feats = feats.detach().to('cpu')
        
        #print(feats.shape)
        sample[output_status_key] = True
        sample[output_key] = feats[0,:]

        del img, feats

    except: 
        print(f"Failed for sample ")
        print_exc()

        sample[output_status_key] = False
        sample[output_key] = torch.zeros(512, dtype=torch.float32)

    return sample 

dataset = dataset.map(embedder, writer_batch_size=1000)

dataset.save_to_disk( args.dataset_path[:-3] + f"_{arch}.hf")
