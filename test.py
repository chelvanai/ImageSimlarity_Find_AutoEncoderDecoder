import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from EncoderDecoder import AutoEncoder

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageFolder(root='./test_images',transform=TRANSFORM)

trainLoader = torch.utils.data.DataLoader(trainset, batch_size=2,
                                          shuffle=True, num_workers=8)


img, _ = next(iter(trainLoader))

img = img.to(DEVICE)

model = AutoEncoder().cuda() if torch.cuda.is_available() else AutoEncoder()
recon_img = model(img)

save_image(recon_img, "generated_img.png")

