import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from EncoderDecoder import AutoEncoder

__DEBUG__ = True
LOAD = True
PATH = "./autoencoder.pth"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

EPOCHS = 4
BATCH_SIZE = 4


def get_dataset(train=True):
    trainset = torchvision.datasets.STL10(root='../dataset',
                                          download=True, transform=TRANSFORM)
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=8)
    return trainLoader


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def showRandomImaged(train):
    # get some random training images
    dataiter = iter(train)
    images, labels = dataiter.next()

    # show images
    print(images.shape)
    imshow(torchvision.utils.make_grid(images))


if __name__ == '__main__':
    if __DEBUG__ == True:
        print(DEVICE)
    train = get_dataset()
    if False:
        print("Showing Random images from dataset")
        showRandomImaged(train)

    model = AutoEncoder().cuda() if torch.cuda.is_available() else AutoEncoder()
    if __DEBUG__ == True:
        print(model)

    criterian = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

    if LOAD == True:
        model.load_state_dict(torch.load(PATH))

    for epoch in range(EPOCHS):
        for i, (images, _) in enumerate(train):
            images = images.to(DEVICE)
            out = model(images)
            loss = criterian(out, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ## LOG
            print(f"epoch {epoch}/{EPOCHS}\nLoss : {loss.data}")

            if __DEBUG__ == True:
                if i % 10 == 0:
                    out = out / 2 + 0.5  # unnormalize
                    img_path = "debug_img" + str(i) + ".png"
                    save_image(out, img_path)

        # SAVING
        torch.save(model.state_dict(), PATH)

    # test = get_dataset(train=False)
    # img, _ = next(iter(test))
    # img = img.to(DEVICE)
    # recon_img = model(img)
    # save_image(recon_img, "generated_img.png")
