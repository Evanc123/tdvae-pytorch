import numpy as np
import torch
from torchvision import datasets, transforms
import torch.utils.data
import random
train = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=1, shuffle=True)

def make_dataset(dataset_len, sequence_len):
    data = np.zeros((dataset_len, sequence_len, 28, 28))
    i = 0
    for _, (dat, _) in enumerate(train):
        frame = dat[0][0].cpu().numpy()
        data[i][0] = frame
        for j in range(1, sequence_len):
            if random.randint(0, 1) == 1:
                frame = np.roll(frame, 1, axis=1)
                data[i][j] = np.copy(frame)
            else:
                frame = np.roll(frame, -1, axis=1)
                data[i][j] = np.copy(frame)
        i += 1
        if i == dataset_len:
            break
    return data


