from torch.utils import data
import gtls
import numpy as np
from episodic_gwr import EpisodicGWR

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn

from avalanche import benchmarks

import torch

tf = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# TODO: try to get ILSVRC-2012 pre-trained
vgg16 = models.vgg16(pretrained=True)
core50 = benchmarks.datasets.core50.CORe50(transform=tf, download=True)

# get pytorch dataset


# data_loader = torch.utils.data.DataLoader(
#     mnist_data, batch_size=2, shuffle=True, transform=tf
# )


class FeatureExtractor(nn.module):
    def __init__(self):
        super().__init__()
        self.fe = vgg16
        self.final_conv = nn.Conv1d(2048, 256, 1)

    def forward(self, x):
        x = self.fe(x)
        x = self.final_conv(x)
        return x


# for each sample:

data_iter = iter(core50)
images, labels = data_iter.next()
print(images[0])

net = FeatureExtractor()
with torch.no_grad():
    outputs = net(images)

    print(outputs)
    print(outputs.shape)

# get pre-trained vgg 16

# run 1x1 conv on output (256 dim fV)

# run through gwr training

# eval with gwr

"""
model = FeatureExtractor+GWR
optimizer = no back propagation
criterion = average_quantization_error?
cl_strategy = Naive?(model)

scenario = Core50(...)

for experience in scenario.train_stream:
    cl_strategy.train(experience)

    results.append(cl_strategy.eval(scenario.test_stream))

"""
