from torch.utils import data
#import gtls
import numpy as np
#from episodic_gwr import EpisodicGWR

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

model = models.vgg16(pretrained=True)
core50 = benchmarks.datasets.core50.CORe50(transform=tf, download=True)



class FeatureExtractor(nn.Module):
    def __init__(self,model):
        super(FeatureExtractor, self).__init__()
 #VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        #VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        #Image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        self.fc = model.classifier[0]
        
        #additional 
        self.last = nn.Linear(4096, 256)

  
    def forward(self, x):

        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out) 
        

        out = self.last(out)
        return out 


# for each sample:

trainloader_core50 = torch.utils.data.DataLoader(core50, batch_size=1,
                                          shuffle=True, num_workers=0)
data_iter = iter(trainloader_core50)

images, labels = data_iter.next()
print(images[0])



net = FeatureExtractor(model)
with torch.no_grad():
    outputs = net(images)

    print(outputs)
    print(outputs.shape)