import pdb
import torch
import torch.nn as nn
from torchvision.models import resnet34
from torchvision.transforms import (
    Normalize,
    Compose, 
    Resize,
    CenterCrop, 
    ToTensor,
)

class ResNet34FeatureReader(nn.Module):
    def __init__(self):
        super(ResNet34FeatureReader, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Use {self.device}")
        resnet = resnet34(pretrained=True).to(self.device)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = resnet.fc
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.transform = Compose(
            [
                Resize(256), 
                CenterCrop(224), 
                ToTensor(),
            ]
        ) 

    def forward(self, imgs):
        #x = []
        #for img in imgs:
        #    img = self.transform(img)
        #    img = self.normalize(img)
        #    x.append(img)
        #x = torch.stack(x).to(self.device)
        x = imgs.to(self.device)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.avgpool(x)
        out = out.squeeze(-1).squeeze(-1)
        logits = self.fc(out)
        return logits

if __name__ == "__main__":
    a = resnet34(pretrained=True)
    a2 = ResNet34FeatureReader()
    b = torch.zeros(1, 3, 224, 224) #torch.randn(1, 3, 224, 224)
    print(a(b).softmax(-1)[0, :5])
    print(a2(b).softmax(-1)[0, :5])
