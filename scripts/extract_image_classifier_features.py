import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.svm import LinearSVC
import time
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import (
    Normalize,
    Compose, 
    Resize,
    CenterCrop, 
    ToTensor,
)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", 
        default="~/asl_alphabet",
    )
    parser.add_argument(
        "--model_name", 
        default="vgg19",
    )
    parser.add_argument(
        "--out_path",
        default=""
    )
    parser.add_argument(
        "--classify", 
        action="store_true",
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
    )
    return parser

# Reference: https://pytorch.org/vision/0.9/models.html
class ResNet34FeatureReader(nn.Module):
    def __init__(self):
        super(ResNet34FeatureReader, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Use {self.device}")
        resnet = models.resnet34(pretrained=True).to(self.device)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
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
        x = []
        for img in imgs:
            img = self.transform(img)
            img = self.normalize(img)
            x.append(img)
        x = torch.stack(x).to(self.device)
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
        return out   

# Reference: https://pytorch.org/vision/0.9/models.html
class ResNet152FeatureReader(nn.Module):
    def __init__(self):
        super(ResNet152FeatureReader, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Use {self.device}")
        resnet152 = models.resnet152(pretrained=True).to(self.device)
        self.conv1 = resnet152.conv1
        self.bn1 = resnet152.bn1
        self.relu = resnet152.relu
        self.maxpool = resnet152.maxpool
        self.layer1 = resnet152.layer1
        self.layer2 = resnet152.layer2
        self.layer3 = resnet152.layer3
        self.layer4 = resnet152.layer4
        self.avgpool = resnet152.avgpool
        
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
        x = []
        for img in imgs:
            img = self.transform(img)
            img = self.normalize(img)
            x.append(img)
        x = torch.stack(x).to(self.device)
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
        return out

class VGG19FeatureReader(nn.Module):
    def __init__(self):
        super(VGG19FeatureReader, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Use {self.device}")
        vgg19 = models.vgg19(pretrained=True).to(self.device)
        self.features = vgg19.features
        self.avgpool = vgg19.avgpool
        classifier = list(
            vgg19.classifier.children()
        )[:-2]
        self.extractor = nn.Sequential(*classifier)
        
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
        x = []
        for img in imgs:
            img = self.transform(img)
            img = self.normalize(img)
            x.append(img)
        x = torch.stack(x).to(self.device)
        x = self.features(x)
        x = self.avgpool(x)
        out = self.extractor(
            x.view(x.size(0), -1)
        )
        return out

# Extract features from image classifiers pretrained on ImageNet
# for the fingerspelling images and put it into the following 
# format:
#       root/
#           - {split}.tsv
#           - {split}.npy
def main():
    # Get input arguments
    parser = get_parser()
    args = parser.parse_args()
    data_path = Path(args.data_path)
    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    #(out_path / "train").mkdir(parents=True, exist_ok=True)
    #(out_path / "test").mkdir(parents=True, exist_ok=True)
    if args.model_name == "vgg19":
        model = VGG19FeatureReader()
    elif args.model_name == "resnet34":
        model = ResNet34FeatureReader()
    elif args.model_name == "resnet152":
        model = ResNet152FeatureReader()
    else:
        raise ValueError(f"model name {args.model_name} not found")
    letters = None
    svm = LinearSVC()
    
    begin_time = time.time() 
    for split in ["train", "test"]:
        print(
            f"Extracting features for the {split} set",
            flush=True,
        )
        split_path = data_path / f"asl_alphabet_{split}"
        if split == "train":
            letters = [p.name for p in sorted(split_path.iterdir())]
        X = []
        y = []
        nf = 0
        # Extract features for the dataset
        with open(out_path / f"{split}.tsv", "w") as f_tsv:
            f_tsv.write(f"{split_path}\n")
            for i, ltr in enumerate(letters):
                if split == "train":
                    ltr_path = split_path / ltr
                    filenames = list(ltr_path.iterdir())
                    for fn in filenames:
                        f_tsv.write(f"{ltr}/{fn.name}\t1\n")
                else:
                    fname = f"{ltr}_test.jpg"
                    fn = split_path / fname
                    if not fn.exists():
                        continue
                    filenames = [fn]
                    f_tsv.write(f"{fname}\t1\n")
        
                for j, fn in enumerate(filenames):
                    if args.debug and j > 1:
                        break
                    
                    img = Image.open(fn)
                    with torch.no_grad():
                        x = model([img]).squeeze(0)
                    
                    X.append(
                        x.cpu().detach().numpy()
                    )
                    y.append(i)
                    if (nf + 1) % 1000 == 0:
                        print(
                            f"Extract features for {nf} files"
                            f" in {time.time() - begin_time:.2f} s",
                            flush=True,
                        )
                    nf += 1
        
        # Save the feature
        X = np.stack(X)
        np.save(
            out_path / f"{split}.npy", X
        )
        # Train a linear classifier using the features
        if args.classify:
            y = np.asarray(y)
            if split == "train":
                print(
                    "Start training a linear classifier on the train set",
                    flush=True,
                )
                svm.fit(X, y)
            y_pred = svm.predict(X)
            print(
                f"{split} accuracy: {np.mean(y == y_pred):.4f}",
                flush=True,
            )

if __name__ == "__main__":
    main() 
