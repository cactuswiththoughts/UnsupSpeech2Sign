import argparse
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
from PIL import Image
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import videotransforms


import json
import numpy as np
from pathlib import Path

from pytorch_i3d import InceptionI3d

import cv2

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="rgb")
    parser.add_argument("--video_root", default="~/MS-ASL/downloads")
    parser.add_argument("--json_path", default="~/UnsupSpeech2Sign/manifest/asl_librispeech960_100words/wrd2vid.json")
    parser.add_argument("--load_model", default="models/rgb_imagenet.pt")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_dir", default="~/UnsupSpeech2Sign/manifest/MS-ASL/feat/i3d_rgb")
    return parser

class MSASL(torch.utils.data.Dataset):
    def __init__(
        self,
        video_root="~/MS-ASL/downloads",
        json_path="~/UnsupSpeech2Sign/manifest/asl_librispeech960_100words/wrd2vid.json",
        split="train",
        mode="rgb",
        skip=False,
        save_dir="flow_results"
    ):
        self.video_root = Path(video_root)
        
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
  
        wrd2vid = json.load(open(json_path))
        
        self.videos = []
        for w in sorted(wrd2vid):
            for vid_info in wrd2vid[w]:
                video_id = vid_info["url"].split("v=")[-1]
                video_name = f"{video_id}.mp4"
                if not (self.video_root / video_name).exists():
                    video_name = f"{video_id}.mkv"
                if not (self.video_root / video_name).exists():
                    print(f"Warning: {video_name} does not exist")
                    continue
                video_path = self.video_root / video_name
                start = vid_info["start_time"]
                end = vid_info["end_time"]
                sign_id = f"{video_name}_{start}_{end}"
                
                if not os.path.exists(os.path.join(save_dir, sign_id+'.npy')):
                    box = vid_info["box"]  # (y1, x1, y2, x2)
                    self.videos.append(
                        (sign_id, video_path, start, end, box)
                    )
        print(f"Number of videos: {len(self.videos)}") 
        
        self.mode = mode
        
    def compute_TVL1(self, prev, curr, bound=15):
        """Compute the TV-L1 optical flow."""
        TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = TVL1.calc(prev, curr, None)
        flow = np.clip(flow, -20,20) #default values are +20 and -20
        #print(flow)
        assert flow.dtype == np.float32

        flow = (flow + bound) * (255.0 / (2*bound))
        flow = np.round(flow).astype(int)
        flow[flow >= 255] = 255
        flow[flow <= 0] = 0

        return flow

    def load_video(self, video_path, start, end, box, mode="rgb"):    
        try:
            vid_frames = torchvision.io.read_video(
                str(video_path),
                start_pts=start,
                end_pts=end,
                pts_unit="sec",
            )[0]
        except:
            print(f"Failed to open {video_path}")
            return None 
        x, y = vid_frames.shape[2], vid_frames.shape[1]
                
        y1 = int(y*box[0])
        x1 = int(x*box[1])
        y2 = int(y*box[2])
        x2 = int(x*box[3])

        vid_frames = vid_frames[:, y1:y2, x1:x2]
        
        if mode == "flow":
            flow = []
            
            prev = vid_frames[0].numpy()
            prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            prev = cv2.resize(prev,(224,224))
            
            for i in range(vid_frames.shape[0]):
                curr = vid_frames[i].numpy()
                curr = cv2.resize(curr,(224,224))
                curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                tmp_flow = self.compute_TVL1(prev, curr)
                tmp_flow = np.swapaxes(tmp_flow[None], 0, -1)[:,:,:,0]
                flow.append(tmp_flow)
                prev = curr
                
            flow = np.array(flow)
            flow = np.swapaxes(flow, 0, 1)
            flow = (flow/255.)*2 - 1
            
            return torch.from_numpy(flow.astype("float32"))
              
        elif mode == "rgb":
            
            vid_frames = [
                self.transform(
                    Image.fromarray(frame)
                )
                for frame in vid_frames.numpy()
            ]

            vid_frames = torch.stack(vid_frames, dim=1).float() 
            
            return vid_frames
        
        else:
            raise ValueError("Mode not supported")
 
    def __getitem__(self, idx):
        sign_id, video_path, start, end, box = self.videos[idx]
        print(self.mode)
        vid_frames = self.load_video(video_path, start, end, box, mode=self.mode)
        return vid_frames, sign_id

    def __len__(self):
        return len(self.videos)

def main():
    parser = get_parser()
    args = parser.parse_args()
    save_dir = Path(args.save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    dataset = MSASL(args.video_root, args.json_path, mode=args.mode)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # setup dataset
    datasets = {'train': dataset}

    # setup the model
    if args.mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)

    if args.gpu < 0:
        device = "cpu"
    else:
        device = "cuda"
    print(f"Device: {device}")

    if "charades" in args.load_model:
        i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(args.load_model))
    i3d.to(device)
    i3d.train(False)  # Set model to evaluate mode

    # Iterate over data.
    phase = "train"
    for data in datasets[phase]:
        start_runtime = time.time() 

        # get the inputs
        inputs, name = data
        if inputs is None:
            continue
        if os.path.exists(os.path.join(save_dir, name+'.npy')):
            continue

        c,t,h,w = inputs.shape
        print(f"{name}, inputs.shape: {inputs.shape}", flush=True)
        with torch.no_grad():
            min_len = 56
            if t < min_len:
                pad = inputs.new_zeros(inputs.shape[0], min_len-t, inputs.shape[2], inputs.shape[3])
                inputs = torch.cat((inputs, pad), dim=1)

            max_len = 200
            if t > max_len:
                features = []
                # for start in range(1, t-56, max_len):
                #    end = min(t-1, start+max_len+56)
                #    start = max(1, start-48)
                for start in range(0, t-56, max_len):
                    end = start + max_len
                    ip = Variable(
                        torch.from_numpy(
                            inputs.numpy()[:, start:end]
                        ).to(device),
                    ).unsqueeze(0)
                    feat = i3d.extract_features(ip).squeeze(0).squeeze(-1).squeeze(-1)
                    feat = feat.data.t().cpu().numpy()
                    features.append(feat)
                features = np.concatenate(features, axis=0)
            else:
                # wrap them in Variable
                inputs = Variable(
                    inputs.to(device),
                ).unsqueeze(0)
                features = i3d.extract_features(inputs)
                features = features.squeeze(0).squeeze(-1).squeeze(-1)
                features = features.data.t().cpu().numpy()
            np.save(
                save_dir / f"{name}.npy",
                features,
            )
        print(f"{name}, features.shape: {features.shape}", flush=True)
        print(f"Takes {time.time() - start_runtime}s to process video", flush=True)

if __name__ == "__main__":
    main()
