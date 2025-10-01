import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from sklearn.preprocessing import MultiLabelBinarizer

class DiffusionDataset(Dataset):
    def __init__(self,
                 img_root: str,
                 ann_file: str,
                 objects_file: str,
                 high_res: int = 128,
                 mode: str = 'train'):
        """
        mode = 'train' or 'test'
        - Training: ann_file is a dictionary {filename: [labels]}}
        - Test: ann_file is a list of [labels]
        """
        with open(objects_file, 'r') as f:
            self.obj2idx = json.load(f)

        with open(ann_file, 'r') as f:
            data = json.load(f)

        if mode == 'train':
            # data: dict filename -> list of names
            self.filenames = list(data.keys())
            self.labels = [[self.obj2idx[n] for n in data[fn]]
                           for fn in self.filenames]
        else:
            # data: list of lists of names
            self.filenames = None
            self.labels = [[self.obj2idx[n] for n in entry]
                           for entry in data]

        # prepare one-hot encoder
        self.mlb = MultiLabelBinarizer(classes=list(range(len(self.obj2idx))))
        self.mlb.fit(self.labels)

        # transforms for images (only used in train mode)
        self.transform = T.Compose([
            T.Resize((high_res, high_res), T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize((0.5,)*3, (0.5,)*3),
        ])
        self.img_root = img_root
        self.mode = mode

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        onehot = torch.from_numpy(
            self.mlb.transform([self.labels[idx]])[0]
        ).float()

        if self.mode == 'train':
            fn = self.filenames[idx]
            img = Image.open(os.path.join(self.img_root, fn)).convert('RGB')
            img = self.transform(img)
            return img, onehot
        else:
            # test mode: return only label (img slot unused)
            return onehot
