import os, json, torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class ICLEVR(Dataset):
    def __init__(self,
                 root_dir: str,
                 mode:     str = "train",
                 img_root: str = None):
        """
        root_dir: 放 JSON 及 objects.json 的資料夾
        mode:     "train" | "test" | "new_test"
        img_root: 訓練圖片資料夾路徑 (mode=='train' 時必填)
        """
        assert mode in ("train","test","new_test")
        self.mode     = mode
        self.root_dir = root_dir
        # 讀 JSON
        json_path = os.path.join(root_dir, f"{mode}.json")
        with open(json_path) as f:
            meta = json.load(f)
        # 讀 objects.json
        with open(os.path.join(root_dir,"objects.json")) as f:
            self.obj2idx = json.load(f)

        if mode == "train":
            self.names    = list(meta.keys())
            label_list    = list(meta.values())
            assert img_root is not None, "train 模式需傳入 img_root"
            self.img_root = img_root
        else:
            self.names    = None
            label_list    = meta
            self.img_root = None

        # one-hot labels
        self.labels = []
        C = len(self.obj2idx)
        for lbl in label_list:
            v = torch.zeros(C)
            for o in lbl:
                v[self.obj2idx[o]] = 1
            self.labels.append(v)
        self.labels = torch.stack(self.labels)

        # 圖像轉換
        self.tf = transforms.Compose([
            transforms.Resize((64,64), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        if self.mode == "train":
            fn  = self.names[idx]
            img = Image.open(os.path.join(self.img_root, fn)).convert("RGB")
            return self.tf(img), label
        else:
            return label
