import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from copy import deepcopy
import cv2
import numpy as np
from PIL import Image

img_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB") if x.mode != 'RGB' else x),  # 如果不是RGB，转换为RGB
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化处理
])


def process_txtdata(txt_path):
    with open(txt_path, 'r') as f:
        content = f.readlines()
        boxes = []
        for i in content:
            line = i.strip()
            parts = line.split()# 去除空格
            new_part = [int(parts[x]) if x==0  else float(parts[x])  for x in range(len(parts))]
            boxes.append(deepcopy(new_part))
        boxes = torch.tensor(boxes) # [num_bbox, 5], 第一维度为一张图片中的框数量, 第二维度为每个框的[id, xc, yc, w, h]的归一化数据
        box_ids = boxes[:, 0].long() # tensor(1,0,1,2...) (num_bbox,)
        box_xywhs = boxes[:, 1:] # (num_bbox, 4)
        return box_ids, box_xywhs
    
class OpenCVResizer:
    def __init__(self, target_size=(224,224)):
        self.tw, self.th = target_size

    def __call__(self,img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale_w = self.tw / w
        scale_h = self.th / h
        scale = min(scale_w, scale_h)

        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        new_img = np.zeros((self.th, self.tw, 3), dtype=np.uint8)
        new_img[0:new_h, 0:new_w] = resized_img
        #cv2.imshow("src",img)
        #cv2.imshow("resized_img",resized_img)
        #cv2.waitKey(0)
        return Image.fromarray(new_img),(new_w,new_h)

def collate_fn_train_val(batch):
    images, ids_list, bboxes_list, w1h1 = zip(*batch)
    
    # 堆叠图像和掩码
    images = torch.stack(images, dim=0)
    
    # 处理变长的标签数据（不堆叠）
    return images, ids_list, bboxes_list, w1h1

class TrainDataset_for_DETR(Dataset):
    def __init__(self, imgdir_path, txtdir_path, target_size=(320, 240)):
        super().__init__()
        self.img_lst = []
        self.txt_lst = []
        self.target_size = target_size
        
        # 使用OpenCV加速器
        self.resizer = OpenCVResizer(target_size=target_size)
        
        # 构建图像和标签路径列表
        self.img_lst = [os.path.join(imgdir_path, f) for f in os.listdir(imgdir_path)]
        self.txt_lst = [os.path.join(txtdir_path, f) for f in os.listdir(txtdir_path)]
        
        # 确保图像和标签匹配
        self.img_lst.sort()
        self.txt_lst.sort()
        assert len(self.img_lst) == len(self.txt_lst), "图像和标签数量不匹配"
        
        # 预缓存图像路径映射 (可选优化)
        self.img_cache = {}
        print(f"Dataset initialization completed, a total of {len(self.img_lst)} samples")

    def __len__(self):
        return len(self.img_lst)   
    
    def __getitem__(self, index):
        # 处理图像和掩码
        resizer = self.resizer(self.img_lst[index])
        resized_img, w1h1 = resizer  # 获取填充后的图像和掩码
        
        # 分别转换图像和掩码
        input_img = img_transform(resized_img)
        
        # 处理标签数据
        real_id, real_bbox = process_txtdata(self.txt_lst[index])
        w1, h1 = w1h1
        # real_bbox归一化到 tw, th大小上
        real_bbox[:, 0] *= w1 / self.target_size[0]  # 第 0 列乘以 w1/tw
        real_bbox[:, 1] *= h1 / self.target_size[1]  # 第 1 列乘以 h1/th
        real_bbox[:, 2] *= w1 / self.target_size[0]  # 第 2 列乘以 w1/tw
        real_bbox[:, 3] *= h1 / self.target_size[1]  # 第 3 列乘以 h1/th
        # real_bbox不仅适用于xywh格式, 也适用于xyxy格式, 调整到指定目标大小下的归一化坐标
        #print("input_img:",input_img.shape)
        #print("real_id:",real_id.shape)
        #print("real_bbox:",real_bbox.shape)
        #print("w1h1",w1h1)
        
        return input_img, real_id, real_bbox, w1h1
    

if __name__ == "__main__":
    #a = OpenCVResizer()
    #a("D:\\BaiduNetdiskDownload\\DETR_code(1)\\ViT_code\\coco128\\images\\train2017\\000000000009.jpg")
    mydataset = TrainDataset_for_DETR(imgdir_path="D:\\BaiduNetdiskDownload\\DETR_code(1)\\ViT_code\\coco128\\images\\train2017",txtdir_path="D:\\BaiduNetdiskDownload\\DETR_code(1)\\ViT_code\\coco128\\labels\\train2017")
    mydataloader = DataLoader(mydataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    for batch in mydataloader:
        img,real_id,real_bbox,w1h1 = batch
        print(f"图像尺寸: {img.shape}")          # torch.Size([bs, 3, H, W])
        print(f"real_id: {real_id.shape}")         # torch.Size([bs, 800, 800])
        print(f"real_bbox:{real_bbox}")
        print(f"w1h1:{w1h1}")
        break