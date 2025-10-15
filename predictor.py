import os
import cv2
import math
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from detective_transformer import MyTransformer
from dataset_coco_detr import TrainDataset_for_DETR,OpenCVResizer
from hungarain_assignment import xywh_to_xyxy
from torchvision.ops import nms

img_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB") if x.mode != 'RGB' else x),  # 如果不是RGB，转换为RGB
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化处理
])

def draw_bbox(img, bbox, cls, score, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox
    w = img.shape[0]
    h = img.shape[1]
    x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    label = f'{cls} {score:.2f}'
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return img

if __name__ =="__main__":
    conf_thres = 0.1
    num_classes = 80
    iou_threshold = 0.7
    img_path = ".\\coco128\\images\\train2017\\000000000009.jpg"

    resizer = OpenCVResizer()
    img,wh = resizer(img_path=img_path)
    input_img = img_transform(img)
    input_img = input_img.unsqueeze(0)
    device = torch.device('cpu')
    model = MyTransformer(n_dim=256,cls_num=80,return_all_layer=False).to(device)
    model.load_state_dict(torch.load("model_last_coco_1.pth"))
    model.eval()

    last_cls, last_bbox = model(input_img.to(device))
    pred_cls_probability = F.softmax(last_cls, dim=-1)  # shape: (batch_size, 100, num_classes+1)
    pred_bbox = xywh_to_xyxy(last_bbox)  # shape: (batch_size, 100, 4)

    batch_preds = []
    each_img_cls_pre = pred_cls_probability[0]  # shape: (100, num_classes+1)
    max_prob_of_query, max_prob_of_query_cls_idx = torch.max(each_img_cls_pre, dim=-1)
    high_conf_mask = (max_prob_of_query >= conf_thres) & (max_prob_of_query_cls_idx != num_classes)

    print(each_img_cls_pre)
    print(max_prob_of_query)
    print(max_prob_of_query_cls_idx)

    pred_cls = max_prob_of_query_cls_idx[high_conf_mask] # 筛选得到的类别, dim = (n, )
    pred_cls_score = max_prob_of_query[high_conf_mask] # 筛选的类别的概率，也就是置信度分数, dim = (n, )
    pred_boxes = pred_bbox[0][high_conf_mask]

    filtered_boxes_list = []
    filtered_cls_list = []
    filtered_score_list = []

    #print(pred_cls.shape)
    # 获取所有存在的类别
    unique_classes = torch.unique(pred_cls)
    #print(unique_classes.shape)
    # 对每个类别单独进行NMS
    for cls in unique_classes:
    # 创建当前类别的掩码
        cls_mask = (pred_cls == cls)
                        
        # 获取当前类别的框、分数和类别
        cls_boxes = pred_boxes[cls_mask]
        cls_scores = pred_cls_score[cls_mask]
        cls_labels = pred_cls[cls_mask]  # 实际都是同一个类别值
                        
        # 如果当前类别没有框，跳过
        if cls_boxes.numel() == 0:
            continue
                        
        # 对当前类别应用NMS
        keep = nms(cls_boxes, cls_scores, iou_threshold)
                        
        # 添加到结果列表
        filtered_boxes_list.append(cls_boxes[keep])
        filtered_cls_list.append(cls_labels[keep])
        filtered_score_list.append(cls_scores[keep])   

        # 如果存在有效检测结果，合并所有类别
        if filtered_boxes_list:
            filtered_boxes = torch.cat(filtered_boxes_list)
            filtered_cls = torch.cat(filtered_cls_list)
            filtered_cls_score = torch.cat(filtered_score_list)
        else:
            # 如果没有检测结果，创建空张量
            filtered_boxes = torch.empty((0, 4), device=pred_boxes.device)
            filtered_cls = torch.empty((0,), dtype=torch.long, device=pred_cls.device)
            filtered_cls_score = torch.empty((0,), device=pred_cls_score.device)
                    
        # 每一张图片的预测情况，写入batch_preds
        batch_preds.append((filtered_boxes, filtered_cls, filtered_cls_score))

    
    print(len(last_cls),last_cls[0].shape)
    print("----------------------------------------------------------------------------------------")
    print(len(last_bbox),last_bbox[0].shape)
    print("----------------------------------------------------------------------------------------")
    print(len(filtered_boxes_list))
    print(filtered_boxes_list)
    print("----------------------------------------------------------------------------------------")
    
    print(len(filtered_cls_list))
    print("----------------------------------------------------------------------------------------")
    print(len(filtered_score_list))
    print("----------------------------------------------------------------------------------------")

    src = cv2.imread(img_path)
    res = src
    for i in range(len(filtered_boxes_list)):
        res = draw_bbox(src, filtered_boxes_list[i][0], filtered_cls[i], filtered_score_list[i][0], color=(0, 0, 255), thickness=2)
    cv2.imshow("res",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
