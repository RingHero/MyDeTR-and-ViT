import torch
import torch.nn.functional as F
import torchvision.ops.boxes as box_ops
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_convert

def xywh_to_xyxy(xywh):
    if not isinstance(xywh, torch.Tensor):
        xyxy = [box_convert(i, in_fmt="cxcywh", out_fmt="xyxy") for i in xywh]
        return xyxy
    xyxy = box_convert(xywh, in_fmt="cxcywh", out_fmt="xyxy")
    return xyxy

def HungarianMatch(real_id:tuple, real_bbox:tuple, pred_id:torch.Tensor, pred_bbox:torch.Tensor):
    batch_size, num_queries = pred_id.shape[0:2]
    
    out_logits = F.softmax(pred_id, dim=-1) # (bs, 100, num_calsses+1)
    out_bbox = pred_bbox.flatten(0, 1) #(bs*100, 4)

    # real_id: 元组存储每张图的所有box对应的id: (tensor(1,0,1,2), tensor(0,1,2)...)
    # real_bbox:每张图的所有box对应的bbox: ((a1, 4), (a2, 4)...)
    target_id = torch.cat(real_id) # (num_id,)
    target_bbox = torch.cat(real_bbox, dim=0) # (num_bbox, bbox_xyxy=4)
    
    # class_cost
    out_pro = out_logits.view(-1, out_logits.size(-1))  # (bs*100, num_calsses+1)
    cls_cost = -out_pro[:, target_id] # (bs*100, all_true_id_in_the_batch's_num)

    # l1_cost, giou_cost
    l1_cost = torch.cdist(out_bbox, target_bbox, p=1)
    giou_cost = -box_ops.generalized_box_iou(out_bbox, target_bbox)

    C = 5*cls_cost + 2 * l1_cost + 1 * giou_cost
    C = C.view(batch_size, num_queries, -1)

    # sizes是一个列表，里面每个元素代表每张图的gt框个数
    sizes = [len(i) for i in real_id]
    # C.split(sizes, -1)返回一个元组,为batch_size长度的元组
    indices = [linear_sum_assignment(c[i].detach().cpu().numpy()) for i, c in enumerate(C.split(sizes, -1))]
    # indices：输出batch_size个最优化元组每一个列表[(row_id1, col_id1), (row_id2, col_id2),...]
    return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices] #转换为tensor形式


def compute_loss(all_cls_o, all_bbox_o, real_id:tuple, real_bbox:tuple, device):
    # 获取查询框个数、计算类别数
    _, num_queries, num_classes_plus_one = all_cls_o[0].shape
    num_classes = num_classes_plus_one -1 
    # 解码器的层数
    num_decoder_layers = len(all_cls_o)
    # 真实框的个数
    num_gt = sum([len(i) for i in real_id])

    real_id = [id_tensor.to(device) for id_tensor in real_id]
    real_bbox = [bbox_tensor.to(device) for bbox_tensor in real_bbox]
    
    total_cls_loss=0
    total_l1_loss=0
    total_giou_loss=0
    for i in range(num_decoder_layers):
        all_class_o_i = all_cls_o[i] # (bs, num_queries=100, num_classes+1)
        all_bbox_o_i = xywh_to_xyxy(all_bbox_o[i])  # (bs, num_queries=100, bbox=4)
        # HungarianMatch函数返回的是batch_size大小的一个列表，其中包含元组 (query_indices, target_indices), 存放匹配数据
        hungarian_matcher = HungarianMatch(real_id, real_bbox, all_class_o_i, all_bbox_o_i)

        # 预分配目标张量, matched_id=(bs, num_queries), 内部全填入num_classes这个数值, 对于coco数据集, num_classes=80
        matched_id = torch.full((len(hungarian_matcher), num_queries), num_classes, dtype=torch.long).to(device)
        l1_loss = 0
        giou = 0
        for j, (query_indices, target_indices) in enumerate(hungarian_matcher):
            # 使用 real_id 直接填充
            arranged_id = real_id[j][target_indices]
            matched_id[j, query_indices] = arranged_id

            # L1_loss与Giou_loss
            arranged_bbox = real_bbox[j][target_indices]
            pred_bbox = all_bbox_o_i[j][query_indices]
            # l1_loss
            l1_loss_img = F.l1_loss(pred_bbox, arranged_bbox, reduction="sum")
            l1_loss += l1_loss_img
            # giou_loss
            giou_img = torch.diag(box_ops.generalized_box_iou(pred_bbox, arranged_bbox)).sum()
            giou += giou_img
        
        # 分类损失, 每一个query框的cls_loss
        cls_loss = F.cross_entropy(all_class_o_i.flatten(0, 1), matched_id.flatten(0, 1), 
                                     weight=torch.tensor([1]*num_classes+[0.1],dtype=torch.float).to(device=device))
        # 边界框损失, 一个query框的bbox_loss
        l1_loss = l1_loss /num_gt
        giou_loss = 1 - giou/num_gt

        total_cls_loss += cls_loss * 1/num_decoder_layers
        total_l1_loss += l1_loss * 1/num_decoder_layers
        total_giou_loss += giou_loss * 1/num_decoder_layers

    # 以下部分用于监视模型回归的状态, 对于拟合较好的情况, 应该在300~400epochs趋于拟合，随后的训练可以缩减
    # 模型趋于稳定收敛后, 判断其收敛的依据如下:
    # cls_loss: 0.1~0.3
    # l1_loss: 0.01~0.03
    # giou_loss: 0.1~0.3
    loss_dict = {
        'cls_loss': total_cls_loss.item(),
        'l1_loss': total_l1_loss.item(),
        'giou_loss': total_giou_loss.item()
    }
    
    # 总损失, 趋于收敛后, total_loss: 0.8~1.2之间时, 就可以提前退出训练了
    total_loss = 5*total_cls_loss + 2 * total_l1_loss + 1 * total_giou_loss
    return total_loss, loss_dict  # 返回总损失和损失字典