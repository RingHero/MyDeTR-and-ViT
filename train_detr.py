import torch
import detective_transformer
from torch import optim
from tqdm import tqdm
from dataset_coco_detr import TrainDataset_for_DETR
from torch.utils.data import DataLoader
from hungarain_assignment import compute_loss
from dataset_coco_detr import collate_fn_train_val

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = detective_transformer.MyTransformer(n_dim=256,cls_num=80,return_all_layer=True).to(device)
    optimizer = optim.AdamW(model.parameters(),lr=0.00001,weight_decay=0.0001)
    mydataset = TrainDataset_for_DETR(imgdir_path="./coco128/images/train2017",txtdir_path="./coco128/labels/train2017")
    mydataloader = DataLoader(mydataset, batch_size=64, shuffle=True, collate_fn=collate_fn_train_val,num_workers=0, pin_memory=True)
    epoches = 300
    
    for epoch in range(epoches):
        model.train()
        epoch_losses = {'total': 0.0, 'cls': 0.0, 'l1': 0.0, 'giou': 0.0}
        batch_num = 0
        t = tqdm(mydataloader, unit='batch', desc=f'Training Epoch {epoch+1}/{epoches}')
        for batch in t:
            img, real_id, real_bbox, _ = batch
            all_cls,all_bbox = model(img.to(device))
            total_loss, loss_dict = compute_loss(
                all_cls, all_bbox, real_id, real_bbox, 
                device=device
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            optimizer.zero_grad()
            batch_num += 1
            epoch_losses['total'] += total_loss.item()
            for i in ['cls', 'l1', 'giou']:
                epoch_losses[i] += loss_dict[f'{i}_loss']
            t.set_postfix({k: v for k, v in loss_dict.items()})

        print(f"\nEpoch {epoch+1} Training summary:")
        for k in ['cls', 'l1', 'giou', 'total']:
            avg = epoch_losses[k] / batch_num
            print(f"  {k}_loss: {avg:.4f}")
        
        if epoch_losses['giou']<0 and abs(0.5*epoch_losses['giou'])> (epoch_losses['cls']+epoch_losses['l1']):
            print("early stop!")
            break
    torch.save(model.state_dict(),"model_last_coco.pth")
    
if __name__ == "__main__":
    main()
