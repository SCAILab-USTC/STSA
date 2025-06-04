import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class mseloss(nn.Module):
    def __init__(self, use_weight_map=True):
        super(mseloss, self).__init__()

        self.use_weight_map = use_weight_map

    def __repr__(self):
        return "mseloss()"

    def generate_weight_map(self, heatmap, gt_pose, k_size=3, w=2):
        dilate_whole = F.max_pool2d(heatmap, kernel_size=k_size, stride=1, padding=1)
        dilate_pose = F.max_pool2d(gt_pose, kernel_size=k_size, stride=1, padding=1)
        
        dilate_lower = dilate_whole - dilate_pose
        
        weight_map_whole = torch.where(dilate_whole < 0.2, torch.zeros_like(heatmap), torch.ones_like(heatmap))
        weight_map_lower = torch.where(dilate_lower < 0.2, torch.zeros_like(heatmap), torch.ones_like(heatmap))
        
        weight_map = weight_map_whole + weight_map_lower # 3-level mask : lower-5 upper-3 background-1
        
        return w * weight_map + 1

    def forward(self, output, groundtruth, gt_pose):
        """
        input:  b x n x h x w
        output: b x n x h x w => 1
        """
        loss = (output - groundtruth) ** 2
        if self.use_weight_map:
            weight = self.generate_weight_map(groundtruth, gt_pose)
            loss = loss * weight
        return loss.mean()
    
if __name__ == "__main__":
    heatmap_whole = np.load("processed_lrs2_ours/lrs2_heatmap_whole/5576857409011173607/00001/0.npy")
    heatmap_pose = np.load("processed_lrs2_ours/lrs2_heatmap_upper/5576857409011173607/00001/0.npy")
    gt1 = torch.FloatTensor(heatmap_whole).unsqueeze(0).unsqueeze(0)
    gt2 = torch.FloatTensor(heatmap_pose).unsqueeze(0).unsqueeze(0)

    pre = torch.randn(1, 1, 128, 128)
    
    aw = mseloss()
    weight = aw.generate_weight_map(gt1, gt2)
    print(weight.max())
    weight_img = weight[0][0].cpu().detach().numpy() *10
    # print(weight.shape) (1, 1, 128, 128)
    cv2.imwrite("test2.jpg", weight_img)
    
    loss = aw(pre, gt1, gt2)
    print(loss)
    