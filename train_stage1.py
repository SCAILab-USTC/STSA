import os

import cv2
from tqdm import tqdm
import logging 
import torch
import numpy as np
from glob import glob
from os.path import join, isfile
import  random
from boundary_heatmap_draw import heatmap_visualize
from torch.utils.tensorboard import SummaryWriter
from models import Heatmap_generator 
import argparse

from MSEloss import mseloss

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)


parser=argparse.ArgumentParser()
parser.add_argument('--pre_audio_root',default='./processed_lrs2/lrs2_audio',
                    help='root path for preprocessed  audio')
parser.add_argument('--landmarks_root',default='./processed_lrs2/lrs2_landmarks',
                    help='root path for preprocessed  landmarks')
parser.add_argument('--whole_heatmap_root',default='./processed_lrs2/lrs2_heatmap_whole',
                    help='root path for preprocessed  landmarks')
parser.add_argument('--upper_heatmap_root',default='./processed_lrs2/lrs2_heatmap_upper',
                    help='root path for preprocessed  landmarks')

args=parser.parse_args()

#network parameters
d_model=512
dim_feedforward=1024
nlayers=8
nhead=4
dropout=0.1 # 0.5
Nl=5
T = 5
Project_name = 'heatmap_gen'
print('Project_name:', Project_name)
finetune_path = None
num_workers = 14
batch_size = 4  # 512
batch_size_val = 64  #512
evaluate_interval = 5000  #
checkpoint_interval = evaluate_interval
mel_step_size = 16
w2v_step_size = 10
fps = 25
lr = 1e-4 #0:1e-4  #75k:1e-5 #130k:1e-6 
global_step, global_epoch = 0, 0 

landmark_root=args.landmarks_root
whole_heatmap_root = args.whole_heatmap_root
upper_heatmap_root = args.upper_heatmap_root

filelist_name = 'lrs2'
checkpoint_root = './checkpoints/heatmap_generation/'
checkpoint_dir = os.path.join(checkpoint_root, Project_name)
reset_optimizer = False
save_optimizer_state = True
writer = SummaryWriter('tensorboard_runs/Project_{}'.format(Project_name))
#we arrange the landmarks in some order
ori_sequence_idx=[162,127,234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454,356,389,  #
    70,63,105,66,107,55,65,52,53,46,#
    336,296,334,293,300,276,283,282,295,285,#
    168,6,197,195,5,#
    48,115,220,45,4,275,440,344,278,#
     33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,#
    362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382,#
    61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146,#
    78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]
full_face_sequence=[*list(range(0, 4)), *list(range(21, 25)), *list(range(25, 91)), *list(range(4, 21)), *list(range(91, 131))]
class LandmarkDict(dict):
    def __init__(self, idx, x, y):
        self['idx'] = idx
        self['x'] = x
        self['y'] = y

    def __getattr__(self, name):
        try:
            return self[name]
        except:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


class Dataset(object):
    def get_vidname_list(self, split):
        vid_name_list = []
        with open('filelists/{}/{}.txt'.format(filelist_name, split)) as f:
            for line in f:
                line = line.strip()
                if ' ' in line: line = line.split()[0]
                vid_name_list.append(line)
        return vid_name_list

    def __init__(self, split):
        min_len = 25  #filter videos that is too short
        vid_name_lists = self.get_vidname_list(split) 
        self.all_video_names = []
        print("init dataset,filtering very short videos.....")
        for vid_name in tqdm(vid_name_lists, total=len(vid_name_lists)):
            pkl_paths = list(glob(join(landmark_root,vid_name, '*.npy')))
            vid_len=len(pkl_paths)
            if vid_len >= min_len:
                self.all_video_names.append((vid_name, vid_len))
        print("complete,with available vids: ", len(self.all_video_names), '\n')

    def normalize_and_transpose(self, window):
        x = np.asarray(window) / 255.
        x = np.transpose(x, (0, 3, 1, 2))
        return torch.FloatTensor(x)  # B,3,H,W
    
    def __len__(self):
        # return 150*len(self.all_video_names)
        return len(self.all_video_names)

    def __getitem__(self, idx):
        while 1:
            
            vid_idx = random.randint(0, len(self.all_video_names) - 1)   
            vid_name = self.all_video_names[vid_idx][0]
            vid_len=self.all_video_names[vid_idx][1]
            
            frame_list = os.listdir(os.path.join(whole_heatmap_root, vid_name))
            i_list = []
            for item in frame_list:
                i = int(item.split('/')[-1].replace(".npy",''))
                i_list.append(i)
            i_list.sort()
            
            # 00.randomly select a window of T video frames
            # random_start_idx = random.randint(i_list[0] + 2, i_list[0] + vid_len - T - 2)          # 1. 起点不能是2
            T_idxs = []
            random_start_idx = random.randint(0 + 2, len(i_list) - T - 2)
            for i in range(0, T):
                T_idxs.append(i_list[random_start_idx + i])
            # print(T_idxs)
            if T_idxs[-1] - T_idxs[0] != (T-1):
                # print("nonono")
                continue
            # 01. get reference whole_heatmap
            # all_list=[i for i in range(i_list[0], i_list[0] + vid_len) if i not in T_idxs]  # 2. range也要有个起点
            all_list = [x for x in i_list if x not in T_idxs]
            
            Nl_idxs = random.sample(all_list, Nl) # 5 reference frames
            Nl_heatmap_paths = [os.path.join(whole_heatmap_root, vid_name, str(idx) + '.npy') for idx in Nl_idxs]

            Nl_whole_heatmap = []
            for frame_heatmap_path in Nl_heatmap_paths:
                if not os.path.exists(frame_heatmap_path):
                    print("looking new")
                    new_id = random.sample(all_list, 1)[0]
                    frame_heatmap_path = frame_heatmap_path.replace(frame_heatmap_path.split('/')[-1], str(new_id)+'.npy')
                    if os.path.exists(frame_heatmap_path):
                        print("success")
                        
                whole_heatmap=np.load(frame_heatmap_path)
                Nl_whole_heatmap.append(whole_heatmap)
            if len(Nl_whole_heatmap) != Nl:
                continue
            Nl_whole_heatmap = np.asarray(Nl_whole_heatmap) # (Nl, 128,128)
            Nl_whole_heatmap = torch.FloatTensor(Nl_whole_heatmap).unsqueeze(1)  # (Nl,1,128,128)
            
            # 02. get T content landmark 
            T_landmark_paths = [os.path.join(landmark_root, vid_name, str(idx) + '.npy') for idx in T_idxs]
            T_content_landmarks=[]
            for frame_landmark_path in T_landmark_paths:
                if not os.path.exists(frame_landmark_path):
                    break
                landmarks=np.load(frame_landmark_path,allow_pickle=True).item()
                T_content_landmarks.append(landmarks['content_landmarks'])
            if len(T_content_landmarks)!=T:
                continue
            T_content=torch.zeros((T,2,57))  #57 landmark
            for idx in range(T):
                T_content_landmarks[idx] = sorted(T_content_landmarks[idx],key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))

                T_content[idx, 0, :] = torch.FloatTensor([T_content_landmarks[idx][i][1] for i in range(len(T_content_landmarks[idx]))])  # x
                T_content[idx, 1, :] = torch.FloatTensor([T_content_landmarks[idx][i][2] for i in range(len(T_content_landmarks[idx]))])  # y
            T_content_landmark = T_content # (T, 2, 57)
            
            # 03. get T pose heatmap
            T_pose_heatmap_paths = [os.path.join(upper_heatmap_root, vid_name, str(idx) + '.npy') for idx in T_idxs]
            T_pose_heatmap=[]
            for frame_pose_heatmap_path in T_pose_heatmap_paths:
                if not os.path.exists(frame_pose_heatmap_path):
                    break
                pose_heatmap=np.load(frame_pose_heatmap_path)
                T_pose_heatmap.append(pose_heatmap)
            if len(T_pose_heatmap)!=T:
                continue
            
            T_pose_heatmap = np.asarray(T_pose_heatmap) # (T, 128,128)
            T_pose_heatmap = torch.FloatTensor(T_pose_heatmap).unsqueeze(1)  # (T,1,128,128)
            

            # 04. get T whole heatmap
            T_whole_heatmap_paths = [os.path.join(whole_heatmap_root, vid_name, str(idx) + '.npy') for idx in T_idxs]
            T_whole_heatmap=[]
            for frame_whole_heatmap_path in T_whole_heatmap_paths:
                if not os.path.exists(frame_whole_heatmap_path):
                    break
                whole_heatmap=np.load(frame_whole_heatmap_path)
                T_whole_heatmap.append(whole_heatmap)
            if len(T_whole_heatmap)!=T:
                continue
            
            T_whole_heatmap = np.asarray(T_whole_heatmap) # (T, 128,128)
            T_whole_heatmap = torch.FloatTensor(T_whole_heatmap).unsqueeze(1)  # (T,1,128,128)

            # 05. get T audio wav2vec
            try:
                audio_w2v = np.load(join(args.pre_audio_root,vid_name, "wav2vec2_768.npy"))
            except Exception as e:
                continue
            T_w2vs = []
            for frame_idx in T_idxs:
                mel_start_frame_idx = frame_idx - 2  ###around the frame
                if mel_start_frame_idx < 0:
                    break
                start_idx = int(50. * (mel_start_frame_idx / float(fps)))
                m = audio_w2v[0][start_idx: start_idx + w2v_step_size, :]  # get five frames around
                if m.shape[0] != w2v_step_size:  # in the end of vid
                    break
                T_w2vs.append(m.T)  # transpose to 768,10
            if len(T_w2vs) != T:
                continue
            T_w2vs = np.asarray(T_w2vs)  # (T,768,10)
            # T_w2vs = torch.FloatTensor(T_w2vs).unsqueeze(1)  # (T,1,768,10)
            T_w2vs = torch.FloatTensor(T_w2vs)  # (T,768,10)
            
            #  return value
            return T_w2vs,    T_pose_heatmap,    T_whole_heatmap,   T_content_landmark,    Nl_whole_heatmap
            #     (T,768,10) (T,1,128, 128)     (T,1,128,128)        (T, 2, 57)   Reference: (Nl,1,128,128)
            
def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch
    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    # for k, v in s.items():
    #     new_s['module.'+k] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]
    return model


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(
        checkpoint_dir, "epoch_{}_step{:09d}.pth".format(epoch, global_step))
    if isfile(checkpoint_path):
        os.remove(checkpoint_path)
    optimizer_state = optimizer.state_dict() if save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

criterion_L1 = torch.nn.L1Loss()
mse = mseloss()

def get_velocity_loss (pred, gt):  #(B*T,2,57) (B*T,2,57)

    pred = torch.stack(torch.split(pred,T,dim=0),dim=0)  #(B,T,2,57) # mouth and jaw
    gt = torch.stack(torch.split(gt, T, dim=0), dim=0)  # (B,T,2,57)

    pred = torch.cat([pred[:,:,:,i] for i in range(pred.size(3))],dim=2)  #(B,T,57*2)
    gt = torch.cat([gt[:, :, :, i] for i in range(gt.size(3))], dim=2)  # (B,T,57*2)
    b, t, c = pred.shape
    pred_spiky = pred[:, 1:, :] - pred[:, :-1, :]   
    gt_spiky = gt[:, 1:, :] - gt[:, :-1, :]

    pred_spiky = pred_spiky.view(b * (t - 1), c)
    gt_spiky = gt_spiky.view(b * (t - 1), c)
    pairwise_distance = torch.nn.functional.pairwise_distance(pred_spiky, gt_spiky)
    return torch.mean(pairwise_distance)

def get_heatmap_velocity_loss (pred, gt):  #(B*T,1,128,128) (B*T,1,128,128)

    pred = torch.stack(torch.split(pred,T,dim=0),dim=0).squeeze(2)  # (B, T, 128, 128)
    gt = torch.stack(torch.split(gt, T, dim=0), dim=0).squeeze(2)  # (B, T, 128, 128)


    pred = torch.cat([pred[:,:,:,i] for i in range(pred.size(3))],dim=2)  #(B,T,128*128)
    gt = torch.cat([gt[:, :, :, i] for i in range(gt.size(3))], dim=2)  # (B,T,128*128)
    b, t, c = pred.shape
    pred_spiky = pred[:, 1:, :] - pred[:, :-1, :]  
    gt_spiky = gt[:, 1:, :] - gt[:, :-1, :]

    pred_spiky = pred_spiky.view(b * (t - 1), c)
    gt_spiky = gt_spiky.view(b * (t - 1), c)
    pairwise_distance = torch.nn.functional.pairwise_distance(pred_spiky, gt_spiky)
    return torch.mean(pairwise_distance)

def fix_bn(m):
   classname = m.__class__.__name__
   if classname.find('BatchNorm') != -1:
       m.eval()

def evaluate(model, val_data_loader):
    global global_epoch, global_step
    eval_epochs = 5
    print('Evaluating model for {} epochs'.format(eval_epochs))
    eval_L1_loss = 0.
    eval_velocity_loss=0.
    eval_MSE_loss = 0.
    eval_heatmap_velocity_loss = 0.
    count = 0


    for epoch in tqdm(range(eval_epochs),total=eval_epochs):
        prog_bar = tqdm(enumerate(val_data_loader), total=len(val_data_loader))
        for step, (T_mels, T_pose_heatmap, T_whole_heatmap, T_content_landmark, Nl_whole_heatmap) in prog_bar:
            model.eval()
            T_mels, T_pose_heatmap, T_whole_heatmap,T_content_landmark,Nl_whole_heatmap = T_mels.cuda(non_blocking=True), T_pose_heatmap.cuda(non_blocking=True), T_whole_heatmap.cuda(non_blocking=True), \
                                                                  T_content_landmark.cuda(non_blocking=True), Nl_whole_heatmap.cuda(non_blocking=True)


            # (B,T,1,hv,wv) (B,T,1,128,128)  (B,T,1,128,128)
            predict_whole_heatmap, predict_content_landmark = model(T_mels, T_pose_heatmap, Nl_whole_heatmap)  # (B*T,1, 128 ,128) (B*T,2,57)

            T_content_landmark=torch.cat([T_content_landmark[i] for i in range(T_content_landmark.size(0))],dim=0) #(B*T,2,57):ground truth lip and jaw landmarks
            T_whole_heatmap=torch.cat([T_whole_heatmap[i] for i in range(T_whole_heatmap.size(0))],dim=0) # (B*T, 1, 128, 128)
            T_pose_heatmap=torch.cat([T_pose_heatmap[i] for i in range(T_pose_heatmap.size(0))],dim=0) # (B*T, 1, 128, 128)


            eval_MSE_loss += mse(predict_whole_heatmap, T_whole_heatmap, T_pose_heatmap)
            eval_L1_loss += criterion_L1(predict_content_landmark, T_content_landmark).item()

            eval_velocity_loss += get_velocity_loss(predict_content_landmark, T_content_landmark).item()
            eval_heatmap_velocity_loss += 0.2 * get_heatmap_velocity_loss(predict_whole_heatmap, T_whole_heatmap)
            
            count += 1
            #-------------save heatmap everytime-------------
            ground_truth1 = heatmap_visualize(T_whole_heatmap[0][0].cpu().detach().numpy())
            ground_truth2 = heatmap_visualize(T_whole_heatmap[1][0].cpu().detach().numpy())
            ground_truth3 = heatmap_visualize(T_whole_heatmap[2][0].cpu().detach().numpy())
            ground_truth4 = heatmap_visualize(T_whole_heatmap[3][0].cpu().detach().numpy())
            ground_truth5 = heatmap_visualize(T_whole_heatmap[4][0].cpu().detach().numpy())
            ground_truth = np.concatenate([ground_truth1, ground_truth2, ground_truth3, ground_truth4, ground_truth5], axis=1)
            predict1 = heatmap_visualize(predict_whole_heatmap[0][0].cpu().detach().numpy())
            predict2 = heatmap_visualize(predict_whole_heatmap[1][0].cpu().detach().numpy())
            predict3 = heatmap_visualize(predict_whole_heatmap[2][0].cpu().detach().numpy())
            predict4 = heatmap_visualize(predict_whole_heatmap[3][0].cpu().detach().numpy())
            predict5 = heatmap_visualize(predict_whole_heatmap[4][0].cpu().detach().numpy())
            predict  = np.concatenate([predict1, predict2, predict3, predict4, predict5], axis=1)
            full = np.concatenate([ground_truth, predict], axis=0)
            cv2.imwrite('training_heatmap_vis.jpg', full) 
    
    writer.add_scalar('eval_L1_loss', eval_L1_loss / count, global_step)
    print('eval_L1_loss', eval_L1_loss / count, 'global_step:', global_step)
    writer.add_scalar('eval_MSE_loss', eval_MSE_loss / count, global_step)
    print('eval_MSE_loss', eval_MSE_loss / count, 'global_step:', global_step)
    writer.add_scalar('eval_velocity_loss', eval_velocity_loss / count, global_step)
    print('eval_velocity_loss', eval_velocity_loss / count, 'global_step:', global_step)
    writer.add_scalar('eval_heatmap_velocity_loss', eval_heatmap_velocity_loss / count, global_step)
    print('eval_heatmap_velocity_loss', eval_heatmap_velocity_loss / count, 'global_step:', global_step)
    
    return  eval_L1_loss / count, eval_MSE_loss / count, eval_velocity_loss / count, eval_heatmap_velocity_loss / count
    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # create a logger  
    logger = logging.getLogger('trainging_logger')  
    logger.setLevel(logging.INFO)  
    
    # create a handler
    log_file = 'training_logs/heatmap/{}.log'.format(Project_name)  
    fh = logging.FileHandler(log_file)  
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()  
    ch.setLevel(logging.INFO)  
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  
    fh.setFormatter(formatter)  
    ch.setFormatter(formatter)  
    
    logger.addHandler(fh)  
    logger.addHandler(ch)  
    
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda")
    # create a model and optimizer
    model = Heatmap_generator(T,d_model,nlayers,nhead,dim_feedforward,dropout) # 5, 512, 8, 4, 1024, 0.1
    if finetune_path is not None:  ###fine tune
        model_dict = model.state_dict()
        print('load module....from :', finetune_path)
        checkpoint = torch.load(finetune_path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        state_dict_needed = {k: v for k, v in new_s.items() if k in model_dict.keys()}  # we need in model
        model_dict.update(state_dict_needed)
        model.load_state_dict(model_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.cuda()
    train_dataset = Dataset('train')
    val_dataset = Dataset('test')
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    while global_epoch < 9999999999:
        prog_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        running_L1_loss, running_velocity_loss, running_MSE_loss, running_heatmap_velocity_loss = 0., 0., 0., 0.
        for step, (T_mels, T_pose_heatmap, T_whole_heatmap, T_content_landmark, Nl_whole_heatmap) in prog_bar:
            if global_step % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch, prefix=Project_name)
                
            # ------------------Test------------------- # 
            if (global_step % evaluate_interval == 0) and global_step != -1:
                with torch.no_grad():
                     eval_L1_loss, eval_MSE_loss, eval_velocity_loss, eval_hm_velocity_loss = evaluate(model, val_data_loader)
                     logger.info(f'Epoch:{global_epoch}--Step:{global_step}--eval_L1_loss:{eval_L1_loss}--eval_MSE_loss:{eval_MSE_loss}--eval_velocity_loss:{eval_velocity_loss}--eval_hm_velocity_loss:{eval_hm_velocity_loss}')
            # -------------------------------------------         
            T_mels, T_pose_heatmap, T_whole_heatmap, T_content_landmark, Nl_whole_heatmap = T_mels.cuda(non_blocking=True), T_pose_heatmap.cuda(non_blocking=True), T_whole_heatmap.cuda(non_blocking=True), \
                T_content_landmark.cuda(non_blocking=True),Nl_whole_heatmap.cuda(non_blocking=True)
            #(B,T,1,hv,wv) (B,T,1,128, 128) (B,T,1,128,128) (B,T,2,57)  (B,Nl,1,128,128) 
            ground_truth = heatmap_visualize(T_whole_heatmap[0][0][0].cpu().detach().numpy())
            # ------------------Train------------------- # 
            model.train()
            # model.apply(fix_bn) # important
            optimizer.zero_grad()
            predict_whole_heatmap, predict_content_landmark = model(T_mels, T_pose_heatmap, Nl_whole_heatmap)  # (B*T,1, 128 ,128) (B*T,2,57)
            
            T_content_landmark=torch.cat([T_content_landmark[i] for i in range(T_content_landmark.size(0))],dim=0) #(B*T,2,57):ground truth lip and jaw landmarks
            T_whole_heatmap=torch.cat([T_whole_heatmap[i] for i in range(T_whole_heatmap.size(0))],dim=0) # (B*T, 1, 128, 128)
            T_pose_heatmap=torch.cat([T_pose_heatmap[i] for i in range(T_pose_heatmap.size(0))],dim=0) # (B*T, 1, 128, 128)
            
            
            
            predict = heatmap_visualize(predict_whole_heatmap[0][0].cpu().detach().numpy())
            full = np.concatenate([ground_truth, predict], axis=1)
            cv2.imwrite('training_heatmap_vis.jpg', full) 
            
            MSE_loss = mse(predict_whole_heatmap, T_whole_heatmap, T_pose_heatmap)
            L1_loss=criterion_L1(predict_content_landmark,T_content_landmark)
            # print(L1_loss)
            Velocity_loss=get_velocity_loss(predict_content_landmark, T_content_landmark)
            heatmap_Velocity_loss =  0.2 * get_heatmap_velocity_loss(predict_whole_heatmap, T_whole_heatmap)

            loss= L1_loss + Velocity_loss + MSE_loss +  heatmap_Velocity_loss

            loss.backward()
            optimizer.step()
            running_MSE_loss+= MSE_loss.item()
            running_L1_loss+=L1_loss.item()
            running_velocity_loss+=Velocity_loss.item()
            running_heatmap_velocity_loss+=heatmap_Velocity_loss.item()


            prog_bar.set_description('epoch: %d step: %d running_L1_loss: %.4f  running_MSE_loss: %.4f  running_velocity_loss: %.4f  hm_velocity_loss: %.4f '
                    % (global_epoch, global_step, running_L1_loss / (step + 1), running_MSE_loss / (step + 1), running_velocity_loss / (step + 1), running_heatmap_velocity_loss / (step + 1)))
            writer.add_scalar('running_L1_loss', running_L1_loss / (step + 1), global_step)
            writer.add_scalar('running_MSE_loss', running_MSE_loss / (step + 1), global_step)
            writer.add_scalar('running_velocity_loss', running_velocity_loss / (step + 1), global_step)
            writer.add_scalar('running_heatmap_velocity_loss', running_heatmap_velocity_loss / (step + 1), global_step)
            global_step += 1
        global_epoch += 1
    print("end")