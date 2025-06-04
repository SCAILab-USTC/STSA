from os.path import join, isfile
from tqdm import tqdm
import logging
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data as data_utils
import torch.nn.functional as F
import numpy as np
from glob import glob
import os, random
import cv2
from piq import psnr, ssim, FID
import face_alignment
from piq.feature_extractors import InceptionV3
from models import define_D
from models import Super_SyncNet_wav2vec as SyncNet
from loss import GANLoss
from models import Face_renderer

import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


parser=argparse.ArgumentParser()
parser.add_argument('--sketch_root',default='./processed_lrs2/lrs2_heatmap_whole')
parser.add_argument('--face_img_root',default='./processed_lrs2/lrs2_face')
parser.add_argument('--audio_root',default='./processed_lrs2s/lrs2_audio')
parser.add_argument('--syncnet_checkpoint_path',default='checkpoints/syncnet/ckp_step000393000.pth')

args=parser.parse_args()
#other parameters
num_workers = 14
Project_name = 'face_gen'   #Project_name
finetune_path = None
finetune_path_disc = None

ref_N = 5
T = 5
print('Project_name:', Project_name)
batch_size = 8 # 64       #### batch_size
batch_size_val = 64 # 64    #### batch_size

mel_step_size = 16  # 16
w2v_step_size = 10 

fps = 25
img_size = 128
FID_batch_size = 1024
evaluate_interval = 1500  #
checkpoint_interval=evaluate_interval
lr = 1e-4
global_step, global_epoch = 0, 0
use_cuda = torch.cuda.is_available()

syncnet_T = 5
syncnet_wt = 0 # 0.1 used for resume 

sketch_root = args.sketch_root
face_img_root = args.face_img_root
audio_root=args.audio_root
filelist_name = 'lrs2'
checkpoint_root = './checkpoints/renderer/'
checkpoint_dir = os.path.join(checkpoint_root, Project_name)
reset_optimizer = False
save_optimizer_state = True
writer = SummaryWriter('tensorboard_runs/Project_{}'.format(Project_name))

criterionFeat = torch.nn.L1Loss()
#--------------------------------------------------------------------------------------------------------------
# syncnet
logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)
    return loss

syncnet = SyncNet().cuda()
for p in syncnet.parameters():
    p.requires_grad = False
    
#--------------------------------------------------------------------------------------------------------------

n_layers_D = 3
num_D = 2
disc = define_D(input_nc=3, ndf=64, n_layers_D=n_layers_D, norm='instance', use_sigmoid=False, num_D=num_D,
                    getIntermFeat=True)
criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
# criterion_L1 = nn.L1Loss()
#evaluate index
fid_metric = FID()
feature_extractor = InceptionV3() #.cuda()

#--------------------------------------------------------------------------------------------------------------

class Dataset(object):
    def get_vid_name_list(self, split):
        filelist = []
        with open('filelists/{}/{}.txt'.format(filelist_name, split)) as f:
            for line in f:
                line = line.strip()
                if ' ' in line: line = line.split()[0]
                filelist.append(line)
        return filelist

    def __init__(self, split):
        min_len = 25
        vid_name_lists= self.get_vid_name_list(split)
        self.available_video_names=[]
        print("filter videos with min len of ",min_len,'....')
        for vid_name in tqdm(vid_name_lists,total=len(vid_name_lists)):
            img_paths = list(glob(join(face_img_root,vid_name, '*.png')))
            vid_len=len(img_paths)
            if  vid_len>= min_len:
                self.available_video_names.append((vid_name,vid_len))
        print("complete,with available vids: ", len(self.available_video_names), '\n')

    def normalize_and_transpose(self, window):
        x = np.asarray(window) / 255.
        x = np.transpose(x, (0, 3, 1, 2))
        return torch.FloatTensor(x)  # B,3,H,W
    def __len__(self):
        return len(self.available_video_names)
    def normalize_and_transpose_heatmap(self, window):
        x = np.asarray(window)
        x = torch.FloatTensor(x).unsqueeze(1)
        x = x.repeat(1,3,1,1)
        return x  # B,3,H,W
    
    def crop_w2v_window(self, w2v, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = start_frame
            
            # start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(50. * (start_frame_num / float(fps)))

        end_idx = start_idx + w2v_step_size

        return w2v[0][start_idx : end_idx, :]
    
    def get_segmented_w2vs(self, w2v, start_frame):
        w2vs = []
        assert syncnet_T == 5
        # start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing  ???.
        start_frame_num = start_frame
        
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            w = self.crop_w2v_window(w2v, i - 2)
            if w.shape[0] != w2v_step_size:
                return None
            w2vs.append(w.T)

        w2vs = np.asarray(w2vs) # T, 768, 10

        return w2vs
    

    def __getitem__(self, idx):
        while 1:
            vid_idx = random.randint(0, len(self.available_video_names) - 1)
            vid_name = self.available_video_names[vid_idx][0]
            vid_len = self.available_video_names[vid_idx][1]
            face_img_paths = list(glob(join(face_img_root,vid_name, '*.png')))
            window_T=5
            
            frame_list = os.listdir(os.path.join(face_img_root, vid_name))
            i_list = []
            for item in frame_list:
                i = int(item.split('/')[-1].replace(".png",''))
                i_list.append(i)
            i_list.sort()
            T_idxs = []
            
            # 1.randomly select a windows of 5 frame
            random_start_idx = random.randint(0 + 2, len(i_list) - window_T - 2)
            for i in range(0, window_T):
                T_idxs.append(i_list[random_start_idx + i])
            # print(T_idxs)
            if T_idxs[-1] - T_idxs[0] != (window_T-1):
                # print("nonono") # ensure continous frames
                continue
            if ( (T_idxs[0] - 2) not in i_list ) or ( (T_idxs[-1] + 2) not in i_list ):
                # ensure window sequences all exist
                continue
            
            # 2. read  face image and sketch
            T_face_paths = [os.path.join(face_img_root, vid_name, str(idx) + '.png') for idx in T_idxs]
            
            ref_N_fpaths = random.sample(list(set(face_img_paths) - set(T_face_paths)), ref_N)

            
            T_frame_img=[]
            T_frame_sketch = []
            for img_path in T_face_paths:
                sketch_path = os.path.join(sketch_root,
                            '/'.join(img_path.split('/')[-3:])).replace('png', 'npy')
                if os.path.isfile(img_path)  and os.path.isfile(sketch_path):
                    T_frame_img.append(cv2.resize(cv2.imread(img_path),(img_size,img_size)))
                    T_frame_sketch.append(np.load(sketch_path)) # (128,128)
                else:
                    break
            if len(T_frame_img)!=window_T:  #T (H,W,3)
                continue 
            ########################################################
            

            ref_N_frame_img,ref_N_frame_sketch = [],[]
            for img_path in ref_N_fpaths:
                sketch_path = os.path.join(sketch_root,
                                           '/'.join(img_path.split('/')[-3:])).replace('png', 'npy')
                if os.path.isfile(img_path)  and os.path.isfile(sketch_path):
                    ref_N_frame_img.append(cv2.resize(cv2.imread(img_path),(img_size,img_size)))
                    ref_N_frame_sketch.append(np.load(sketch_path))
                else:
                    break
            if len(ref_N_frame_img) != ref_N:  # ref_N (H,W,3)
                continue

            T_frame_img = self.normalize_and_transpose(T_frame_img)  #: T,3,H,W
            T_frame_sketch = self.normalize_and_transpose_heatmap(T_frame_sketch)  #: 5,3,H,W -> target: T, 5, 3, H, W

            ref_N_frame_img = self.normalize_and_transpose(ref_N_frame_img)  # ref_N,3,H,W
            ref_N_frame_sketch = self.normalize_and_transpose_heatmap(ref_N_frame_sketch)  # ref_N,3,H,W

            # 3. get window audio feat
            try:
                melpath = join(audio_root,vid_name, "audio.npy")
                audio_mel = np.load(melpath)
                
                w2vpath = melpath.replace("audio.npy", "wav2vec2_768.npy")
                orig_w2v = np.load(w2vpath)
            except Exception as e:
                continue
            frame_idx=T_idxs[2]
            mel_start_frame_idx = frame_idx - 2  ###around the frame idx
            if mel_start_frame_idx < 0:
                continue
            start_idx = int(80. * (mel_start_frame_idx / float(fps)))
            m = audio_mel[start_idx: start_idx + mel_step_size, :]  # get five frame around
            if m.shape[0] != mel_step_size:  # in the end of vid
                continue
            T_mels = m.T  # (hv,wv)
            T_mels = torch.FloatTensor(T_mels).unsqueeze(0) # (1,hv,wv)
            
            # 4. get window w2v
            w2v = self.crop_w2v_window(orig_w2v.copy(), T_idxs[0]) # arround the target frame
            if (w2v.shape[0] != w2v_step_size):
                continue
            T_w2v = torch.FloatTensor(w2v.T) # (768, 10)            
            # 5. get 5 windows w2v
            indiv_w2vs = self.get_segmented_w2vs(orig_w2v.copy(), T_idxs[0])
            if indiv_w2vs is None: continue
            w2v_sequences = torch.FloatTensor(indiv_w2vs)  # (5, 768, 10) 

            return T_frame_img,  T_frame_sketch,  ref_N_frame_img,  ref_N_frame_sketch, T_mels,   T_w2v,   w2v_sequences
            #       (T,3,H,W)     (T,3,H,W)         (ref_N,3,H,W)      (ref_N,3,H,W)    (1,hv,wv) (768,10) (T,768,10)

def load_checkpoint(path, model, optimizer, scheduler, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch
    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '',1)] = v  #
    # for k, v in s.items():
    #     new_s['module.'+k] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        scheduler_state = checkpoint["scheduler"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler_state is not None:
            print("Load scheduler state from {}".format(path))
            scheduler.load_state_dict(checkpoint["scheduler"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]
    return model

def save_checkpoint(model, optimizer, scheduler, step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(
        checkpoint_dir, "{}_epoch_{}_checkpoint_step{:09d}.pth".format(prefix, epoch, global_step))
    if isfile(checkpoint_path):
        os.remove(checkpoint_path)
    optimizer_state = optimizer.state_dict() if save_optimizer_state else None
    scheduler_state = scheduler.state_dict() if save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def get_sync_loss(mel, w2v, g):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * 5, H//2, W
    a, v = syncnet(mel, w2v, g)
    y = torch.ones(g.size(0), 1).float().cuda()
    return cosine_loss(a, v, y)

def compute_generation_quality(gt, fake_image):  # (B*T,3,96,96)   (B*T,3,96,96) cuda
    global global_step
    psnr_values = []
    ssim_values = []
    #############PSNR###########
    psnr_value = psnr(fake_image, gt, reduction='none')
    psnr_values.extend([e.item() for e in psnr_value])
    #############SSIM###########
    ssim_value = ssim(fake_image, gt, data_range=1., reduction='none')
    ssim_values.extend([e.item() for e in ssim_value])

    #############FID###########
    B_mul_T = fake_image.size(0)
    total_images = torch.cat((gt, fake_image), 0)
    if len(total_images) > FID_batch_size:
        total_images = torch.split(total_images, FID_batch_size, 0)
    else:
        total_images = [total_images]

    total_feats = []
    for sub_images in total_images:
        sub_images = sub_images.cuda()
        feats = fid_metric.compute_feats([
            {'images': sub_images},
        ], feature_extractor=feature_extractor)
        feats = feats.detach()
        total_feats.append(feats)
    total_feats = torch.cat(total_feats, 0)
    gt_feat, pd_feat = torch.split(total_feats, (B_mul_T, B_mul_T), 0)

    gt_feats = gt_feat.cuda()
    pd_feats = pd_feat.cuda()

    fid = fid_metric.compute_metric(pd_feats, gt_feats).item()
    return np.asarray(psnr_values).mean(), np.asarray(ssim_values).mean(), fid


def save_sample_images_gen(T_frame_sketch, ref_N_frame_img, generated_img, gt, global_step, checkpoint_dir):
    #                        (B,T,3,H,W)  (B,ref_N,3,H,W)  (B*T,3,H,W) (B*T,3,H,W) (B*T,3,H,W)
    ref_N_frame_img = ref_N_frame_img.unsqueeze(1).expand(-1, T, -1, -1, -1, -1)  # (B,T,ref_N,3,H,W)
    ref_N_frame_img = (ref_N_frame_img.cpu().numpy().transpose(0, 1, 2, 4, 5, 3) * 255.).astype(np.uint8)  # ref: (B,T,ref_N,H,W,3)

    fake_image = torch.stack(torch.split(generated_img, T, dim=0), dim=0)  #(B,T,3,H,W)
    fake_image = (fake_image.clone().detach().cpu().numpy().transpose(0, 1, 3, 4, 2) * 255.).astype(np.uint8)  # (B,T,H,W,3)


    gt = torch.stack(torch.split(gt, T, dim=0), dim=0)  # (B,T,3,H,W)
    gt = (gt.cpu().numpy().transpose(0, 1, 3, 4, 2) * 255.).astype(np.uint8)  # (B,T,H,W,3)

    T_frame_sketch=(T_frame_sketch[:,2].unsqueeze(1).cpu().numpy().transpose(0, 1, 3, 4, 2) * 255.).astype(np.uint8)  # (B,T,H,W,3)

    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder):
        os.mkdir(folder)
    collage = np.concatenate((T_frame_sketch, *[ref_N_frame_img[:, :, i] for i in range(ref_N_frame_img.shape[2])], fake_image, gt),
                             axis=-2)
    for batch_idx, c in enumerate(collage):   # require (B,T,H,W,3)
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.png'.format(folder, batch_idx, t), c[t])

def evaluate(model, val_data_loader):
    global global_epoch, global_step
    eval_epochs = 5
    print('Evaluating model for {} epochs'.format(eval_epochs))
    eval_gen_loss,eval_sync_loss, eval_mine_loss = 0.,0.,0.
    count = 0
    psnrs, ssims, fids = [], [], []
    for epoch in range(eval_epochs):
        prog_bar = tqdm(enumerate(val_data_loader), total=len(val_data_loader))
        for step, (T_frame_img, T_frame_sketch, ref_N_frame_img, ref_N_frame_sketch, T_mels, T_w2v, w2v_sequences) in prog_bar:
            #     (B,T,3,H,W)     (B,T,3,H,W)     (B,ref_N,3,H,W)  (B,ref_N,3,H,W) (B,1,hv,wv) (B,768,10) (B,T,768,10)
            model.eval()
            
            T_frame_img, T_frame_sketch, ref_N_frame_img, ref_N_frame_sketch, T_mels, T_w2v, w2v_sequences = \
                T_frame_img.cuda(non_blocking=True), T_frame_sketch.cuda(non_blocking=True), ref_N_frame_img.cuda(non_blocking=True),\
                    ref_N_frame_sketch.cuda(non_blocking=True), T_mels.cuda(non_blocking=True), T_w2v.cuda(non_blocking=True), w2v_sequences.cuda(non_blocking=True)
            
            generated_img, perceptual_gen_loss, ref_spatial_deformed_feats, ref_temporal_deformed_feats, mine_loss_total\
                = model(T_frame_img, T_frame_sketch, ref_N_frame_img, ref_N_frame_sketch, w2v_sequences)# (B,3,T,H,W)
                        
            gt = torch.cat([T_frame_img[:, i] for i in range(T_frame_img.size(1))], dim=0) # (B*T, 3, 128, 128)
            gen_merge = torch.cat([generated_img[:,:, i] for i in range(generated_img.size(2))], dim=0)
            
            perceptual_gen_loss = perceptual_gen_loss.sum()
            sync_loss = get_sync_loss(T_mels, T_w2v, generated_img)

            mine_loss_total = mine_loss_total.sum()

            eval_gen_loss += perceptual_gen_loss.item()
            eval_sync_loss += sync_loss.item()
            eval_mine_loss += mine_loss_total.item()
            count += 1
            
            #--------------save img everytime-----------------
            ref_N_frame_img = ref_N_frame_img.permute(0,2,1,3,4) # (B, C, T, H, W)
            
            T_frame_sketch_ = T_frame_sketch.permute(0,2,1,3,4) # (B, C, T, H, W)
            
            T_frame_img_ = T_frame_img.permute(0,2,1,3,4) # (B, C, T, H, W)
            save_real_time_result(ref_N_frame_img, T_frame_sketch_, generated_img, T_frame_img_)
            #########compute evaluation index ###########
            psnr, ssim, fid = compute_generation_quality(gt, gen_merge)
            psnrs.append(psnr)
            ssims.append(ssim)
            fids.append(fid)
    psnr, ssim, fid= np.asarray(psnrs).mean(), np.asarray(ssims).mean(), np.asarray(fids).mean()
    print('psnr %.3f ssim %.3f fid %.3f'% (psnr, ssim, fid))
    logger.info(f'Epoch:{global_epoch}--Step:{global_step}--psnr:{psnr}--ssim:{ssim}--fid:{fid}')

    print('eval_gen_loss : ', eval_gen_loss / count, 'eval_sync_loss : ', eval_sync_loss / count ,'eval_mine_loss:', eval_mine_loss / count)
    logger.info(f'Epoch:{global_epoch}--Step:{global_step}--eval_gen_loss:{eval_gen_loss / count}--eval_sync_loss:{eval_sync_loss / count}--eval_mine_loss:{eval_mine_loss / count}')

    return eval_sync_loss / count

def _load_sync(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path,  map_location='cpu')
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint_sync(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load_sync(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
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

def save_real_time_result(reference_sequences, target_heatmap, g, target_face_sequences):
    # B, 3, 5 ,128, 128 
    
    reference_sequences = (reference_sequences[0].detach().cpu().numpy().transpose(1, 2, 3, 0) * 255.).astype(np.uint8) # T, 128, 128, 3
    target_heatmap = (target_heatmap[0].detach().cpu().numpy().transpose(1, 2, 3, 0) * 255.).astype(np.uint8)
    g = (g[0].detach().cpu().numpy().transpose(1, 2, 3, 0) * 255.).astype(np.uint8)
    target_face_sequences = (target_face_sequences[0].detach().cpu().numpy().transpose(1, 2, 3, 0) * 255.).astype(np.uint8) # (ref_num=5, 128, 128, 3)
    

    con_ref_face = np.concatenate([reference_sequences[i] for i in range(reference_sequences.shape[0])], axis=1)
    con_tar_heatmap = np.concatenate([target_heatmap[i] for i in range(target_heatmap.shape[0])], axis=1)
    con_g = np.concatenate([g[i] for i in range(g.shape[0])], axis=1)
    con_tar_face = np.concatenate([target_face_sequences[i] for i in range(target_face_sequences.shape[0])], axis=1)

    merge = np.concatenate((con_ref_face, con_tar_heatmap, con_g, con_tar_face), axis=0)
    save_path = 'training_logs/face/{}.jpg'.format(Project_name)  
    cv2.imwrite(save_path, merge) 

if __name__ == '__main__':
    
    # create a logger  
    logger = logging.getLogger('trainging_logger')  
    logger.setLevel(logging.INFO)  
    
    # create a handler
    log_file = 'training_logs/face/{}.log'.format(Project_name)  
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
    model = Face_renderer().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5000, gamma=0.95)

    disc = disc.cuda()
    disc_optimizer = optim.Adam([p for p in disc.parameters() if p.requires_grad],lr=1e-4, betas=(0.5, 0.999))
    disc_scheduler = torch.optim.lr_scheduler.StepLR(disc_optimizer, step_size = 5000, gamma=0.95)
    

    
    if finetune_path is not None:  ###fine tune
        load_checkpoint(finetune_path, model, optimizer, scheduler, reset_optimizer=False, overwrite_global_states=False)
    if finetune_path_disc is not None:  ###fine tune
        load_checkpoint(finetune_path_disc, disc, disc_optimizer, disc_scheduler, reset_optimizer=False, overwrite_global_states=False)
    
    for param_group in optimizer.param_groups:
        print(f"Learning rate after loading: {param_group['lr']}")

    
    load_checkpoint_sync(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, overwrite_global_states=False)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        disc = nn.DataParallel(disc)
        
    # create dataset
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
        running_gen_loss, running_flow_cons_loss, running_sync_loss, running_mine_loss = 0.,0.,0.,0.
            
        for step, (T_frame_img, T_frame_sketch, ref_N_frame_img, ref_N_frame_sketch, T_mels, T_w2v, w2v_sequences) in prog_bar:
            #       (B,T,3,H,W)     (B,T,3,H,W)     (B,ref_N,3,H,W)  (B,ref_N,3,H,W) (B,1,hv,wv) (B,768,10) (B,T,768,10)
            model.train()
            disc.train()
            optimizer.zero_grad()
            disc_optimizer.zero_grad()
            
            T_frame_img, T_frame_sketch, ref_N_frame_img, ref_N_frame_sketch, T_mels, T_w2v, w2v_sequences = \
                T_frame_img.cuda(non_blocking=True), T_frame_sketch.cuda(non_blocking=True), ref_N_frame_img.cuda(non_blocking=True),\
                    ref_N_frame_sketch.cuda(non_blocking=True), T_mels.cuda(non_blocking=True), T_w2v.cuda(non_blocking=True), w2v_sequences.cuda(non_blocking=True)
            
            generated_img, perceptual_gen_loss, ref_spatial_deformed_feats, ref_temporal_deformed_feats, mine_loss_total\
                = model(T_frame_img, T_frame_sketch, ref_N_frame_img, ref_N_frame_sketch, w2v_sequences)# (B*T,3,H,W)
            # (B, C, T, H, W)
            
            
            gt = torch.cat([T_frame_img[:, i] for i in range(T_frame_img.size(1))], dim=0) # (B*T, 3, 128, 128)
            gen_merge = torch.cat([generated_img[:, :, i] for i in range(generated_img.size(2))], dim=0) # (B*T, 3, 128, 128)
            #--------------save img everytime-----------------
            ref_N_frame_img = ref_N_frame_img.permute(0,2,1,3,4) # (B, C, T, H, W)
            
            T_frame_sketch_ = T_frame_sketch.permute(0,2,1,3,4) # (B, C, T, H, W)
            
            T_frame_img_ = T_frame_img.permute(0,2,1,3,4) # (B, C, T, H, W)
            save_real_time_result(ref_N_frame_img, T_frame_sketch_, generated_img, T_frame_img_)

            #---------------perceptual loss------------------- 
            perceptual_gen_loss=perceptual_gen_loss.sum()
            mine_loss_total = mine_loss_total.sum()
            #-------------------------------------------------
            
            #----------GAN loss, sync loss, MI loss-----------
            # 1. GAN loss
            # discriminator
            pred_fake = disc.forward(gen_merge.detach()) 
            loss_D_fake = criterionGAN(pred_fake, False)
            # Real Detection and Loss
            pred_real = disc.forward(gt.clone().detach())
            loss_D_real = criterionGAN(pred_real, True)
            loss_D = (loss_D_fake + loss_D_real).mean() * 0.5
            #
            # GAN loss
            pred_fake = disc.forward(gen_merge)
            loss_G_GAN = criterionGAN(pred_fake, True).mean()
            # GAN feature matching loss
            loss_G_GAN_Feat = 0
            feat_weights = 4.0 / (n_layers_D + 1)
            D_weights = 1.0 / num_D
            for i in range(num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                                       criterionFeat(pred_fake[i][j], pred_real[i][j].detach()).mean() * 2.5
            # 2. syncloss
            if syncnet_wt > 0.:
                sync_loss = get_sync_loss(T_mels, T_w2v, generated_img)
            else:
                sync_loss = 0.
            #-----------------------------------------------
            if global_epoch>5:
                loss = 4*perceptual_gen_loss+0.1*2.5*loss_G_GAN+ loss_G_GAN_Feat + syncnet_wt * sync_loss + 0.01 * mine_loss_total
            else:
                loss = 4*perceptual_gen_loss + syncnet_wt * sync_loss + 0.01 * mine_loss_total 

            # update model:
            loss.backward()
            optimizer.step()
            scheduler.step()
            # update discriminator weights:
            loss_D.backward()
            disc_optimizer.step()
            disc_scheduler.step()
            
            ##log#
            running_gen_loss += perceptual_gen_loss.item()
            if syncnet_wt > 0.:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.
            running_mine_loss += mine_loss_total.item()
            if global_step % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, checkpoint_dir, global_epoch, prefix=Project_name)
                save_checkpoint(disc, disc_optimizer, disc_scheduler, global_step, checkpoint_dir, global_epoch, prefix=(Project_name+'disc'))
                    
            if  (global_step % evaluate_interval == 0 or global_step == 100 or global_step == 500) and global_step!=-1:
                with torch.no_grad():
                    average_sync_loss = evaluate(model, val_data_loader)
                    if average_sync_loss < 2: # 1 -> 2
                        syncnet_wt = 0.1
            prog_bar.set_description('epoch: %d step: %d loss_gen: %.4f  loss_GAN: %.4f  loss_Feat:  %.4f  loss_sync:  %.4f  loss_mine:  %.4f' \
                                     % (global_epoch,global_step, running_gen_loss / (step + 1), loss_G_GAN, loss_G_GAN_Feat, running_sync_loss/ (step + 1),\
                                        running_mine_loss / (step + 1)))
            global_step += 1
        global_epoch += 1
print("end")