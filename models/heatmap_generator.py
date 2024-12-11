import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import torch.nn.functional as F

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class LandmarkDict(dict):# Makes a dictionary that behave like an object to represent each landmark
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

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model=512, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class Conv1d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False,act='ReLU', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv1d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm1d(cout)
                            )
        if act=='ReLU':
            self.act = nn.ReLU()
        elif act=='Tanh':
            self.act =nn.Tanh()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, act='ReLU',*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'Tanh':
            self.act = nn.Tanh()

        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)
    
class ConvT2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, act='ReLU',*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.convT_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'Tanh':
            self.act = nn.Tanh()

        self.residual = residual

    def forward(self, x):
        out = self.convT_block(x)
        if self.residual:
            out += x
        return self.act(out)

class Heatmap_decoder(nn.Module):  
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_t1 = ConvT2d(in_channels, in_channels, kernel_size=4, stride=1, padding=0)
        self.conv_t2 = ConvT2d(in_channels, int(in_channels/2), kernel_size=4, stride=2, padding=1)
        self.conv_t3 = ConvT2d(int(in_channels/2), int(in_channels/4), kernel_size=4, stride=2, padding=1)
        self.conv_t4 = ConvT2d(int(in_channels/4), int(in_channels/8), kernel_size=4, stride=2, padding=1)
        self.conv_t5 = ConvT2d(int(in_channels/8), int(in_channels/16), kernel_size=4, stride=2, padding=1)
        self.conv_t6 = ConvT2d(int(in_channels/16), out_channels, kernel_size=4, stride=2, padding=1)
        # -----------------------------------------------------------------------------------------------------------------------------
        self.conv_t7 = ConvT2d(in_channels, in_channels, kernel_size=4, stride=1, padding=0)
        self.conv_t8 = ConvT2d(in_channels, int(in_channels/2), kernel_size=4, stride=2, padding=1)
        self.conv_t9 = ConvT2d(int(in_channels/2), int(in_channels/4), kernel_size=4, stride=2, padding=1)
        self.conv_t10 = ConvT2d(int(in_channels/4), int(in_channels/8), kernel_size=4, stride=2, padding=1)
        self.conv_t11 = ConvT2d(int(in_channels/8), int(in_channels/16), kernel_size=4, stride=2, padding=1)
        self.conv_t12 = ConvT2d(int(in_channels/16), out_channels, kernel_size=4, stride=2, padding=1)
        
        self.message_l2j = nn.Sequential(
            Conv2d(int(in_channels/2), int(in_channels/2), kernel_size=3, stride=1, padding=1)
        )
        
        self.message_j2l = nn.Sequential(
            Conv2d(int(in_channels/2), int(in_channels/2), kernel_size=3, stride=1, padding=1)
        )
        
        self.message_p2j = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        )

        self.message_p2l = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        )
        
        self.tanh = nn.Tanh()
        
    def forward(self, lip_embedding, jaw_embedding, T_pose_heatmap): # input(B*T,512,1,1) (B*T,1,128,128)
        
        x = self.conv_t1(lip_embedding) # (B*T, 512, 4, 4)
        y = self.conv_t7(jaw_embedding)
        
        x = self.conv_t2(x) # (B*T, 256, 8, 8)
        y = self.conv_t8(y)
        
        # --message passing-- 
        message_l2j = self.message_l2j(x) # (B*T, 256, 8, 8)
        message_j2l = self.message_l2j(y) 
        
        x = x + message_j2l
        y = y + message_l2j
        # -------------------    
        
        x = self.conv_t3(x) # (B*T, 128, 16, 16)
        y = self.conv_t9(y)
         
        x = self.conv_t4(x) # (B*T, 64, 32, 32)
        y = self.conv_t10(y)
        
        # --message passing-- 
        message_p2l = self.message_p2l(T_pose_heatmap) # (B*T, 256, 8, 8)
        message_p2j = self.message_p2j(T_pose_heatmap) 
        x = x + message_p2l
        y = y + message_p2j
        # -------------------   
 
        x = self.conv_t5(x) # (B*T, 32, 64, 64)
        y = self.conv_t11(y)
 
        x = self.conv_t6(x) # (B*T, 1, 128, 128)
        y = self.conv_t12(y)
 
        x = self.tanh(x)
        y = self.tanh(y)
        return x, y
    
class Fusion_transformer_encoder(nn.Module):
    def __init__(self,T, d_model, nlayers, nhead, dim_feedforward,  # 1024   128
                 dropout=0.1):
        super().__init__()
        self.T=T
        self.position_v = PositionalEmbedding(d_model=512)  #for visual landmarks
        self.position_a = PositionalEmbedding(d_model=512)  #for audio embedding
        self.modality = nn.Embedding(4, 512, padding_idx=0)  # 1 for pose,  2  for  audio, 3 for reference landmarks
        self.dropout = nn.Dropout(p=dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self,ref_embedding,mel_embedding,pose_embedding):#(B,Nl,512)  (B,T,512)    (B,T,512)

        # (1).  positional(temporal) encoding
        position_v_encoding = self.position_v(pose_embedding)  # (1,  T, 512)
        position_a_encoding = self.position_a(mel_embedding)

        #(2)  modality encoding
        modality_v = self.modality(1 * torch.ones((pose_embedding.size(0), self.T), dtype=torch.int).to(ref_embedding.device))
        modality_a = self.modality(2 * torch.ones((mel_embedding.size(0),  self.T), dtype=torch.int).to(ref_embedding.device))

        pose_tokens = pose_embedding + position_v_encoding + modality_v    #(B , T, 512 )
        audio_tokens = mel_embedding + position_a_encoding + modality_a    #(B , T, 512 )
        ref_tokens = ref_embedding + self.modality(
            3 * torch.ones((ref_embedding.size(0), ref_embedding.size(1)), dtype=torch.int).to(ref_embedding.device))

        #(3) concat tokens
        input_tokens = torch.cat((ref_tokens, audio_tokens, pose_tokens), dim=1)  # (B, 1+T+T, 512 )
        input_tokens = self.dropout(input_tokens)

        #(4) input to transformer
        output = self.transformer_encoder(input_tokens)
        return output

class Heatmap_generator(nn.Module):
    def __init__(self,T,d_model,nlayers,nhead,dim_feedforward,dropout=0.1,N_l=15):
        super(Heatmap_generator, self).__init__()
        self.mel_encoder = nn.Sequential(
            Conv1d(768, 128, kernel_size=3, stride=1, padding=1),

            Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv1d(512, 512, kernel_size=1, stride=1, padding=0, act='Tanh'),)
        
        self.pose_encoder=nn.Sequential(  #  (B*T,1,128,128)
            Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=3, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=3, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0,act='Tanh'),
            )
        
        self.ref_encoder=nn.Sequential(  #  (B*T,1,128,128)
            Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=3, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=3, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0,act='Tanh'),
            )
        
        self.lip_encoder=nn.Sequential(  #  (B*T,1,128,128)
            Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=3, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=3, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0,act='Tanh'),
            )
        
        self.jaw_encoder=nn.Sequential(  #  (B*T,1,128,128)
            Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=3, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=3, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0,act='Tanh'),
            )
        
        self.heatmap_decoder = Heatmap_decoder(512, 1)
        self.fusion_transformer = Fusion_transformer_encoder(T,d_model,nlayers,nhead,dim_feedforward,dropout)
        self.mouse_keypoint_map = nn.Linear(d_model, 40 * 2)
        self.jaw_keypoint_map = nn.Linear(d_model, 17 * 2)        

        self.apply(weight_init)
        self.Norm=nn.LayerNorm(512)
        self.sigmoid = nn.Sigmoid()


    def forward(self, T_mels, T_pose_heatmap, ref_Nl_whole_heatmap):
                  # (B,T,hv,wv)  (B,T,1,128,128) (B,Nl,1,128,128) 
        B,T,N_l= T_mels.size(0),T_mels.size(1),ref_Nl_whole_heatmap.size(1)

        #1. T_mel_spectrum     reference_heatmap      T_pose_heatmap
        T_mels=torch.cat([T_mels[i] for i in range(T_mels.size(0))],dim=0) # (B*T,hv,wv)
        T_pose_heatmap = torch.cat([T_pose_heatmap[i] for i in range(T_pose_heatmap.size(0))],dim=0)  # (B*T,1,128,128)
        ref_Nl_whole_heatmap = torch.cat([ref_Nl_whole_heatmap[i] for i in range(ref_Nl_whole_heatmap.size(0))], dim=0)  # (B*Nl,1,128,128)

        # 2. get embedding
        mel_embedding=self.mel_encoder(T_mels).squeeze(-1).squeeze(-1)#(B*T,512)
        pose_embedding=self.pose_encoder(T_pose_heatmap).squeeze(-1).squeeze(-1)  # (B*T,512)
        ref_embedding = self.ref_encoder(ref_Nl_whole_heatmap).squeeze(-1).squeeze(-1) # (B*Nl,512)
        
        # normalization
        mel_embedding = self.Norm(mel_embedding)  # (B*T,512)
        pose_embedding =self.Norm(pose_embedding)   # (B*T,512)
        ref_embedding = self.Norm(ref_embedding) # (B*Nl,512)
        # split
        mel_embedding = torch.stack(torch.split(mel_embedding,T),dim=0) #(B,T,512)
        pose_embedding = torch.stack(torch.split(pose_embedding, T), dim=0) # (B,T,512)
        ref_embedding=torch.stack(torch.split(ref_embedding,N_l,dim=0),dim=0) #(B,N_l,512)

        # 3. fuse embedding
        output_tokens=self.fusion_transformer(ref_embedding,mel_embedding,pose_embedding)
        
        # 4.output  embedding
        lip_embedding=output_tokens[:,N_l:N_l+T,:] #(B,T,dim)
        jaw_embedding=output_tokens[:,N_l+T:,:] #(B,T,dim)
        
        # 5.output heatmaps
        lip_embedding=torch.cat([lip_embedding[i] for i in range(lip_embedding.size(0))],dim=0).unsqueeze(-1).unsqueeze(-1) # (B*T, 512, 1, 1)
        jaw_embedding=torch.cat([jaw_embedding[i] for i in range(jaw_embedding.size(0))],dim=0).unsqueeze(-1).unsqueeze(-1)
        
        lip_heatmap, jaw_heatmap = self.heatmap_decoder(lip_embedding, jaw_embedding, T_pose_heatmap) # (B*T, 1, 128, 128)
        
        # 6. output landmarks
        lip_landmarks_embedding = self.lip_encoder(lip_heatmap).squeeze(-1).squeeze(-1) # (B*T, 512)  # 这边可以内部通信
        jaw_landmarks_embedding = self.jaw_encoder(jaw_heatmap).squeeze(-1).squeeze(-1)
        
        # shortcut
        lip_landmarks_embedding = lip_landmarks_embedding + lip_embedding.squeeze(-1).squeeze(-1)
        jaw_landmarks_embedding = jaw_landmarks_embedding + jaw_embedding.squeeze(-1).squeeze(-1)
        
        lip_landmarks = self.mouse_keypoint_map(lip_landmarks_embedding) # (B*T, 40*2)
        jaw_landmarks = self.jaw_keypoint_map(jaw_landmarks_embedding) # (B*T, 17*2)
        
        predict_content_heatmap = lip_heatmap + jaw_heatmap # (B*T, 1, 128, 128)
        
        predict_content_landmark = torch.reshape(torch.cat([jaw_landmarks,lip_landmarks],dim=1),(B*T,-1,2)).permute(0,2,1) #(B*T,2,57)

        return predict_content_heatmap, predict_content_landmark # (B*T, 1, 128, 128), #(B*T,2,57)

if __name__ == '__main__':
    device = torch.device('cuda:0')
    N_l = 5
    B = 2
    T = 5
    hv, hw = 768, 10
    d_model=512
    dim_feedforward=1024
    nlayers=8
    nhead=4
    dropout=0.1 # 0.5
    model = Heatmap_generator(T,d_model,nlayers,nhead,dim_feedforward,dropout,N_l).to(device) # 5, 512, 4, 4, 1024, 0.1, 15
    
    T_mels = torch.randn(B, T, hv, hw).to(device)
    
    T_pose_heatmap = torch.randn(B, T, 1, 128, 128).to(device)
    
    ref_Nl_whole_heatmap = torch.randn(B, T, 1, 128, 128).to(device)
    
    predict_content_heatmap, predict_content_landmark = model(T_mels, T_pose_heatmap, ref_Nl_whole_heatmap)
    print(predict_content_heatmap.shape)
    print(predict_content_landmark.shape)