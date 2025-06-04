import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .conv import Conv2d, Conv1d

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class region_aware_face_encoder(nn.Module):
    def __init__(self):
        super(region_aware_face_encoder, self).__init__()
        
        """self.f_ave_conv = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(7,7), stride=1, padding=3) , nn.ReLU() )
        self.f_max_conv = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(7,7), stride=1, padding=3) , nn.ReLU() )
        self.final_conv = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(7,7), stride=1, padding=3) , nn.ReLU() )"""
        self.f_ave_conv = Conv2d(cin=2, cout=1, kernel_size=(7,7), stride=1, padding=3)
        self.f_max_conv = Conv2d(cin=2, cout=1, kernel_size=(7,7), stride=1, padding=3)
        self.final_conv = Conv2d(cin=2, cout=1, kernel_size=(7,7), stride=1, padding=3)
        
    def forward(self, visual_feature, audio_feature):
        # (B, C_1, 16, 16) (B, C_2, 16, 16)
        visual_ave_pool = torch.mean(visual_feature, dim=1, keepdim=True) # (B, 1, 16, 16)
        audio_ave_pool = torch.mean(audio_feature, dim=1, keepdim=True) # (B, 1, 16, 16)
        
        visual_max_pool, _ = torch.max(visual_ave_pool, dim=1, keepdim=True)
        audio_max_pool, _ = torch.max(audio_ave_pool, dim=1, keepdim=True)
        
        f_ave = torch.cat((visual_ave_pool, audio_ave_pool), dim=1) # (B, 2, 16, 16)
        f_max = torch.cat((visual_max_pool, audio_max_pool), dim=1) # (B, 2, 16, 16)
        
        f_ave = self.f_ave_conv(f_ave)  # (B, 1, 16, 16)
        f_max = self.f_max_conv(f_max)  # (B, 1, 16, 16)
        f_ = torch.cat((f_ave, f_max), dim=1)
        attention_map = self.final_conv(f_) # (B, 1, 16, 16)
        
        visual_feature_attention = visual_feature * attention_map # (B, C_1, 16, 16)
        visual_feature_attention += visual_feature  # residual
        
        return visual_feature_attention
        
class Super_SyncNet_wav2vec(nn.Module):
    def __init__(self):
        super(Super_SyncNet_wav2vec, self).__init__()

        self.w2v_encoder = nn.Sequential(
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
            Conv1d(512, 512, kernel_size=1, stride=1, padding=0),)
        
        self.face_encoder_map = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # (B, 256, 16, 16)
            )
        
        self.face_encoder_embedding = nn.Sequential(
            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=4, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

        self.mel_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)
        
        self.mel_attention_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(2, 1), padding=2),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=5, stride=(2, 1), padding=2),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

        )
        
        # self.audio_fc = Conv2d(cin=1024, cout=512, kernel_size=1, stride=1, padding=0)
        # attention_layers = TransformerEncoderLayer(d_model=512, nhead=2, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.avam = region_aware_face_encoder()
        # self.attention_layer = TransformerEncoder(encoder_layer=attention_layers, num_layers=1)
        self.dropout = nn.Dropout(p=0.1)
        
        ###################################################################################################################################
        
        self.modality = nn.Embedding(4, 512, padding_idx=0)  # 1 for face,  2  for  audio
        attention_layers = TransformerEncoderLayer(d_model=512, nhead=2, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.transformer_layer = TransformerEncoder(encoder_layer=attention_layers, num_layers=2)
        self.act = nn.ReLU()
        ###################################################################################################################################
        self.apply(weight_init)

    def forward(self, mel_input, w2v_input, face_input): # 
        ###########################################################################################################
        
        w2v_embedding = self.w2v_encoder(w2v_input)
        w2v_embedding = w2v_embedding.squeeze(-1).unsqueeze(1)
        # print(w2v_embedding.shape) # (B, 1, 512)
        
        # audio_input_tokens = torch.cat((mel_embedding, w2v_embedding), dim=1) # (B, 1+1, 512)
        # audio_input_tokens = self.dropout(audio_input_tokens)
        # audio_output_embedding = self.attention_layer(audio_input_tokens).view(-1, 1024).unsqueeze(-1).unsqueeze(-1)
        audio_output_embedding = w2v_embedding.view(-1, 512).unsqueeze(-1).unsqueeze(-1)
        # audio_output_embedding = self.audio_fc(audio_input_tokens)
        # print(audio_output_embedding.shape) # (B, 512, 1, 1)
        
        ###########################################################################################################
        mel_input_resize = F.interpolate(mel_input, size=(64,16), mode='bicubic', align_corners=False)
        audio_feature_map = self.mel_attention_encoder(mel_input_resize)
        # print(audio_feature_map.shape)  # (B, 128, 16, 16)
        
        face_feature_map = self.face_encoder_map(face_input)
        face_feature_map_attention = self.avam(face_feature_map, audio_feature_map)
        # print(face_feature_map_attention.shape) # (B, 256, 16, 16)
        
        face_output_embedding = self.face_encoder_embedding(face_feature_map_attention)
        # print(face_output_embedding.shape)  # (B, 512, 1, 1)
        
        
        ###########################################################################################################
        # Transformer
        # modality embedding
        face_modality_embedding = self.modality(1 * torch.ones((face_output_embedding.size(0), 1), dtype=torch.int).to(face_output_embedding.device)) # (B, 1, 512)
        audio_modality_embedding = self.modality(2 * torch.ones((audio_output_embedding.size(0), 1), dtype=torch.int).to(face_output_embedding.device))
        
        face_output_embedding = face_output_embedding.squeeze(-1).squeeze(-1).unsqueeze(1) + face_modality_embedding
        audio_output_embedding = audio_output_embedding.squeeze(-1).squeeze(-1).unsqueeze(1) + audio_modality_embedding

        input_tokens = torch.cat((face_output_embedding, audio_output_embedding), dim=1) # (B, 1+1, 512)
        input_tokens = self.dropout(input_tokens)
        output_tokens = self.transformer_layer(input_tokens)
        
        face_output_embedding = self.act(output_tokens[:,0,:]) #(B,1,512)
        audio_output_embedding = self.act(output_tokens[:,1,:]) #(B,1,512)
        ###########################################################################################################

        face_output_embedding = face_output_embedding.view(face_output_embedding.size(0), -1)
        audio_output_embedding = audio_output_embedding.view(audio_output_embedding.size(0), -1) # (B, 512)
        
        face_output_embedding = F.normalize(face_output_embedding, p=2, dim=1)
        audio_output_embedding = F.normalize(audio_output_embedding, p=2, dim=1)


        return audio_output_embedding, face_output_embedding

        
if __name__ == '__main__':
    import audio
    import numpy as np
    device = torch.device('cuda:0')
    model = Super_SyncNet_wav2vec().to(device).eval()
    mel_input = torch.randn((2,1,80,16)).to(device) 
    face_input = torch.randn((2,15,64,128)).to(device)
    w2v_input = torch.randn((2,768,10)).to(device)
    
    audio_embedding, face_embedding = model(mel_input, w2v_input, face_input)
    print(audio_embedding.shape)
    print(face_embedding.shape)
    

    
    # wavpath = '/data5/store1/dzj/datasets/processed_lrs2_ours/lrs2_audio/6313656042293295351/00001/audio.wav'  
    # wav_input = audio.load_wav(wavpath, sr = 16000)  # 1.536 * 16000 = 24576           5 frame 对应 3200 step size
    # orig_mel = audio.melspectrogram(wav).T     # （123， 80）                     5 frame 对应 16 step size
    # print(orig_mel.shape)
    
    # Load model directly
    # from transformers import AutoModelForPreTraining,  AutoFeatureExtractor 

    # wav_input = torch.randn((1, 24576)).to(device)
    # wav_input = random_array = np.random.rand(1, 24576)
    
    # feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    # model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-xls-r-300m")
    # model = model.to('cuda:0')
    # syncnet = Super_SyncNet_mamba(model)
    # input = feature_extractor(wav_input, sampling_rate=16000, return_tensors="pt")
    # input = input.to('cuda:0')
    # output = model(**input)
    # output_featute = output.projected_quantized_states[0].cpu().detach().numpy()
    # print(output_featute.shape)
    # print(syncnet)
    # out = syncnet(wav_input)
    # print(out.shape)
    # print(model)
    # 1s --- 50dimension  5frame--0.2s--10dimension