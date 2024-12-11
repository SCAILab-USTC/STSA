from torch.nn import functional as F
import torch
import torch.nn as nn
import torchvision
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def make_coordinate_grid_3d(spatial_size, type):
    '''
        generate 3D coordinate grid
    '''
    d, h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    z = torch.arange(d).type(type)
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    z = (2 * (z / (d - 1)) - 1)
    yy = y.view(1,-1, 1).repeat(d,1, w)
    xx = x.view(1,1, -1).repeat(d,h, 1)
    zz = z.view(-1,1,1).repeat(1,h,w)
    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3)], 3)
    return meshed,zz

def downsample(x, size):
    if len(x.size()) == 5:
        size = (x.size(2), size[0], size[1])
        return torch.nn.functional.interpolate(x, size=size, mode='nearest')
    return  torch.nn.functional.interpolate(x, size=size, mode='nearest')

def convert_flow_to_deformation(flow):
    r"""convert flow fields to deformations.
    Args:
        flow (tensor): Flow field obtained by the model
    Returns:
        deformation (tensor): The deformation used for warpping
    """
    b, c, h, w = flow.shape
    flow_norm = 2 * torch.cat([flow[:, :1, ...] / (w - 1), flow[:, 1:, ...] / (h - 1)], 1)
    grid = make_coordinate_grid(flow)
    deformation = grid + flow_norm.permute(0, 2, 3, 1)
    return deformation

def make_coordinate_grid(flow):
    r"""obtain coordinate grid with the same size as the flow filed.
    Args:
        flow (tensor): Flow field obtained by the model
    Returns:
        grid (tensor): The grid with the same size as the input flow
    """
    b, c, h, w = flow.shape

    x = torch.arange(w).to(flow)
    y = torch.arange(h).to(flow)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    meshed = meshed.expand(b, -1, -1, -1)
    return meshed

def warping(source_image, deformation):
    r"""warp the input image according to the deformation
    Args:
        source_image (tensor): source images to be warpped
        deformation (tensor): deformations used to warp the images; value in range (-1, 1)
    Returns:
        output (tensor): the warpped images
    """
    _, h_old, w_old, _ = deformation.shape
    _, _, h, w = source_image.shape
    if h_old != h or w_old != w:
        deformation = deformation.permute(0, 3, 1, 2)
        deformation = torch.nn.functional.interpolate(deformation, size=(h, w), mode='bilinear')
        deformation = deformation.permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(source_image, deformation)

class MINE(nn.Module):
    def __init__(self, input_size=256, hidden_size=128):
        super(MINE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant_(self.fc3.bias, 0)
        
    # Donsker-Varadhan (DV) bound loss
    def mine_loss(self, joint, marginal):
        joint_term = torch.mean(joint)
        marginal_term = torch.log(torch.mean(torch.exp(marginal)))
        return -(joint_term - marginal_term)

    def forward(self, x, y):
        # 连接两个输入的特征
        xy = torch.cat((x, y), dim=1)
        h = F.elu(self.fc1(xy))
        h = F.elu(self.fc2(h))
        return self.fc3(h)

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

class SameBlock2d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features, out_features,  kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out

class DownBlock2d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=2):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out

class ResBlock2d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features,out_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                               padding=padding)
        if out_features != in_features:
            self.channel_conv = nn.Conv2d(in_features,out_features,1)
        self.norm1 = BatchNorm2d(in_features)
        self.norm2 = BatchNorm2d(in_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.in_features != self.out_features:
            out += self.channel_conv(x)
        else:
            out += x
        return out

class AdaINLayer(nn.Module):
    def __init__(self, input_nc, modulation_nc):
        super().__init__()

        self.InstanceNorm2d = nn.InstanceNorm2d(input_nc, affine=False)

        nhidden = 128
        use_bias=True

        self.mlp_shared = nn.Sequential(
            nn.Linear(modulation_nc, nhidden, bias=use_bias),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, input_nc, bias=use_bias)
        self.mlp_beta = nn.Linear(nhidden, input_nc, bias=use_bias)

    def forward(self, input, modulation_input):

        # Part 1. generate parameter-free normalized activations
        normalized = self.InstanceNorm2d(input)

        # Part 2. produce scaling and bias conditioned on feature
        modulation_input = modulation_input.view(modulation_input.size(0), -1)
        actv = self.mlp_shared(modulation_input)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        gamma = gamma.view(*gamma.size()[:2], 1,1)
        beta = beta.view(*beta.size()[:2], 1,1)
        out = normalized * (1 + gamma) + beta
        return out

class AdaIN(torch.nn.Module):

    def __init__(self, input_channel, modulation_channel,kernel_size=3, stride=1, padding=1):
        super(AdaIN, self).__init__()
        self.conv_1 = torch.nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = torch.nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.adain_layer_1 = AdaINLayer(input_channel, modulation_channel)
        self.adain_layer_2 = AdaINLayer(input_channel, modulation_channel)

    def forward(self, x, modulation):

        x = self.adain_layer_1(x, modulation)
        x = self.leaky_relu(x)
        x = self.conv_1(x)
        x = self.adain_layer_2(x, modulation)
        x = self.leaky_relu(x)
        x = self.conv_2(x)

        return x

class SPADELayer(torch.nn.Module):
    def __init__(self, input_channel, modulation_channel, hidden_size=256, kernel_size=3, stride=1, padding=1):
        super(SPADELayer, self).__init__()
        self.instance_norm = torch.nn.InstanceNorm2d(input_channel)

        self.conv1 = torch.nn.Conv2d(modulation_channel, hidden_size, kernel_size=kernel_size, stride=stride,
                                     padding=padding)
        self.gamma = torch.nn.Conv2d(hidden_size, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.beta = torch.nn.Conv2d(hidden_size, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, input, modulation):
        norm = self.instance_norm(input)

        conv_out = self.conv1(modulation)

        gamma = self.gamma(conv_out)
        beta = self.beta(conv_out)

        return norm + norm * gamma + beta

class SPADE(torch.nn.Module):
    def __init__(self, num_channel, num_channel_modulation, hidden_size=256, kernel_size=3, stride=1, padding=1):
        super(SPADE, self).__init__()
        self.conv_1 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.spade_layer_1 = SPADELayer(num_channel, num_channel_modulation, hidden_size, kernel_size=kernel_size,
                                        stride=stride, padding=padding)
        self.spade_layer_2 = SPADELayer(num_channel, num_channel_modulation, hidden_size, kernel_size=kernel_size,
                                        stride=stride, padding=padding)

    def forward(self, input, modulations):
        input = self.spade_layer_1(input, modulations)
        input = self.leaky_relu(input)
        input = self.conv_1(input)
        input = self.spade_layer_2(input, modulations)
        input = self.leaky_relu(input)
        input = self.conv_2(input)
        return input

class AdaAT(nn.Module):
    '''
       AdaAT operator
    '''
    def __init__(self,  para_ch,feature_ch):
        super(AdaAT, self).__init__()
        self.para_ch = para_ch
        self.feature_ch = feature_ch
        self.commn_linear = nn.Sequential(
            nn.Linear(para_ch, para_ch),
            nn.ReLU()
        )
        self.scale = nn.Sequential(
                    nn.Linear(para_ch, feature_ch),
                    nn.Sigmoid()
                )
        self.rotation = nn.Sequential(
                nn.Linear(para_ch, feature_ch),
                nn.Tanh()
            )
        self.translation = nn.Sequential(
                nn.Linear(para_ch, 2 * feature_ch),
                nn.Tanh()
            )
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, feature_map,para_code):
        batch,d, h, w = feature_map.size(0), feature_map.size(1), feature_map.size(2), feature_map.size(3)
        para_code = self.commn_linear(para_code)
        scale = self.scale(para_code).unsqueeze(-1) * 2
        angle = self.rotation(para_code).unsqueeze(-1) * 3.14159#
        rotation_matrix = torch.cat([torch.cos(angle), -torch.sin(angle), torch.sin(angle), torch.cos(angle)], -1)
        rotation_matrix = rotation_matrix.view(batch, self.feature_ch, 2, 2)
        translation = self.translation(para_code).view(batch, self.feature_ch, 2)
        grid_xy, grid_z = make_coordinate_grid_3d((d, h, w), feature_map.type())
        grid_xy = grid_xy.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).repeat(batch, 1, 1, 1)
        scale = scale.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
        rotation_matrix = rotation_matrix.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1, 1)
        translation = translation.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
        trans_grid = torch.matmul(rotation_matrix, grid_xy.unsqueeze(-1)).squeeze(-1) * scale + translation
        full_grid = torch.cat([trans_grid, grid_z.unsqueeze(-1)], -1)
        trans_feature = F.grid_sample(feature_map.unsqueeze(1), full_grid).squeeze(1)
        return trans_feature

class ref_imgs_encoder(torch.nn.Module):
    def __init__(self):
        super(ref_imgs_encoder, self).__init__()
        self.conv1 = nn.Sequential(
            SameBlock2d(15, 32, kernel_size=3, padding=1),
            ResBlock2d(32, 32, kernel_size=3, padding=1),
            ResBlock2d(32, 32, kernel_size=3, padding=1),   # 128*128*32
            
        )
        self.conv2 = nn.Sequential(
            DownBlock2d(32, 64, kernel_size=3, padding=1),
            SameBlock2d(64, 128, kernel_size=3, padding=1),
            ResBlock2d(128, 128, kernel_size=3, padding=1),
            ResBlock2d(128, 128, kernel_size=3, padding=1),  # 64*64*128
        )
        self.conv3 = nn.Sequential(
            DownBlock2d(128, 256, kernel_size=3, padding=1),
            SameBlock2d(256, 256, kernel_size=3, padding=1),
            ResBlock2d(256, 256, kernel_size=3, padding=1),
            ResBlock2d(256, 256, kernel_size=3, padding=1),  # 32*32*256
        )
    
    def forward(self, x):
        x = torch.cat([x[:,:,i] for i in range(x.size(2))], dim=1)  # (B, 3*5, 128, 128)
        
        x = self.conv1(x)
        ref_feat_128 = x
        x = self.conv2(x)
        ref_feat_64 = x
        x = self.conv3(x)
        ref_feat_32 = x
        
        return [ref_feat_128, ref_feat_64, ref_feat_32]

class audio_encoer_w2v(torch.nn.Module):
    def __init__(self):
        super(audio_encoer_w2v, self).__init__()
        self.w2v_encoder = nn.Sequential(
                Conv1d(768, 128, kernel_size=3, stride=1, padding=1),

                Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
                Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                
                Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
                Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                
                Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv1d(128, 128, kernel_size=3, stride=1, padding=0), # 128, 1
                
                )
    def forward(self, x):
        audio_feature = self.w2v_encoder(x) # 128,1
        
        return audio_feature
        
class Heatmap_encoder_tar(torch.nn.Module):
    def __init__(self):
        super(Heatmap_encoder_tar,self).__init__()
        
        self.heatmap_down = nn.Sequential(
            SameBlock2d(3, 32, kernel_size=7, padding=3),
            DownBlock2d(32, 32, kernel_size=3, padding=1),
            DownBlock2d(32, 32, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        x = self.heatmap_down(x) # #  32chennels, 32*32
        return x
    
class Heatmap_encoder_ref(torch.nn.Module):
    def __init__(self):
        super(Heatmap_encoder_ref,self).__init__()
        
        self.heatmap_down = nn.Sequential(
            SameBlock2d(3*5, 32, kernel_size=7, padding=3),
            DownBlock2d(32, 32, kernel_size=3, padding=1),
            DownBlock2d(32, 32, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        x = self.heatmap_down(x) # #  32chennels, 32*32
        return x
      
class SpatialNetwork(torch.nn.Module):
    def __init__(self):
        super(SpatialNetwork, self).__init__()
        
        self.alignment_encoder = nn.Sequential(
            SameBlock2d(256+32+32, 128, kernel_size=3, padding=1),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            DownBlock2d(128, 128, 3, 1, 3),
            
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, 3, 1, 2),
            
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, 3, 1, 2),
            
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, 3, 0, 2)
        )
        
        self.global_avg2d_audio = nn.AdaptiveAvgPool2d(1)
        self.global_avg2d_alignment = nn.AdaptiveAvgPool2d(1)
        
        self.adaAT_32 = AdaAT(256, 256)
        self.adaAT_64 = AdaAT(256, 128)
        self.adaAT_128 = AdaAT(256, 32)
        
        self.adaAT = nn.ModuleList([self.adaAT_128, self.adaAT_64, self.adaAT_32])
        
        appearance_conv_list_32 = []
        for i in range(2):
            appearance_conv_list_32.append(
                nn.Sequential(
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                )
            )
        appearance_conv_list_64 = []
        for i in range(2):
            appearance_conv_list_64.append(
                nn.Sequential(
                    ResBlock2d(128, 128, 3, 1),
                    ResBlock2d(128, 128, 3, 1),
                )
            )
        appearance_conv_list_128 = []
        for i in range(2):
            appearance_conv_list_128.append(
                nn.Sequential(
                    ResBlock2d(32, 32, 3, 1),
                    ResBlock2d(32, 32, 3, 1),
                )
            )
        self.appearance_conv_list_32 = nn.ModuleList(appearance_conv_list_32)
        self.appearance_conv_list_64 = nn.ModuleList(appearance_conv_list_64)
        self.appearance_conv_list_128 = nn.ModuleList(appearance_conv_list_128)
        
        self.appearance_conv_list = nn.ModuleList([self.appearance_conv_list_128, self.appearance_conv_list_64, self.appearance_conv_list_32])


    def forward(self, ref_img_feats, ref_heatmap_feats, T_driving_heatmap_feats, T_audio_feats): #to output:  # (B, 128, 32, 32)  
        #   [(B, 32, 128, 128), (B, 32, 64, 64), (B, 256, 32, 32),]  (B, 32, 32, 32) (B, 32, 32, 32)   # audio:(B, 128, 1) 

        alignment_feature = torch.cat([ref_img_feats[2], ref_heatmap_feats, T_driving_heatmap_feats], dim=1)  # (B, 256+32+32, 32, 32)
        alignment_feature = self.alignment_encoder(alignment_feature)  # (B, 128, 1, 1)
        alignment_feature = self.global_avg2d_alignment(alignment_feature).squeeze(-1).squeeze(-1) # (B, 128)

        ## concat alignment feature and audio feature        
        spatial_feature = torch.cat([T_audio_feats.squeeze(-1), alignment_feature], 1) # (B, 256)
        
        ref_spatial_deformed_feats = []
        
        for i, ref_img_feat in enumerate(ref_img_feats):
            
            ## use AdaAT do spatial deformation on reference feature maps
            ref_spatial_deformed_feature = self.appearance_conv_list[i][0](ref_img_feat)  
            ref_spatial_deformed_feature = self.adaAT[i](ref_spatial_deformed_feature, spatial_feature) # (B, 128, 32, 32)  
            ref_spatial_deformed_feature = self.appearance_conv_list[i][1](ref_spatial_deformed_feature)
            ref_spatial_deformed_feats.append(ref_spatial_deformed_feature)
            # 
            # print(ref_spatial_deformed_feature.shape) 
        
        return ref_spatial_deformed_feats # [(B, 32, 128, 128), (B, 128, 64, 64), (B, 256, 32, 32),]

class TemporalNetwork(torch.nn.Module):
    def __init__(self, num_channel=6, num_channel_modulation=3, hidden_size=256):
        super(TemporalNetwork, self).__init__()

        # Convolutional Layers
        self.conv1 = torch.nn.Conv2d(num_channel, 32, kernel_size=7, stride=1, padding=3)
        self.conv1_bn = torch.nn.BatchNorm2d(num_features=32, affine=True)
        self.conv1_relu = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_bn = torch.nn.BatchNorm2d(num_features=128, affine=True)
        self.conv2_relu = torch.nn.ReLU()
        
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_bn = torch.nn.BatchNorm2d(num_features=256, affine=True)
        self.conv3_relu = torch.nn.ReLU()

        # SPADE Blocks
        self.spade_layer_1 = SPADE(128, num_channel_modulation, hidden_size)
        self.adain_1 = AdaIN(128,128)
        self.spade_layer_2 = SPADE(128, num_channel_modulation, hidden_size)
        self.adain_2 = AdaIN(128,128)
        self.pixel_shuffle_1 = torch.nn.PixelShuffle(2)
        self.spade_layer_4 = SPADE(32, num_channel_modulation, hidden_size)

        # Final Convolutional Layer
        self.conv_4 = torch.nn.Conv2d(32, 2, kernel_size=7, stride=1, padding=3)
        self.conv_5= nn.Sequential(torch.nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3),
                                   torch.nn.ReLU(),
                                   torch.nn.Conv2d(32, 1, kernel_size=7, stride=1, padding=3),
                                   torch.nn.Sigmoid(),
                                   )#predict weight

    def forward(self, ref_N_frame_img, ref_N_frame_heatmap, T_driving_heatmap, T_audio_feats): #to output: (B*T,3,H,W)
                   #   (B, N, 3, H, W)    (B, N, 3, H, W)    (B, 3, H, W)  # (B, 128, 1)
        ref_N = ref_N_frame_img.size(1)
        T_audio_feats = T_audio_feats.permute(0,2,1)
        wrapped_128_sum, wrapped_64_sum, wrapped_32_sum, wrapped_ref_sum=0.,0.,0.,0.
        softmax_denominator=0.
        for ref_idx in range(ref_N): # each ref img provide information for each B*T frame
            ref_img= ref_N_frame_img[:, ref_idx]  #(B, 3, H, W)

            ref_sketch = ref_N_frame_heatmap[:, ref_idx] #(B, 3, H, W)

            #predict flow and weight
            flow_module_input = torch.cat((ref_img, ref_sketch), dim=1)  #(B*T, 3+3, H, W)
            # Convolutional Layers
            h_128 = self.conv1_relu(self.conv1_bn(self.conv1(flow_module_input)))   #(32,128,128)
            h_64 = self.conv2_relu(self.conv2_bn(self.conv2(h_128)))    #(128,64,64)
            h_32 = self.conv3_relu(self.conv3_bn(self.conv3(h_64)))    #(256,32,32)
            
            # SPADE Blocks
            downsample_64 = downsample(T_driving_heatmap, (64, 64))   # T_driving_heatmap:(B*T, 3, H, W)

            spade_layer = self.spade_layer_1(h_64, downsample_64)  #(128,64,64)
            spade_layer = self.adain_1(spade_layer, T_audio_feats)
            spade_layer = self.spade_layer_2(spade_layer, downsample_64)   #(128,64,64)
            spade_layer = self.adain_2(spade_layer, T_audio_feats)

            spade_layer = self.pixel_shuffle_1(spade_layer)   #(32,128,128)

            spade_layer = self.spade_layer_4(spade_layer, T_driving_heatmap)    #(32,128,128)

            # Final Convolutional Layer
            output_flow = self.conv_4(spade_layer)      #   (B*T,2,128,128)
            output_weight=self.conv_5(spade_layer)       #  (B*T,1,128,128)

            deformation=convert_flow_to_deformation(output_flow)
            wrapped_h_128 = warping(h_128, deformation)  #(32,128,128)
            wrapped_h_64 = warping(h_64, deformation)   #(128,64,64)
            wrapped_h_32 = warping(h_32, deformation)   #(256,32,32)
            
            wrapped_ref = warping(ref_img, deformation)  #(3,128,128)

            softmax_denominator+=output_weight
            wrapped_128_sum+=wrapped_h_128*output_weight
            wrapped_64_sum+=wrapped_h_64*downsample(output_weight, (64,64))
            wrapped_32_sum+=wrapped_h_32*downsample(output_weight, (32,32))
            wrapped_ref_sum+=wrapped_ref*output_weight
        #return weighted warped feataure and images
        softmax_denominator+=0.00001
        wrapped_128_sum=wrapped_128_sum/softmax_denominator
        wrapped_64_sum = wrapped_64_sum / downsample(softmax_denominator, (64,64))
        wrapped_32_sum = wrapped_32_sum / downsample(softmax_denominator, (32,32))
        wrapped_ref_sum = wrapped_ref_sum / softmax_denominator
        
        ref_temporal_deformed_feats = [wrapped_128_sum, wrapped_64_sum, wrapped_32_sum]
        
        return ref_temporal_deformed_feats, wrapped_ref_sum

class TranslationNetwork(torch.nn.Module):
    def __init__(self):
        super(TranslationNetwork, self).__init__()
        # Encoder
        self.conv1 = torch.nn.Conv2d(in_channels=3+3, out_channels=32, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1_bn = torch.nn.BatchNorm2d(num_features=32, affine=True)
        self.conv1_relu = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2_bn = torch.nn.BatchNorm2d(num_features=128, affine=True)
        self.conv2_relu = torch.nn.ReLU()
        
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3_bn = torch.nn.BatchNorm2d(num_features=512, affine=True)
        self.conv3_relu = torch.nn.ReLU()

        # Decoder
        self.spade_1 = SPADE(num_channel=512, num_channel_modulation=256)
        self.adain_1 = AdaIN(512,128)
        self.pixel_suffle_1 = nn.PixelShuffle(upscale_factor=2)

        self.spade_2 = SPADE(num_channel=128, num_channel_modulation=128)
        self.adain_2 = AdaIN(128,128)
        self.pixel_suffle_2 = nn.PixelShuffle(upscale_factor=2)

        self.spade_3 = SPADE(num_channel=32, num_channel_modulation=32)
        self.adain_3 = AdaIN(32,128)

        # Final layer
        self.leaky_relu = torch.nn.LeakyReLU()
        self.conv_last = torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=7, stride=1, padding=3, bias=False)
        self.Sigmoid=torch.nn.Sigmoid()
    def forward(self, translation_input, ref_temporal_deformed_feats, T_audio_feats):
        #              (B,3+3*5,H,W)   [(B, 32, 128, 128), (B, 128, 64, 64), (B, 256, 32, 32),], B, 128, 1
        # prepare audio feature
        audio_feature = T_audio_feats.permute(0,2,1) #(B*T,1,128)
        # Encode
        x = self.conv1_relu(self.conv1_bn(self.conv1(translation_input)))    #32,128,128
        x = self.conv2_relu(self.conv2_bn(self.conv2(x)))  #128,64,64
        x = self.conv3_relu(self.conv3_bn(self.conv3(x)))  #256,32,32


        # Decode
        x = self.spade_1(x, ref_temporal_deformed_feats[2]) # (C=512,32,32)
        x = self.adain_1(x, audio_feature)  # (C=512,32,32)
        x = self.pixel_suffle_1(x)   # (C=128,64,64)

        x = self.spade_2(x, ref_temporal_deformed_feats[1])   # (C=128,64,64)
        x = self.adain_2(x, audio_feature)  # (64,128,128)
        x = self.pixel_suffle_2(x)   # (C=32,128,128)
        
        x = self.spade_3(x, ref_temporal_deformed_feats[0])    # (32,128,128)
        x = self.adain_3(x, audio_feature)  # (32,128,128)

        # output layer
        x = self.leaky_relu(x)
        x = self.conv_last(x)
        x = self.Sigmoid(x)
        return x

class ChannelAlign(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelAlign, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)  
        return x

class Face_renderer(torch.nn.Module):
    def __init__(self):
        super(Face_renderer, self).__init__()
        # Spatio-temporal renderer
        # 0. ref feature encoder
        self.ref_imgs_encoder = ref_imgs_encoder()
        
        # 1. heatmap encoder
        self.heatmap_encoder_tar = Heatmap_encoder_tar()
        self.heatmap_encoder_ref = Heatmap_encoder_ref()
        
        # 2. audio encoder
        self.audio_encoder = audio_encoer_w2v()
        
        # 3. spatial Network  (spatial deformation)
        self.spatial_module = SpatialNetwork()
         
        # 4. temporal Network   (temporal dimension constraint)
        self.temporal_module = TemporalNetwork()
        
        # 5. source feature encoder
        self.source_img_encoder = nn.Sequential(
            SameBlock2d(3*1, 32, kernel_size=7, padding=3),
            DownBlock2d(32, 64, kernel_size=3, padding=1),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
        )
        # 6.fuse dual-path feats
        self.fuse_32 = nn.Sequential(
            SameBlock2d(512, 256, kernel_size=3, padding=1),
            SameBlock2d(256, 256, kernel_size=3, padding=1),
        )
        self.fuse_64 = nn.Sequential(
            SameBlock2d(256, 128, kernel_size=3, padding=1),
            SameBlock2d(128, 128, kernel_size=3, padding=1),
        )
        self.fuse_128 = nn.Sequential(
            SameBlock2d(64, 32, kernel_size=3, padding=1),
            SameBlock2d(32, 32, kernel_size=3, padding=1),
        )
        
        # 7. translation module
        self.translation = TranslationNetwork()
        
        # 8. return loss
        self.perceptual = PerceptualLoss(network='vgg19',
                                         layers=['relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1'],
                                         num_scales=2)
        self.mse = torch.nn.MSELoss()
        
        # 9. feat align and pool
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.align_layers = nn.ModuleList([ChannelAlign(c, 128) for c in [32, 128, 256]])  # 通道对齐
        
        self.apply(weight_init)
        self.mine = MINE()


    def forward(self, T_frame_img, T_driving_heatmap, ref_N_frame_img, ref_N_frame_heatmap, T_audio_w2v): #T=1
        #            (B,3,H,W)   (B,3,H,W)   (B,N,3,H,W)   (B,N,3,H,W)  (B,hv,wv)T=1
        #            (B,T,3,H,W) (B,T,3,H,W) (B,N,3,H,W)   (B,N,3,H,W)  (B,T,hv,wv)
        # gt_face = torch.cat([T_frame_img[:,:,i] for i in range(T_frame_img.size(2))], dim=1)  # (B, 3*T, 128, 128) 
        input_dim_size = len(T_frame_img.size())
        T = T_frame_img.size(1)
        B = T_frame_img.size(0)
        if input_dim_size > 4:
            
            T_frame_img = torch.cat([T_frame_img[:, i] for i in range(T_frame_img.size(1))], dim=0)  # B*T, 3, H, W
            T_driving_heatmap = torch.cat([T_driving_heatmap[:, i] for i in range(T_driving_heatmap.size(1))], dim=0)  # B*T, 3, H, W
            T_audio_w2v = torch.cat([T_audio_w2v[:, i] for i in range(T_audio_w2v.size(1))], dim=0)  # B*T, hv, wv
            ref_N_frame_img = ref_N_frame_img.repeat(T, 1, 1, 1, 1)  # B*T, N, 3, H, W
            ref_N_frame_heatmap = ref_N_frame_heatmap.repeat(T, 1, 1, 1, 1)  # B*T, N, 3, H, W
        
        gt_face = T_frame_img # (B, 3, 128, 128) 
        gt_mask_face = gt_face.clone()
        gt_mask_face[:, :, gt_mask_face.size(2) // 2:, :] = 0  # (B,3,H,W)
        # T_driving_heatmap = torch.cat([T_driving_heatmap[:,i] for i in range(T_driving_heatmap.size(1))], dim=1)  #(B, 3*5, H, W)
        ref_N_frame_heatmap_cat = torch.cat([ref_N_frame_heatmap[:,i] for i in range(ref_N_frame_heatmap.size(1))], dim=1)  #(B, 3*5, H, W)
        
        # (1) referemce images fetures for spatial
        ref_img_feats = self.ref_imgs_encoder(ref_N_frame_img)
        # (2) reference heatmaps features for spatial
        ref_heatmap_feats = self.heatmap_encoder_ref(ref_N_frame_heatmap_cat)
        # (3) driving heatmaps features for spatial
        T_driving_heatmap_feats = self.heatmap_encoder_tar(T_driving_heatmap)
        # (4) audio features for spatial and temporal
        T_audio_feats = self.audio_encoder(T_audio_w2v) # B, 128, 1
        
        # (5) reference images deformed feature  via two paths
        ref_spatial_deformed_feats = self.spatial_module(ref_img_feats, ref_heatmap_feats, T_driving_heatmap_feats, T_audio_feats)

        ref_temporal_deformed_feats, wrapped_ref_img = self.temporal_module(ref_N_frame_img, ref_N_frame_heatmap, T_driving_heatmap, T_audio_feats)    # (B, 128, 32, 32)    (B，2， 128， 128)  (B，2， 128， 128)
        # [(B, 32, 128, 128), (B, 128, 64, 64), (B, 256, 32, 32),]
        
        ref_spatial_deformed_feats_pool = [self.pool(self.align_layers[i](ref_spatial_deformed_feats[i])).squeeze(-1).squeeze(-1) for i in range(len(ref_spatial_deformed_feats))]
        ref_temporal_deformed_feats_pool = [self.pool(self.align_layers[i](ref_temporal_deformed_feats[i])).squeeze(-1).squeeze(-1) for i in range(len(ref_temporal_deformed_feats))]
        # [(B, 128), (B, 128), (B, 128)]
        mine_loss_total = 0.
        for (feat_x, feat_y) in zip(ref_spatial_deformed_feats_pool, ref_temporal_deformed_feats_pool):
            joint = self.mine(feat_x, feat_y)
            shuffled_y = feat_y[torch.randperm(feat_y.size(0))]  
            marginal = self.mine(feat_x, shuffled_y)
            mine_loss_total += self.mine.mine_loss(joint, marginal)
        mine_loss_total = mine_loss_total / len(ref_spatial_deformed_feats_pool)
        # (6) mix dual paths features 
        # print("here")
        ref_deformed_feats = [
            self.fuse_128(torch.cat([ref_spatial_deformed_feats[0], ref_temporal_deformed_feats[0]], dim=1)),
            self.fuse_64(torch.cat([ref_spatial_deformed_feats[1], ref_temporal_deformed_feats[1]], dim=1)),
            self.fuse_32(torch.cat([ref_spatial_deformed_feats[2], ref_temporal_deformed_feats[2]], dim=1)),
        ]
        
        # (7) face generate
        translation_input=torch.cat([gt_mask_face, T_driving_heatmap], dim=1) #  (B,3+3*5,H,W)
        generated_face = self.translation(translation_input, ref_deformed_feats, T_audio_feats) #translation_input (B*T, 3, H, W)
        
        
                
        perceptual_gen_loss = self.perceptual(generated_face, gt_face, use_style_loss=True,
                                              weight_style_to_perceptual=250).mean()

        if input_dim_size > 4:
            x = torch.split(generated_face, B, dim=0) # [(B, C, H, W)]
            generated_face = torch.stack(x, dim=2) # (B, C, T, H, W)
        
        
        return generated_face, torch.unsqueeze(perceptual_gen_loss, 0), ref_spatial_deformed_feats_pool, ref_temporal_deformed_feats_pool, torch.unsqueeze(mine_loss_total, 0)  # (B,3,H,W) and losses

#the following is the code for Perceptual(VGG) loss

def apply_imagenet_normalization(input):
    r"""Normalize using ImageNet mean and std.

    Args:
        input (4D tensor NxCxHxW): The input images, assuming to be [-1, 1].

    Returns:
        Normalized inputs using the ImageNet normalization.
    """
    # normalize the input back to [0, 1]
    normalized_input = (input + 1) / 2
    # normalize the input using the ImageNet mean and std
    mean = normalized_input.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = normalized_input.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    output = (normalized_input - mean) / std
    return output

class _PerceptualNetwork(nn.Module):
    r"""The network that extracts features to compute the perceptual loss.

    Args:
        network (nn.Sequential) : The network that extracts features.
        layer_name_mapping (dict) : The dictionary that
            maps a layer's index to its name.
        layers (list of str): The list of layer names that we are using.
    """

    def __init__(self, network, layer_name_mapping, layers):
        super().__init__()
        assert isinstance(network, nn.Sequential), \
            'The network needs to be of type "nn.Sequential".'
        self.network = network
        self.layer_name_mapping = layer_name_mapping
        self.layers = layers
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        r"""Extract perceptual features."""
        output = {}
        for i, layer in enumerate(self.network):
            x = layer(x)
            layer_name = self.layer_name_mapping.get(i, None)
            if layer_name in self.layers:
                # If the current layer is used by the perceptual loss.
                output[layer_name] = x
        return output

def _vgg19(layers):
    r"""Get vgg19 layers"""
    network = torchvision.models.vgg19(weights=True).features
    layer_name_mapping = {1: 'relu_1_1',
                          3: 'relu_1_2',
                          6: 'relu_2_1',
                          8: 'relu_2_2',
                          11: 'relu_3_1',
                          13: 'relu_3_2',
                          15: 'relu_3_3',
                          17: 'relu_3_4',
                          20: 'relu_4_1',
                          22: 'relu_4_2',
                          24: 'relu_4_3',
                          26: 'relu_4_4',
                          29: 'relu_5_1'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)

class PerceptualLoss(nn.Module):
    r"""Perceptual loss initialization.

    Args:
        network (str) : The name of the loss network: 'vgg16' | 'vgg19'.
        layers (str or list of str) : The layers used to compute the loss.
        weights (float or list of float : The loss weights of each layer.
        criterion (str): The type of distance function: 'l1' | 'l2'.
        resize (bool) : If ``True``, resize the input images to 224x224.
        resize_mode (str): Algorithm used for resizing.
        instance_normalized (bool): If ``True``, applies instance normalization
            to the feature maps before computing the distance.
        num_scales (int): The loss will be evaluated at original size and
            this many times downsampled sizes.
    """

    def __init__(self, network='vgg19', layers='relu_4_1', weights=None,
                 criterion='l1', resize=False, resize_mode='bilinear',
                 instance_normalized=False, num_scales=1,):
        super().__init__()
        if isinstance(layers, str):
            layers = [layers]
        if weights is None:
            weights = [1.] * len(layers)
        elif isinstance(layers, float) or isinstance(layers, int):
            weights = [weights]

        assert len(layers) == len(weights), \
            'The number of layers (%s) must be equal to ' \
            'the number of weights (%s).' % (len(layers), len(weights))
        if network == 'vgg19':
            self.model = _vgg19(layers)
        else:
            raise ValueError('Network %s is not recognized' % network)

        self.num_scales = num_scales
        self.layers = layers
        self.weights = weights
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError('Criterion %s is not recognized' % criterion)
        self.resize = resize
        self.resize_mode = resize_mode
        self.instance_normalized = instance_normalized


        print('Perceptual loss:')
        print('\tMode: {}'.format(network))

    def forward(self, inp, target, mask=None,use_style_loss=False,weight_style_to_perceptual=0.):
        r"""Perceptual loss forward.

        Args:
           inp (4D tensor) : Input tensor.
           target (4D tensor) : Ground truth tensor, same shape as the input.

        Returns:
           (scalar tensor) : The perceptual loss.
        """
        # Perceptual loss should operate in eval mode by default.
        self.model.eval()
        inp, target = \
            apply_imagenet_normalization(inp), \
            apply_imagenet_normalization(target)
        if self.resize:
            inp = F.interpolate(
                inp, mode=self.resize_mode, size=(256, 256),
                align_corners=False)
            target = F.interpolate(
                target, mode=self.resize_mode, size=(256, 256),
                align_corners=False)

        # Evaluate perceptual loss at each scale.
        loss = 0
        style_loss=0
        for scale in range(self.num_scales):
            input_features, target_features = \
                self.model(inp), self.model(target)
            for layer, weight in zip(self.layers, self.weights):
                # Example per-layer VGG19 loss values after applying
                # [0.03125, 0.0625, 0.125, 0.25, 1.0] weighting.
                # relu_1_1, 0.014698
                # relu_2_1, 0.085817
                # relu_3_1, 0.349977
                # relu_4_1, 0.544188
                # relu_5_1, 0.906261
                input_feature = input_features[layer]
                target_feature = target_features[layer].detach()
                if self.instance_normalized:
                    input_feature = F.instance_norm(input_feature)
                    target_feature = F.instance_norm(target_feature)

                if mask is not None:
                    mask_ = F.interpolate(mask, input_feature.shape[2:],
                                          mode='bilinear',
                                          align_corners=False)
                    input_feature = input_feature * mask_
                    target_feature = target_feature * mask_
                    # print('mask',mask_.shape)


                loss += weight * self.criterion(input_feature,
                                                target_feature)
                if use_style_loss and scale==0:
                    style_loss += self.criterion(self.compute_gram(input_feature),
                                                 self.compute_gram(target_feature))

            # Downsample the input and target.
            if scale != self.num_scales - 1:
                inp = F.interpolate(
                    inp, mode=self.resize_mode, scale_factor=0.5,
                    align_corners=False, recompute_scale_factor=True)
                target = F.interpolate(
                    target, mode=self.resize_mode, scale_factor=0.5,
                    align_corners=False, recompute_scale_factor=True)

        if use_style_loss:
            return loss + style_loss*weight_style_to_perceptual
        else:
            return loss


    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

if __name__ == '__main__':
    device = torch.device('cuda:0')
    ref_N = 5
    B = 5
    T = 5
    hv, hw = 768, 10
    
    ref_img_feats = [torch.randn(B, 32, 128, 128).to(device), torch.randn(B, 128, 64, 64).to(device), torch.randn(B, 256, 32, 32).to(device)]
    ref_heatmap_feats = torch.randn(B, 32, 32, 32).to(device)
    T_driving_heatmap_feats = torch.randn(B, 32, 32, 32).to(device)
    T_audio_feats = torch.randn(B, 128, 1).to(device)
    source_feats = torch.randn(B, 128, 32, 32).to(device)
    
    model = Face_renderer().to(device)
        
    T_frame_img = torch.randn(B, T, 3, 128, 128).to(device)
    
    T_driving_heatmap = torch.randn(B, T, 3, 128, 128).to(device)
    
    ref_N_frame_img = torch.randn(B, ref_N, 3, 128, 128).to(device)
    
    ref_N_frame_heatmap = torch.randn(B, ref_N, 3, 128, 128).to(device)
    
    T_w2v = torch.randn(B, T, hv, hw).to(device)
    

    
    face, per_gen_loss, x, y = model(T_frame_img, T_driving_heatmap, ref_N_frame_img, ref_N_frame_heatmap, T_w2v)
    print(face.shape)
    print(x[0].shape)
    print(x[1].shape)
    print(x[2].shape)
    