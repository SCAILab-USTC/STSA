import sys
sys.path.append("..")
from models import audio
import numpy as np
import torch
from tqdm import tqdm
import glob
import os 
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class MyWav2Vec(nn.Module):
    def __init__(self):
        super(MyWav2Vec, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2Vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    def forward(self, x):
        return self.wav2Vec(x).last_hidden_state

    def process(self, x):
        return self.processor(x, sampling_rate=16000, return_tensors="pt").input_values


input_wav_root = './processed_lrs2/lrs2_audio'

model = MyWav2Vec()
model = model.to('cuda:0')

print('looking up audios.... ')
audio_list = glob.glob(input_wav_root + '/*/*/*.wav')  
print('total audios :', len(audio_list))

wav_input = audio.load_wav(audio_list[0], sr = 16000)

input = model.process(wav_input)

input = input.to('cuda:0')

output = model(input)

print(input.shape)
print(output[0].shape)

for audio_path in tqdm(audio_list, desc='Processing', unit='item'):

    output_path = audio_path.replace('audio.wav', 'wav2vec2_768.npy')
    
    wav_input = audio.load_wav(audio_path, sr = 16000) 
    
    input = model.process(wav_input)

    input = input.to('cuda:0')
    
    output = model(input)
    output_featute = output.cpu().detach().numpy()
    np.save(output_path, output_featute)
    
print('done')
    
