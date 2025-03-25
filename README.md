# STSA: Spatial-Temporal Semantic Alignment for Facial Visual Dubbing

Pytorch implementation for our ICME2025 submission "STSA: Spatial-Temporal Semantic Alignment for Facial Visual Dubbing".
![framework](inserts/model.png)
## Todo:
- [x] inference code
- [ ] paper & supplementary material
- [ ] youtube demo 
- [ ] training code
- [ ] fine-tuning code 

## Demo:
### Multilingual Generation

[Chinese](https://github.com/user-attachments/assets/4e52356a-ed42-40ef-9ea3-5ffca7bbd3d1)

[Korean](https://github.com/user-attachments/assets/e71cce15-0a18-45e5-b253-52c5e9fc4064)

[Japanese](https://github.com/user-attachments/assets/3880dc0d-aa2c-4ba7-8793-a29ab33dd129)

[Spanish](https://github.com/user-attachments/assets/3fc89023-1b10-4902-a950-130c359ac81e)

### Long Video Generation Compared with SOTA Methods
We compare our method with DiffTalk(CVPR23'), DINet(AAAI23'), IP-LAP(CVPR23'), MuseTalk(Arxiv2024), PC-AVS(CVPR21'), TalkLip(CVPR23'), Wav2Lip(MM'20)

[Ours](https://github.com/user-attachments/assets/b6e9b594-4e7a-41f3-ad8e-1998caa12b3b)

[DiffTalk](https://github.com/user-attachments/assets/297fcb43-00f4-4d81-a022-70f07867ce03)

[DINet](https://github.com/user-attachments/assets/10b7ea15-0d01-4bcd-a036-fbe58b8bda33)

[IP-LAP](https://github.com/user-attachments/assets/55466ea9-2d30-42cc-8ed8-ffe8878f2eb7)

[MuseTalk](https://github.com/user-attachments/assets/4233c7cb-8eb4-4977-8239-3c39055fc27f)

[PC-AVS](https://github.com/user-attachments/assets/ca5e0b92-249a-4fe1-bf53-85d21e09e059)

[TalkLip](https://github.com/user-attachments/assets/e5e3d6ac-75dd-443f-af79-c60b94c7062c)

[Wav2Lip](https://github.com/user-attachments/assets/0fe501d0-1c83-48c6-8998-6958377e9d4e)

---
## Inference:
### Requirements
- Python 3.8.7
- torch 1.12.1
- torchvision 0.13.1
- librosa 0.9.2
- ffmpeg

### Prepare Environment
First create conda environment:
```
conda create -n stsa python=3.8
conda activate stsa
```
[Pytorch](https://pytorch.org/)  1.12.1 is used, other requirements are listed in "requirements.txt". Please run:
```
pip install -r requirements.txt
```
### Quick Start
Download the [pretrained weights](https://www.jianguoyun.com/p/DW9UAjMQqcOQDRiotuMFIAA), and put the weights under ./checkpoints 
After this, run the following command:
```
python inference.py --video_path "demo_templates/video/speakerine.mp4" --audio_path "demo_templates/audio/education.wav"
```
You can specify the `--video_path` and `--audio_path` option to inference other videos.
