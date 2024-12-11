# STSA: Spatial-Temporal Semantic Alignment for Facial Visual Dubbing

Pytorch implementation for our ICME2025 submission "STSA: Spatial-Temporal Semantic Alignment for Facial Visual Dubbing".

<!-- <img src='./pbf.png' width=900> -->

## Demo:
### Long Video Generation Compared with SOTA Methods
[Ours](https://github.com/user-attachments/assets/b7fb5bfd-7a15-4f73-a7e2-83916165c54c)

[DiffTalk](https://github.com/user-attachments/assets/a4c9dc00-2310-4f0f-9c03-8a033711d868)

[DINet](https://github.com/user-attachments/assets/f1900c8c-a657-461a-84f2-062d2dc1930e)

[IP-LAP](https://github.com/user-attachments/assets/a383e735-b204-436e-b4bd-75f85742837e)

[MuseTalk](https://github.com/user-attachments/assets/d6d5ae67-95e3-4708-9259-966b54365344)

[PC-AVS](https://github.com/user-attachments/assets/3b601d3f-04a0-4779-a2bd-68d249180ad2)

[TalkLip](https://github.com/user-attachments/assets/9db72cc8-e9d4-4ced-8680-bafe60ccbed3)

[Wav2Lip](https://github.com/user-attachments/assets/7d397030-5773-4d9e-a5d6-0e22deba5e4c)

---
## Inference:
### Requirements
- Python 3.8.7
- torch 1.12.1
- torchvision 0.12.1
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
Run the following command:
```
python inference.py --video_path "demo_templates/video/speakerine.mp4" --audio_path "demo_templates/audio/education.wav"
```
You can specify the `--video_path` and `--audio_path` option to inference other videos.
