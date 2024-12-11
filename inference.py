import numpy as np
import cv2, os, argparse
import subprocess
from tqdm import tqdm
import face_alignment
import mediapipe as mp
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import dataclasses
from draw_landmark import draw_landmarks
from typing import Tuple
from models import Heatmap_generator 
from models import Face_renderer
from models import audio
from boundary_heatmap_draw import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ----------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, default="demo_templates/video/speakerine.mp4")
parser.add_argument('--audio_path', type=str, default="demo_templates/audio/education.wav")

parser.add_argument('--output_dir', type=str, default='./demo_results')
parser.add_argument('--heatmap_checkpoint_path', type=str, default="checkpoints/heatmap_weight.pth")
parser.add_argument('--renderer_checkpoint_path', type=str, default="checkpoints/face_weight.pth")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = parser.parse_args()
# ----------------------------------------------------------------------------------------------------
ref_img_N = 5
Nl = 5
T = 5
w2v_step_size = 10
img_size = 128
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')
lip_index = [0, 17]  # the index of the midpoints of the upper lip and lower lip
heatmap_checkpoint_path = args.heatmap_checkpoint_path
renderer_checkpoint_path =args.renderer_checkpoint_path
# ----------------------------------------------------------------------------------------------------
WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

@dataclasses.dataclass
class DrawingSpec:
    # Color for drawing the annotation. Default to the white color.
    color: Tuple[int, int, int] = WHITE_COLOR
    # Thickness for drawing the annotation. Default to 2 pixels.
    thickness: int = 2
    # Circle radius. Default to 2 pixels.
    circle_radius: int = 2

# the following is the index sequence for fical landmarks detected by mediapipe
ori_sequence_idx = [162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288,
                    361, 323, 454, 356, 389,  #
                    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,  #
                    336, 296, 334, 293, 300, 276, 283, 282, 295, 285,  #
                    168, 6, 197, 195, 5,  #
                    48, 115, 220, 45, 4, 275, 440, 344, 278,  #
                    33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,  #
                    362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382,  #
                    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,  #
                    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# the following is the connections of landmarks for drawing sketch image
FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),   # 嘴唇
                           (17, 314), (314, 405), (405, 321), (321, 375),(375, 291), 
                           (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                           
                           (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                           (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
                           (78, 191), (191, 80), (80, 81), (81, 82),
                           (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)])
FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),   # 左眼
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (263, 466), (466, 388), (388, 387), (387, 386),
                               (386, 385), (385, 384), (384, 398), (398, 362)])
FACEMESH_LEFT_EYEBROW = frozenset([(276, 283), (283, 282), (282, 295),           # 左眉毛
                                   (295, 285), (300, 293), (293, 334),
                                   (334, 296), (296, 336)])
FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),       # 右眼
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)])
FACEMESH_RIGHT_EYEBROW = frozenset([(46, 53), (53, 52), (52, 65), (65, 55),        # 右眉
                                    (70, 63), (63, 105), (105, 66), (66, 107)])
FACEMESH_FACE_OVAL = frozenset([(389, 356), (356, 454),                           # 脸轮廓
                                (454, 323), (323, 361), (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 93), (93, 234),
                                (234, 127), (127, 162)])
FACEMESH_NOSE = frozenset([(168, 6), (6, 197), (197, 195), (195, 5), (5, 4),   # 鼻子
                           (4, 45), (45, 220), (220, 115), (115, 48),
                           (4, 275), (275, 440), (440, 344), (344, 278), ])
FACEMESH_FACE_OVAL_UPPER = frozenset([(389, 356), (356, 454),
                                (454, 323),(323,361),
                                
                                
                                
                                                        (132,93),  (93, 234),
                                (234, 127), (127, 162)])

FACEMESH_FACE_OVAL_LOWER = frozenset([
                                                        (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132)    
                                                       ])

FACEMESH_CONNECTION = frozenset().union(*[
    FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL, FACEMESH_NOSE
])

FACEMESH_POSE = frozenset().union(*[
    FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW, FACEMESH_NOSE, FACEMESH_FACE_OVAL_UPPER
])

full_face_landmark_sequence = [*list(range(0, 5)), *list(range(20, 25)), *list(range(25, 91)),  #upper-half face
                               *list(range(5, 20)),  # jaw
                               *list(range(91, 131))]  # mouth

full_face_landmark_sequence_original = [*list(range(0, 4)), *list(range(21, 25)), *list(range(25, 91)),  #upper-half face
                               *list(range(4, 21)),  # jaw
                               *list(range(91, 131))]  # mouth

pose_face_landmark_sequence = [*list(range(0, 5)), *list(range(20, 25)), *list(range(25, 91))]  #upper-half face  

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
# ----------------------------------------------------------------------------------------------------

def summarize_landmark(edge_set):  # summarize all ficial landmarks used to construct edge
    landmarks = set()
    # print(edge_set)
    for a, b in edge_set:
        landmarks.add(a)
        landmarks.add(b)
    return landmarks

all_landmarks_idx = summarize_landmark(FACEMESH_CONNECTION)

pose_landmark_idx = \
    summarize_landmark(FACEMESH_NOSE.union(*[FACEMESH_RIGHT_EYEBROW, FACEMESH_RIGHT_EYE,
                                             FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_FACE_OVAL_UPPER]))
    
pose_landmark_idx_74 = \
    summarize_landmark(FACEMESH_NOSE.union(*[FACEMESH_RIGHT_EYEBROW, FACEMESH_RIGHT_EYE,
                                             FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW])).union(
        [162, 127, 234, 93, 389, 356, 454, 323])
# pose landmarks are landmarks of the upper-half face(eyes,nose,cheek) that represents the pose information

content_landmark_idx = all_landmarks_idx - pose_landmark_idx
# content_landmark include landmarks of lip and jaw which are inferred from audio

def swap_masked_region(target_img, src_img, mask): #function used in post-process
    """From src_img crop masked region to replace corresponding masked region
       in target_img
    """  # swap_masked_region(src_frame, generated_frame, mask=mask_img)
    mask_img = cv2.GaussianBlur(mask, (21, 21), 11)
    mask1 = mask_img / 255
    mask1 = np.tile(np.expand_dims(mask1, axis=2), (1, 1, 3))
    img = src_img * mask1 + target_img * (1 - mask1)
    return img.astype(np.uint8)

def merge_face_contour_only(src_frame, generated_frame, face_region_coord, fa): #function used in post-process
    """Merge the face from generated_frame into src_frame
    """
    input_img = src_frame
    y1, y2, x1, x2 = 0, 0, 0, 0
    if face_region_coord is not None:
        y1, y2, x1, x2 = face_region_coord
        input_img = src_frame[y1:y2, x1:x2]
    ### 1) Detect the facial landmarks
    preds = fa.get_landmarks(input_img)[0]  # 68x2
    if face_region_coord is not None:
        preds += np.array([x1, y1])
    lm_pts = preds.astype(int)
    contour_idx = list(range(0, 17)) + list(range(17, 27))[::-1]
    contour_pts = lm_pts[contour_idx]
    ### 2) Make the landmark region mask image
    mask_img = np.zeros((src_frame.shape[0], src_frame.shape[1], 1), np.uint8)
    cv2.fillConvexPoly(mask_img, contour_pts, 255)
    ### 3) Do swap
    img = swap_masked_region(src_frame, generated_frame, mask=mask_img)
    return img

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        output_path = os.path.join(output_folder, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(output_path, frame)
    cap.release()

# smooth landmarks
def get_smoothened_landmarks(all_landmarks, windows_T=1):
    for i in range(len(all_landmarks)):  # frame i
        if i + windows_T > len(all_landmarks):
            window = all_landmarks[len(all_landmarks) - windows_T:]
        else:
            window = all_landmarks[i: i + windows_T]
        #####
        for j in range(len(all_landmarks[i])):  # landmark j
            all_landmarks[i][j][1] = np.mean([frame_landmarks[j][1] for frame_landmarks in window])  # x
            all_landmarks[i][j][2] = np.mean([frame_landmarks[j][2] for frame_landmarks in window])  # y
    return all_landmarks
# -------------wav2vec model-------------
class MyWav2Vec(torch.nn.Module):
    def __init__(self):
        super(MyWav2Vec, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2Vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    def forward(self, x):
        return self.wav2Vec(x).last_hidden_state

    def process(self, x):
        return self.processor(x, sampling_rate=16000, return_tensors="pt").input_values
# ---------------------------------------
# Load wav2vec extractor
w2v_model = MyWav2Vec()
w2v_model = w2v_model.cuda()
# ----------------------------------------------------------------------------------------------------
# Load Model
heatmap_generator_model = Heatmap_generator(T=T, d_model=512, nlayers=8, nhead=4, dim_feedforward=1024, dropout=0.1)
renderer = Face_renderer()
print(" heatmap_generator_model loaded from : ", heatmap_checkpoint_path)
print(" renderer loaded from : ", renderer_checkpoint_path)
# --------------------------------
print('loading module....from :', heatmap_checkpoint_path)
model_dict = heatmap_generator_model.state_dict()
checkpoint = torch.load(heatmap_checkpoint_path)
s = checkpoint["state_dict"]
new_s = {}
for k, v in s.items():
    new_s[k.replace('module.', '')] = v
state_dict_needed = {k: v for k, v in new_s.items() if k in model_dict.keys()}  # we need in model
model_dict.update(state_dict_needed)
heatmap_generator_model.load_state_dict(model_dict)
heatmap_generator_model = heatmap_generator_model.cuda()
print('Load heatmap generator, Done.')
# --------------------------------
print('loading module....from :', renderer_checkpoint_path)
checkpoint = torch.load(renderer_checkpoint_path)
s = checkpoint["state_dict"]
new_s = {}
for k, v in s.items():
    # new_s[k.replace('module.', '',1)] = v  #
    new_s[k] = v  #
renderer.load_state_dict(new_s)
renderer = renderer.cuda()
print('Load face generator, Done.')
# --------------------------------

# --------------------------------------------------------------------------------
# -------------------------------Single--------------------------------------------
# -----------------------------Inference------------------------------------------
# --------------------------------------------------------------------------------
input_video_path = args.video_path
input_audio_path = args.audio_path
output_dir = os.path.join(args.output_dir, input_video_path.split('/')[-2] + '_' + input_video_path.split('/')[-1][:-4]\
    + '_' + input_audio_path.split('/')[-1][:-4])
temp_dir = 'temp/{}'.format(output_dir.split('/')[-1])

result_imgs_file = os.path.join(output_dir, 'result_imgs')
result_full_imgs_file = os.path.join(output_dir, 'result_full_imgs')


os.makedirs(output_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(result_imgs_file, exist_ok=True)
os.makedirs(result_full_imgs_file, exist_ok=True)


outfile_path = os.path.join(output_dir, 'result.mp4')
result_full_path = os.path.join(output_dir, 'result_full.mp4')
# --------------------------------------------------------------------------------

##(1) Reading input video frames  ###
print('Reading video frames ... from', input_video_path)
if not os.path.isfile(input_video_path):
    raise ValueError('the input video file does not exist')
elif input_video_path.split('.')[1] in ['jpg', 'png', 'jpeg']: #if input a single image for testing
    ori_background_frames = [cv2.imread(input_video_path)]
else:
    video_stream = cv2.VideoCapture(input_video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    if fps != 25:
        print(" input video fps:", fps,',converting to 25fps...')
        command = 'ffmpeg -y -i ' + input_video_path + ' -r 25 ' + '{}/temp_25fps.avi'.format(temp_dir)
        subprocess.call(command, shell=True)
        input_video_path = '{}/temp_25fps.avi'.format(temp_dir)
        video_stream.release()
        video_stream = cv2.VideoCapture(input_video_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
    assert fps == 25

    ori_background_frames = [] # input videos frames (includes background as well as face)
    frame_idx = 0
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        ori_background_frames.append(frame)
        frame_idx = frame_idx + 1
input_vid_len = len(ori_background_frames)
print(f"original videos has {input_vid_len} frames")
##(2) Extracting audio wav2vec####
if not input_audio_path.endswith('.wav'):
    command = 'ffmpeg -y -i {} -strict -2 {}'.format(input_audio_path, '{}/temp.wav'.format(temp_dir))
    subprocess.call(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    input_audio_path = '{}/temp.wav'.format(temp_dir)
# wav = audio.load_wav(input_audio_path, 16000)
wav_input = audio.load_wav(input_audio_path, sr = 16000)
input = w2v_model.process(wav_input)
input = input.cuda()
output = w2v_model(input)
w2v = output[0].cpu().detach().numpy()

# mel = audio.melspectrogram(wav)  # (H,W)   extract mel-spectrum
##read audio mel into list###
w2v_chunks = []  # each mel chunk correspond to 5 video frames, used to generate one video frame
w2v_idx_multiplier = 50. / fps
w2v_chunk_idx = 0
print(w2v.shape)
while 1:
    start_idx = int((w2v_chunk_idx-2) * w2v_idx_multiplier)
    end_idx = start_idx + w2v_step_size
    actual_start_idx = max(0, start_idx)
    actual_end_idx = min(end_idx, len(w2v[:, 0]))
    if (len(w2v[:, 0]) - actual_start_idx) < (2 * w2v_idx_multiplier):
        break
    # calculate the padding length
    padding_start = max(0, -int(start_idx))
    padding_end = max(0, int(end_idx) - len(w2v[:, 0]))
    # create 0-padding array
    padding_array_start = np.zeros((padding_start, w2v.shape[1]))
    padding_array_end = np.zeros((padding_end, w2v.shape[1]))
    w2v_window = np.concatenate((padding_array_start, w2v[actual_start_idx:actual_end_idx, :], padding_array_end), axis=0)
    w2v_chunks.append(w2v_window)  # mel for generate one video frame
    w2v_chunk_idx += 1
print("Length of w2v chunks:", len(w2v_chunks)) # 

##(3) detect facial landmarks using mediapipe tool
boxes = []  #bounding boxes of human face
lip_dists = [] #lip dists
#we define the lip dist(openness): distance between the  midpoints of the upper lip and lower lip
face_crop_results = []
all_pose_landmarks, all_content_landmarks = [], []  #content landmarks include lip and jaw landmarks
all_pose_landmarks_74 = []
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                        min_detection_confidence=0.5) as face_mesh:
    # (1) get bounding boxes and lip dist
    for frame_idx, full_frame in enumerate(ori_background_frames):
        h, w = full_frame.shape[0], full_frame.shape[1]
        results = face_mesh.process(cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            raise NotImplementedError  # not detect face
        face_landmarks = results.multi_face_landmarks[0]

        ## calculate the lip dist
        dx = face_landmarks.landmark[lip_index[0]].x - face_landmarks.landmark[lip_index[1]].x
        dy = face_landmarks.landmark[lip_index[0]].y - face_landmarks.landmark[lip_index[1]].y
        dist = np.linalg.norm((dx, dy))
        lip_dists.append((frame_idx, dist))

        # (1)get the marginal landmarks to crop face
        x_min,x_max,y_min,y_max = 999,-999,999,-999
        for idx, landmark in enumerate(face_landmarks.landmark):
            if idx in all_landmarks_idx:
                if landmark.x < x_min:
                    x_min = landmark.x
                if landmark.x > x_max:
                    x_max = landmark.x
                if landmark.y < y_min:
                    y_min = landmark.y
                if landmark.y > y_max:
                    y_max = landmark.y
        ##########plus some pixel to the marginal region##########
        #note:the landmarks coordinates returned by mediapipe range 0~1
        plus_pixel = 5
        x_min = max(x_min - plus_pixel / w, 0)
        x_max = min(x_max + plus_pixel / w, 1)

        y_min = max(y_min - plus_pixel / h, 0)
        y_max = min(y_max + plus_pixel / h, 1)
        y1, y2, x1, x2 = int(y_min * h), int(y_max * h), int(x_min * w), int(x_max * w)
        boxes.append([y1, y2, x1, x2])
    boxes = np.array(boxes)

    # (2)croppd face
    face_crop_results = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)] \
                        for image, (y1, y2, x1, x2) in zip(ori_background_frames, boxes)]

    # (3)detect facial landmarks
    for frame_idx, full_frame in enumerate(ori_background_frames):
        h, w = full_frame.shape[0], full_frame.shape[1]
        results = face_mesh.process(cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            raise ValueError("not detect face in some frame!")  # not detect
        face_landmarks = results.multi_face_landmarks[0]


        x_min=999
        x_max=-999
        y_min=999
        y_max=-999
        pose_landmarks, content_landmarks = [], []
        pose_landmarks_74 = []
        for idx, landmark in enumerate(face_landmarks.landmark):
            if idx in all_landmarks_idx: 
                if landmark.x<x_min:
                    x_min=landmark.x
                if landmark.x>x_max:
                    x_max=landmark.x

                if landmark.y<y_min:
                    y_min=landmark.y
                if landmark.y>y_max:
                    y_max=landmark.y
            
            if idx in pose_landmark_idx:
                pose_landmarks.append((idx, landmark.x, landmark.y))
            if idx in pose_landmark_idx_74:
                pose_landmarks_74.append((idx, landmark.x, landmark.y))
            if idx in content_landmark_idx:
                content_landmarks.append((idx, landmark.x, landmark.y))
        ##########plus 5 pixel to size##########
        x_min=max(x_min-5/w,0)
        x_max = min(x_max + 5 / w, 1)
        #
        y_min = max(y_min - 5 / h, 0)
        y_max = min(y_max + 5 / h, 1)
        
        # update landmarks
        pose_landmarks=[ \
            (idx,(x-x_min)/(x_max-x_min),(y-y_min)/(y_max-y_min)) for idx,x,y in pose_landmarks]
        pose_landmarks_74=[ \
            (idx,(x-x_min)/(x_max-x_min),(y-y_min)/(y_max-y_min)) for idx,x,y in pose_landmarks_74]
        content_landmarks=[\
            (idx, (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)) for idx, x, y in content_landmarks]

        all_pose_landmarks.append(pose_landmarks)
        all_content_landmarks.append(content_landmarks)
        all_pose_landmarks_74.append(pose_landmarks_74)


##randomly select N_l reference landmarks for landmark transformer##
dists_sorted = sorted(lip_dists, key=lambda x: x[1])
lip_dist_idx = np.asarray([idx for idx, dist in dists_sorted])  #the frame idxs sorted by lip openness

Nl_idxs = [lip_dist_idx[int(i)] for i in torch.linspace(0, input_vid_len - 1, steps=Nl)]
Nl_pose_landmarks, Nl_content_landmarks = [], []  #Nl_pose + Nl_content=Nl reference landmarks
for reference_idx in Nl_idxs:
    frame_pose_landmarks = all_pose_landmarks[reference_idx]
    frame_content_landmarks = all_content_landmarks[reference_idx]
    Nl_pose_landmarks.append(frame_pose_landmarks)
    Nl_content_landmarks.append(frame_content_landmarks)

Nl_pose = torch.zeros((Nl, 2, 76))  # 76 landmark
Nl_content = torch.zeros((Nl, 2, 55))  # 55 landmark
for idx in range(Nl):
    #arrange the landmark in a certain order, since the landmark index returned by mediapipe is is chaotic
    Nl_pose_landmarks[idx] = sorted(Nl_pose_landmarks[idx],
                                    key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
    Nl_content_landmarks[idx] = sorted(Nl_content_landmarks[idx],
                                    key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))

    Nl_pose[idx, 0, :] = torch.FloatTensor(
        [Nl_pose_landmarks[idx][i][1] for i in range(len(Nl_pose_landmarks[idx]))])  # x
    Nl_pose[idx, 1, :] = torch.FloatTensor(
        [Nl_pose_landmarks[idx][i][2] for i in range(len(Nl_pose_landmarks[idx]))])  # y
    
    Nl_content[idx, 0, :] = torch.FloatTensor(
        [Nl_content_landmarks[idx][i][1] for i in range(len(Nl_content_landmarks[idx]))])  # x
    Nl_content[idx, 1, :] = torch.FloatTensor(
        [Nl_content_landmarks[idx][i][2] for i in range(len(Nl_content_landmarks[idx]))])  # y
    
Nl_whole_landmarks = torch.cat([Nl_pose, Nl_content], dim=2).cpu().numpy()  # (Nl,2,131)
Nl_content = Nl_content.unsqueeze(0)  # (1,Nl, 2, 55)
Nl_pose = Nl_pose.unsqueeze(0)  # (1,Nl,2,76)
print("randomly select N_l reference landmarks for landmark transformer, done.")
Nl_whole_heatmap = []
# convert landmark to heatmap
for frame_idx in range(Nl_whole_landmarks.shape[0]):  # Nl
    full_landmarks = Nl_whole_landmarks[frame_idx] # (2,131)
    drawn_sketech = np.ones((128, 128, 3))
    mediapipe_format_landmarks = [LandmarkDict(ori_sequence_idx[full_face_landmark_sequence[idx]], full_landmarks[0, idx],
                                            full_landmarks[1, idx]) for idx in range(full_landmarks.shape[1])]
    drawn_sketech = draw_landmarks(drawn_sketech, mediapipe_format_landmarks, connections=FACEMESH_CONNECTION,
                                connection_drawing_spec=drawing_spec)  # （128， 128， 3）
    drawn_sketech = drawn_sketech[:,:,0] # (128, 128)
    heatmap = sketch2heatmap(drawn_sketech)
    Nl_whole_heatmap.append(heatmap)  # (Nl,128,128)
    
Nl_whole_heatmap = torch.FloatTensor(np.asarray(Nl_whole_heatmap)).cuda().unsqueeze(1).unsqueeze(0) # (1,Nl,1,128,128).
print("convert ref landmark to ref heatmap, done")

##select reference images and draw sketches for rendering according to lip openness##
ref_img_idx = [int(lip_dist_idx[int(i)]) for i in torch.linspace(0, input_vid_len - 1, steps=ref_img_N)]
ref_imgs = [face_crop_results[idx][0] for idx in ref_img_idx]
## (N,H,W,3)
ref_img_pose_landmarks, ref_img_content_landmarks = [], []
for idx in ref_img_idx:
    ref_img_pose_landmarks.append(all_pose_landmarks[idx])
    ref_img_content_landmarks.append(all_content_landmarks[idx])

ref_img_pose = torch.zeros((ref_img_N, 2, 76))  # 76 landmark
ref_img_content = torch.zeros((ref_img_N, 2, 55))  # 55 landmark

for idx in range(ref_img_N):
    ref_img_pose_landmarks[idx] = sorted(ref_img_pose_landmarks[idx],
                                        key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
    ref_img_content_landmarks[idx] = sorted(ref_img_content_landmarks[idx],
                                            key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
    ref_img_pose[idx, 0, :] = torch.FloatTensor(
        [ref_img_pose_landmarks[idx][i][1] for i in range(len(ref_img_pose_landmarks[idx]))])  # x
    ref_img_pose[idx, 1, :] = torch.FloatTensor(
        [ref_img_pose_landmarks[idx][i][2] for i in range(len(ref_img_pose_landmarks[idx]))])  # y

    ref_img_content[idx, 0, :] = torch.FloatTensor(
        [ref_img_content_landmarks[idx][i][1] for i in range(len(ref_img_content_landmarks[idx]))])  # x
    ref_img_content[idx, 1, :] = torch.FloatTensor(
        [ref_img_content_landmarks[idx][i][2] for i in range(len(ref_img_content_landmarks[idx]))])  # y

ref_img_full_face_landmarks = torch.cat([ref_img_pose, ref_img_content], dim=2).cpu().numpy()  # (N,2,131)
ref_img_heatmap = []
for frame_idx in range(ref_img_full_face_landmarks.shape[0]):  # N
    full_landmarks = ref_img_full_face_landmarks[frame_idx]  # (2,131)
    drawn_sketech = np.ones((128, 128, 3))
    mediapipe_format_landmarks = [LandmarkDict(ori_sequence_idx[full_face_landmark_sequence[idx]], full_landmarks[0, idx],
                                            full_landmarks[1, idx]) for idx in range(full_landmarks.shape[1])]
    drawn_sketech = draw_landmarks(drawn_sketech, mediapipe_format_landmarks, connections=FACEMESH_CONNECTION,
                                connection_drawing_spec=drawing_spec)
    drawn_sketech = drawn_sketech[:,:,0] # (128, 128)
    # heatmap = sketch2heatmap(drawn_sketech)
    # ref_heatmap_visual = heatmap_visualize(heatmap)
    # cv2.imwrite('ref_heatmap_visual.png', ref_heatmap_visual) 
    
    ref_img_heatmap.append(heatmap)
    
ref_img_heatmap = torch.FloatTensor(np.asarray(ref_img_heatmap)).cuda().unsqueeze(1).unsqueeze(0).repeat(1,1,3,1,1) # (1,N,3,128,128)

ref_imgs = [cv2.resize(face.copy(), (img_size, img_size)) for face in ref_imgs]
ref_imgs = torch.FloatTensor(np.asarray(ref_imgs) / 255.0).unsqueeze(0).permute(0, 1, 4, 2, 3).cuda()
# (1,N,3,H,W)

print("select reference images and draw sketches for rendering according to lip openness, done.")

##prepare output video stream#
frame_h, frame_w = ori_background_frames[0].shape[:-1]
out_stream_full = cv2.VideoWriter('{}/result_full.avi'.format(temp_dir), cv2.VideoWriter_fourcc(*'DIVX'), fps,
                            (frame_w, frame_h))  # +frame_h*3
out_stream = cv2.VideoWriter('{}/result.avi'.format(temp_dir), cv2.VideoWriter_fourcc(*'DIVX'), fps,
                            (128, 128))  # +frame_h*3

##generate final face image and output video##
input_w2v_chunks_len = len(w2v_chunks)
input_frame_sequence = torch.arange(input_vid_len).tolist()
#the input template video may be shorter than audio
#in this case we repeat the input template video as following
num_of_repeat=input_w2v_chunks_len//input_vid_len+1
input_frame_sequence = input_frame_sequence + list(reversed(input_frame_sequence))
input_frame_sequence=input_frame_sequence*((num_of_repeat+1)//2)


for batch_idx, batch_start_idx in tqdm(enumerate(range(0, input_w2v_chunks_len, 1)),
                                    total=len(range(0, input_w2v_chunks_len, 1))):
    # generate input_mel_chunks_len frames
    T_input_frame, T_ori_face_coordinates = [], []
    #note: input_frame include background as well as face
    T_w2v_batch, T_crop_face,T_pose_landmarks = [], [], []
    T_pose_landmarks_74 = []

    # (1) for each batch of T frame, generate corresponding landmarks using landmark generator
    for w2v_chunk_idx in range(batch_start_idx - 2, batch_start_idx - 2 + T):  # for each T frame
        # 1 input audio
        if (w2v_chunk_idx) < 0 or (w2v_chunk_idx > input_w2v_chunks_len-1):
            T_w2v_batch.append(np.zeros((w2v_step_size, w2v.shape[1])).T) # pad 0.
        else:
            T_w2v_batch.append(w2v_chunks[w2v_chunk_idx].T)

        # 2.input face
        input_frame_idx = int(input_frame_sequence[max(0, w2v_chunk_idx)])
        face, coords = face_crop_results[input_frame_idx]
        T_crop_face.append(face)
        T_ori_face_coordinates.append((face, coords))  ##input face
        # 3.pose landmarks
        T_pose_landmarks.append(all_pose_landmarks[input_frame_idx])
        T_pose_landmarks_74.append(all_pose_landmarks_74[input_frame_idx])
        
        # 4.background
        T_input_frame.append(ori_background_frames[input_frame_idx].copy())
    T_w2vs = torch.FloatTensor(np.asarray(T_w2v_batch)).unsqueeze(0)  # 1,T,h,w
    #prepare pose landmarks
    T_pose = torch.zeros((T, 2, 76))  # 76 landmarks
    T_pose_74 = torch.zeros((T, 2, 74))  # 74 landmarks
    for idx in range(T):
        T_pose_landmarks[idx] = sorted(T_pose_landmarks[idx],
                                    key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
        T_pose_landmarks_74[idx] = sorted(T_pose_landmarks_74[idx],
                                    key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
        T_pose[idx, 0, :] = torch.FloatTensor(
            [T_pose_landmarks[idx][i][1] for i in range(len(T_pose_landmarks[idx]))])  # x
        T_pose[idx, 1, :] = torch.FloatTensor(
            [T_pose_landmarks[idx][i][2] for i in range(len(T_pose_landmarks[idx]))])  # y
        
        T_pose_74[idx, 0, :] = torch.FloatTensor(
            [T_pose_landmarks_74[idx][i][1] for i in range(len(T_pose_landmarks_74[idx]))])  # x
        T_pose_74[idx, 1, :] = torch.FloatTensor(
            [T_pose_landmarks_74[idx][i][2] for i in range(len(T_pose_landmarks_74[idx]))])  # y
    
    # T_pose_landmarks_74 = T_pose_74 # (T, 2, 74)
    T_pose_74 = T_pose_74.unsqueeze(0)  # (1,T, 2,74)
    
    # prepare pose heatmaps
    T_pose_heatmap = []
    for idx in range(T):                                                                                                                                                                    
        drawn_sketech = np.ones((128, 128, 3))
        T_pose_landmarks = T_pose[idx]  # (2,76)
        
        mediapipe_format_landmarks = [LandmarkDict(ori_sequence_idx[pose_face_landmark_sequence[idx]], T_pose_landmarks[0, idx],
                                            T_pose_landmarks[1, idx]) for idx in range(T_pose_landmarks.shape[1])]
        drawn_sketech = draw_landmarks(drawn_sketech, mediapipe_format_landmarks, connections=FACEMESH_POSE,
                                connection_drawing_spec=drawing_spec)
        drawn_sketech = drawn_sketech[:,:,0] # (128, 128)
        heatmap = sketch2heatmap(drawn_sketech)
        T_pose_heatmap.append(heatmap)
        # T_lalala = heatmap_visualize(heatmap)
        # cv2.imwrite('lalala.png', T_lalala) 
    T_pose_heatmap = torch.FloatTensor(np.asarray(T_pose_heatmap)).cuda().unsqueeze(1).unsqueeze(0) # (1,T,1,128,128)
    
    # print("T pose heatmaps have been prepared!")  
    ##########################################################################################################################
    #landmark  generator inference
    Nl_whole_heatmap = Nl_whole_heatmap.cuda() # (1, Nl, 1, 128, 128)
    T_pose_heatmap = T_pose_heatmap.cuda() # (1, T, 1, 128, 128)
    T_w2vs = T_w2vs.cuda() # (1,T,hv,wv)
    T_pose = T_pose.cuda() 
    T_pose_74 = T_pose_74.cuda() # (1, T, 2, 74)

    T_target_heatmaps = []
    with torch.no_grad():  # require    (1,T,1,hv,wv) (1, T, 1, 128, 128) (1, T, 1, 128, 128)
        # print(T_mels.shape, T_pose_heatmap.shape, Nl_whole_heatmap.shape)   # torch.Size([1, 5, 1, 80, 16]) torch.Size([1, 5, 1, 128, 128]) torch.Size([1, 5, 1, 128, 128])
        # print(T_mels.max())
        heatmap_generator_model.eval()
        T_predict_whole_heatmap, predict_content_landmark = heatmap_generator_model(T_w2vs, T_pose_heatmap, Nl_whole_heatmap) # (1*T, 1, 128, 128), (1*T,2,57)
        T_predict_whole_heatmap = torch.stack(torch.split(T_predict_whole_heatmap,T,dim=0),dim=0) # (B, T, 1, 128, 128))
        T_pose_74 = torch.cat([T_pose_74[i] for i in range(T_pose_74.size(0))], dim=0)  # (1*T,2,74)
        T_predict_full_landmarks = torch.cat([T_pose_74, predict_content_landmark], dim=2).cpu().detach().numpy()  # (1*T,2,131)
        
        for frame_idx in range(T):
            full_landmarks = T_predict_full_landmarks[frame_idx]  # (2,131)
            drawn_sketech = np.ones((128, 128, 3))
            mediapipe_format_landmarks = [LandmarkDict(ori_sequence_idx[full_face_landmark_sequence_original[idx]], full_landmarks[0, idx],
                                        full_landmarks[1, idx]) for idx in range(full_landmarks.shape[1])]
            drawn_sketech = draw_landmarks(drawn_sketech, mediapipe_format_landmarks, connections=FACEMESH_CONNECTION,
                                        connection_drawing_spec=drawing_spec)
            drawn_sketech = drawn_sketech[:,:,0] # (128, 128)
            heatmap = sketch2heatmap(drawn_sketech) # (128, 128)
            T_target_heatmaps.append(torch.FloatTensor(heatmap))
            
        T_target_heatmaps = torch.stack(T_target_heatmaps, dim=0)  # (1*T, 128, 128)
        T_target_heatmaps = T_target_heatmaps.unsqueeze(1).cuda()  # (1*T, 1, 128, 128)
        T_target_heatmaps = torch.stack(torch.split(T_target_heatmaps,T,dim=0),dim=0) # (1, T, 1, 128, 128))
    
    
    #1.prepare target whole heatmap
    # T_predict_whole_heatmap = T_target_heatmaps.repeat(1,1,3,1,1) # (1,T,3,128, 128)
    T_predict_whole_heatmap = T_predict_whole_heatmap.repeat(1,1,3,1,1) # (1,T,3,128, 128)
    
    T_predict_whole_heatmap = torch.where(T_predict_whole_heatmap > 1, 1, T_predict_whole_heatmap)
    #T_predict_whole_heatmap[:, :, :, 64:, :] = torch.where(T_predict_whole_heatmap[:, :, :, 64:, :] < 0.1, 0, T_predict_whole_heatmap[:, :, :, 64:, :])
    
    # pre = heatmap_visualize(T_predict_whole_heatmap[0][2][0].cpu().detach().numpy())
    # cv2.imwrite('pre.png', pre) 
    T_w2vs = T_w2vs[:, 2].unsqueeze(0)

    # 2. lower-half masked face
    ori_face_img = torch.FloatTensor(cv2.resize(T_crop_face[2], (img_size, img_size)) / 255).permute(2, 0, 1).unsqueeze(
        0).unsqueeze(0).cuda()  #(1,1,3,H, W)
    T_predict_whole_heatmap = T_predict_whole_heatmap[:, 2].unsqueeze(0)
    # print(ori_face_img.shape)               # torch.Size([1, 1, 3, 128, 128])
    # print(T_predict_whole_heatmap.shape)   # torch.Size([1, 1, 3, 128, 128])
    # print(ref_imgs.shape)                 # torch.Size([1, 5, 3, 128, 128])
    # print(ref_img_heatmap.shape)         # torch.Size([1, 5, 3, 128, 128])
    # print(T_w2vs.shape)                 # torch.Size([1, 1, 768, 10])
    # 3. render the full face
    # require (1,1,3,H,W)   (1,T,3,H,W)  (1,N,3,H,W)   (1,N,3,H,W)  (1,1,1,h,w)
    # return  (1,3,H,W)
    with torch.no_grad():
        
        # print(T_predict_whole_heatmap.max())
        renderer.eval()
        generated_face, _, _, _, _ = renderer(ori_face_img, T_predict_whole_heatmap, ref_imgs, ref_img_heatmap,
                                                    T_w2vs)  # T=1 # (B, C, T, H, W)
    gen_face = (generated_face.squeeze(0).squeeze(1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # (H,W,3)
    # cv2.imwrite("gen_face.jpg", gen_face) # Here
    # 4. paste each generated face
    y1, y2, x1, x2 = T_ori_face_coordinates[2][1]  # coordinates of face bounding box
    original_background = T_input_frame[2].copy()
    
    T_input_frame[2][y1:y2, x1:x2] = cv2.resize(gen_face,(x2 - x1, y2 - y1))  #resize and paste generated face
    
    # gt = cv2.resize(T_crop_face[2], (img_size, img_size))
    # 5. post-process
    full = merge_face_contour_only(original_background, T_input_frame[2], T_ori_face_coordinates[2][1],fa)   #(H,W,3)
    # 6.output
    show_heatmap = heatmap_visualize(T_predict_whole_heatmap[0][0][0].cpu().detach().numpy())
    show_heatmap = cv2.resize(show_heatmap, (frame_w, frame_h))
    
    # full = np.concatenate([show_heatmap, full], axis=1) # if you want to show heatmap, cancel comment
    out_stream_full.write(full)
    out_stream.write(gen_face)

    

    
out_stream_full.release()
out_stream.release()

command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(input_audio_path, '{}/result.avi'.format(temp_dir), outfile_path)
subprocess.call(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(input_audio_path, '{}/result_full.avi'.format(temp_dir), result_full_path)
subprocess.call(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
extract_frames(outfile_path, result_imgs_file)
extract_frames(result_full_path, result_full_imgs_file)

print("succeed output results to:", result_full_path)
