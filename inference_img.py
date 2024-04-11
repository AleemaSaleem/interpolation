import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Interpolation for a folder of frames')
parser.add_argument('--frame_folder', dest='frame_folder', type=str, required=True,
                    help='Path to the folder containing frames')
parser.add_argument('--exp', default=4, type=int,
                    help='Expansion factor for interpolation (default: 4)')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log',
                    help='Directory with trained model files')
args = parser.parse_args()

try:
    try:
        try:
            from model.RIFE_HDv2 import Model
            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v2.x HD model.")
        except:
            from train_log.RIFE_HDv3 import Model
            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v3.x HD model.")
    except:
        from model.RIFE_HD import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v1.x HD model")
except:
    from model.RIFE import Model
    model = Model()
    model.load_model(args.modelDir, -1)
    print("Loaded ArXiv-RIFE model")
model.eval()
model.device()

frame_files = sorted(os.listdir(args.frame_folder))

output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

for i in range(len(frame_files) - 1):
    frame1_path = os.path.join(args.frame_folder, frame_files[i])
    frame2_path = os.path.join(args.frame_folder, frame_files[i + 1])

    frame1 = cv2.imread(frame1_path, cv2.IMREAD_UNCHANGED)
    frame2 = cv2.imread(frame2_path, cv2.IMREAD_UNCHANGED)
    img0 = (torch.tensor(frame1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    img1 = (torch.tensor(frame2.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)

    img_list = [img0, img1]
    for j in range(args.exp):
        tmp = []
        for k in range(len(img_list) - 1):
            mid = model.inference(img_list[k], img_list[k + 1])
            tmp.append(img_list[k])
            tmp.append(mid)
        tmp.append(img1)
        img_list = tmp

    for j in range(len(img_list)):
        cv2.imwrite(os.path.join(output_folder, 'frame{}_{}.png'.format(i, j)),
                    (img_list[j][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

print("Interpolation completed. Generated frames are saved in the output folder.")
