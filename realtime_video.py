from __future__ import print_function

import time
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
#from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.model import Face
from utils.box_utils import decode
from utils.timer import Timer
from IPython.display import Image
from matplotlib import pyplot as plt
from torchsummary import summary

forward_time = 0
forward_time_tot = 0

parser = argparse.ArgumentParser(description='Face')

parser.add_argument('-m', '--trained_model', default='weights/Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.5, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--show_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()

def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
			
def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

# ============================================================================



cv2.namedWindow("preview")
vc = cv2.VideoCapture('Video/HD-IR.mp4')
resize = 1
#vc.set(3,1920)
#vc.set(4,1080)
numframes = 100000000
fps = 30

frame_width = int(vc.get(3))
frame_height = int(vc.get(4))
video_name = str(time.strftime("%Y_%m_%d_%H_%M")) + '.avi'
vout = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc('M','J','P','G'), fps,(frame_width,frame_height))

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # net and model
    net = Face(phase='test', size=None, num_classes=2)    # initialize detector
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    #device = torch.device("cpu" if args.cpu else "cuda:0")
    device = torch.device("cpu")
    net = net.to(device) 
    summary(net, (3, 1024, 1024))

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    for n in range(numframes):
        rval, img_raw = vc.read()

        #_t['forward_pass'].tic()

        #img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2YUV)
        #clahe = cv2.createCLAHE(clipLimit=1,tileGridSize=(11,11))
        #img_raw[:,:,0] = clahe.apply(img_raw[:,:,0])
        #img_raw = cv2.cvtColor(img_raw, cv2.COLOR_YUV2BGR)

        img = np.float32(img_raw)
        
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        out = net(img)  # forward pass
        _t['forward_pass'].toc()

        priorbox = PriorBox(cfg, out[2], (im_height, im_width), phase='test')
        priors = priorbox.forward()
        priors = priors.to(device)
        loc, conf, _ = out
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        #keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]

        #print('frames: {:d} forward_pass_time: {:.4f}s'.format(n+1, _t['forward_pass'].average_time))

        forward_time = _t['forward_pass'].average_time
        forward_time_tot = forward_time_tot + forward_time
        fps = "fps = {:.2f}".format(1/forward_time)


        for b in dets:
            if b[4] < args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))

            ratio = (b[2]-b[0])/150
            #print(ratio)
            overlay = img_raw.copy()
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), -1)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, img_raw, 1 - alpha, 0, img_raw)

            overlay = img_raw.copy()
            cv2.rectangle(img_raw, (b[0], b[1]-int(33*ratio)), (b[2], b[1]-int(10*ratio)), (255, 255, 255), -1)
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, img_raw, 1 - alpha, 0, img_raw)

            draw_border(img_raw, (b[0]-int(3*ratio), b[1]-int(3*ratio)), (b[2]+int(3*ratio), b[3]+int(3*ratio)), (0,64,255), 1, int(20*ratio), int(10*ratio))
            alpha = 0.1
            cv2.addWeighted(overlay, alpha, img_raw, 1 - alpha, 0, img_raw)


            cx = b[0]
            cy = b[1] - int(10*ratio)
            cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, ratio, (0, 0, 0))

        cv2.putText(img_raw, fps, (200, 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))                  
        cv2.imshow("preview", img_raw)
        vout.write(img_raw)


        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")

    ave_time = forward_time_tot/n
    print('frame_total: {:d} '.format(n+1))
    print('forward_pass_time_tot: {:.4f}s'.format(forward_time_tot))   
    print('forward_pass_time_ave: {:.4f}s'.format(ave_time))
    print('fps: {:.4f}'.format(1/(ave_time)))
