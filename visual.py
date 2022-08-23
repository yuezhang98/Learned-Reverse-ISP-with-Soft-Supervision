import os
import numpy as np
import glob
import argparse
import imageio
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from basicsr.models import create_model
from basicsr.utils.options import parse

parser = argparse.ArgumentParser(description='TEST')
parser.add_argument(
        '-opt', default='./opt/SSDNet.yml',
        type=str, required=True, help='Path to option YAML file.')
parser.add_argument("-dataset_dir", type=str, default='../data/S7/competition_val_rgb',
                    help='the folder of the input images')
parser.add_argument("-pretrained_weight", type=str, default='./experiments/SSDNet/models/net_g_276000.pth',
                    required=True, help='the pretrained weight of network')

def parse_options(cfg):
    opt = parse(cfg.opt, is_train=False)    
    opt['dist'] = False
    print('Disable distributed.', flush=True)
    return opt


def test():
    #2os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    cfg = parser.parse_args()
    
    image_names = glob.glob(cfg.dataset_dir+'/*.jpg')
    total_len = len(image_names)
    opt = parse_options(cfg)
    model = create_model(opt)
    
    state_dict  = torch.load(cfg.pretrained_weight) 
    model.net_g.module.load_state_dict(state_dict['params'], strict=True)
    model.net_g.eval()
    cnt = 0
    for filename in image_names:
        print('In processing:', filename, '{0}/{1}'.format(cnt,total_len))
        with torch.no_grad():
            jpg_image = np.asarray(imageio.v2.imread(filename)).astype(np.float32)
            jpg_image = torch.from_numpy(jpg_image.transpose((2, 0, 1)) / 255.).unsqueeze(0)
            jpg_image = jpg_image.to(device, non_blocking=True)
            
            _, _, H, W = jpg_image.size()
            window_size = 16
            pad_flag = False
            if H // 16 != 0 and W // 16 != 0:
                scale = opt.get('scale', 1)
                mod_pad_h, mod_pad_w = 0, 0
                _, _, h, w = jpg_image.size()
                if h % window_size != 0:
                    mod_pad_h = window_size - h % window_size
                if w % window_size != 0:
                    mod_pad_w = window_size - w % window_size
                jpg_image = F.pad(jpg_image, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
                pad_flag = True
            else :
                pass
            
            pred_raw = model.net_g(jpg_image)

            if pad_flag:
                _, _, h, w = pred_raw.size()
                pred_raw = pred_raw[:, :, 0:h - mod_pad_h // scale, 0:w - mod_pad_w // scale]
                jpg_image = jpg_image[:, :, 0:H, 0:W]
            else :
                pass

            pred_raw = pred_raw[0].detach().cpu().permute(1, 2, 0).numpy()             
            

            jpg_image = jpg_image[0].detach().cpu().permute(1, 2, 0).numpy()
            
            # if cnt % 50 == 0:
            #     plot_pair(jpg_image, postprocess_raw(demosaic(pred_raw)))          
            
            cnt += 1           
            
            pred_raw = (pred_raw * 1024).astype(np.uint16)
            
            # Save the results as .png images
            folder = "submission_S7" 
            if not os.path.exists(folder):
                os.makedirs(folder)
            else:
                pass
            basename = os.path.basename(filename)
            assert pred_raw.shape[-1] == 4
            np.save(os.path.join(folder, basename[:-4] + '.npy'), pred_raw)
            
def demosaic (raw):
    """Simple demosaicing to visualize RAW images
    Inputs:
     - raw: (h,w,4) RAW RGGB image normalized [0..1] as float32
    Returns: 
     - Simple Avg. Green Demosaiced RAW image with shape (h*2, w*2, 3)
    """
    
    assert raw.shape[-1] == 4
    shape = raw.shape
    
    red        = raw[:,:,0]
    green_red  = raw[:,:,1]
    green_blue = raw[:,:,2]
    blue       = raw[:,:,3]
    avg_green  = (green_red + green_blue) / 2
    image      = np.stack((red, avg_green, blue), axis=-1)
    image      = cv2.resize(image, (shape[1]*2, shape[0]*2))
    return image


def mosaic(rgb):
    """Extracts RGGB Bayer planes from an RGB image."""
    
    assert rgb.shape[-1] == 3
    shape = rgb.shape
    
    red        = rgb[0::2, 0::2, 0]
    green_red  = rgb[0::2, 1::2, 1]
    green_blue = rgb[1::2, 0::2, 1]
    blue       = rgb[1::2, 1::2, 2]
    
    image = np.stack((red, green_red, green_blue, blue), axis=-1)
    return image


def gamma_compression(image):
    """Converts from linear to gamma space."""
    return np.maximum(image, 1e-8) ** (1.0 / 2.2)

def tonemap(image):
    """Simple S-curved global tonemap"""
    return (3*(image**2)) - (2*(image**3))

def postprocess_raw(raw):
    """Simple post-processing to visualize demosaic RAW imgaes
    Input:  (h,w,3) RAW image normalized
    Output: (h,w,3) post-processed RAW image
    """
    raw = gamma_compression(raw)
    raw = tonemap(raw)
    raw = np.clip(raw, 0, 1)
    return raw

def plot_pair (rgb, raw, t1='RGB', t2='RAW', axis='off'):
    
    fig = plt.figure(figsize=(12, 6), dpi=80)
    plt.subplot(1,2,1)
    plt.title(t1)
    plt.axis(axis)
    plt.imshow(rgb)

    plt.subplot(1,2,2)
    plt.title(t2)
    plt.axis(axis)
    plt.imshow(raw)
    plt.savefig('./test_S7.png')
    #plt.show()


if __name__ == '__main__':
    test()

