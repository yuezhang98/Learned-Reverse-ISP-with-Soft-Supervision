from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import random_augmentation

import random
import numpy as np
import torch
import os
import imageio

def get_patch(*args, patch_size, scale):
    ih, iw = args[0].shape[:2]

    tp = patch_size  # target patch (HR)
    ip = tp // scale  # input patch (LR)

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]  # results
    return ret

class Dataset_PairedImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            input_size (int): Cropped patched size for input patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        
        self.io_backend_opt = opt['io_backend']
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

    def __getitem__(self, index):

        scale = self.opt['scale']
        index = index % len(self.paths)
        
        gt_path = self.paths[index]['gt_path']
        raw_image = np.load(gt_path).astype(np.float32)
        
        lq_path = self.paths[index]['lq_path']
        jpg_image = np.asarray(imageio.imread(lq_path)).astype(np.float32)
        
        if self.opt['phase'] == 'train':
            self.input_size = self.opt['input_size']

            # random crop
            sub_raw, sub_input = get_patch(
                raw_image, jpg_image, patch_size=self.input_size, scale=scale)

            # flip, rotation augmentations
            sub_raw, sub_input = random_augmentation(sub_raw, sub_input)
            
            sub_input = torch.from_numpy(sub_input.transpose((2, 0, 1)) / 255.)
            sub_raw = torch.from_numpy(sub_raw.transpose((2, 0, 1)) / 1024.)
            return {
            'lq': sub_input,
            'gt': sub_raw,
            'lq_path': lq_path,
            'gt_path': gt_path
            }
        else:
            jpg_image = torch.from_numpy(jpg_image.transpose((2, 0, 1)) / 255.)
            raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)) / 1024.)  
            return {
            'lq': jpg_image,
            'gt': raw_image,
            'lq_path': lq_path,
            'gt_path': gt_path
            } 

    def __len__(self):
        return len(self.paths)
