# Learned-Reverse-ISP-with-Soft-Supervision
SSDNet for Reversed ISP learning


## Installation

This repository is built in PyTorch 1.10.1 (Python3.7, CUDA11.2, cuDNN7.6).
Follow these intructions

1. Clone our repository
```
git clone https://github.com/yuezhang98/Learned-Reverse-ISP-with-Soft-Supervision.git
cd Learned-Reverse-ISP-with-Soft-Supervision
```

2. Make conda environment
```
conda create -n pytorch1.10 python=3.7
conda activate pytorch1.10
```

3. Install dependencies
```
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```

4. Install basicsr
```
python setup.py develop
```

##  Run 

1. training
```
./train.sh opt/SSDNet.yml
```

2. testing
```
python visual.py
```

**Acknowledgment:** This code is based on the [BasicSR](https://github.com/xinntao/BasicSR) toolbox, [HINet](https://github.com/megvii-model/HINet) and [Restormer](https://github.com/swz30/Restormer).

