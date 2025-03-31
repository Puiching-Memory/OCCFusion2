# Step-by-step installation instructions

**a. Create a conda virtual environment and activate it.**

```shell
conda create -n occfusion python=3.12
conda activate occfusion
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**

```shell
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

**c. Install mmengine, mmcv, mmdet, mmdet3d, and mmseg.**

```shell
git clone --branch v1.4.0 https://github.com/open-mmlab/mmdetection3d.git
git clone --branch v2.1.0 https://github.com/open-mmlab/mmcv.git
git clone --branch v3.3.0 https://github.com/open-mmlab/mmdetection.git
git clone --branch v0.10.7 https://github.com/open-mmlab/mmengine.git
git clone --branch v1.2.2 https://github.com/open-mmlab/mmsegmentation.git

cd mmcv
pip install -r requirements/optional.txt
pip install -e . -v
```

**d. Install others.**

```shell
pip install focal_loss_torch
```

**e. Download code and backbone pretrain weight.**

```shell
git clone https://github.com/DanielMing123/OCCFusion.git
cd OCCFusion
mkdir ckpt
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```

**f. Download Fixed param [here](https://drive.google.com/drive/folders/15riDPe25gVZ79jGeamfftBrzRBbcfQjP?usp=sharing). The OCCFusion repo core structure should be like the following**

```
OCCFusion
├── ckpt/
├── configs/
├── fix_param_small/
├── occfusion/
├── tools/
├── data/
│   ├── nuscenes/
```
