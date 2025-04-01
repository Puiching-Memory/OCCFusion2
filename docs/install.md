# Step-by-step installation instructions

### 1. Create a conda virtual environment and activate it.

```shell
conda create -n occfusion python=3.12
conda activate occfusion
apt update && apt upgrade -y
apt install libgl1-mesa-glx libglib2.0-0 -y
```

### 2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).

```shell
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4
```

### 3. set env

| GPU       | sm  |
| --------- | --- |
| A100      | 8.0 |
| 3090      | 8.6 |
| H800/H100 | 9.0 |

```
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;ptx"
```

### 4. Install mmengine, mmcv, mmdet, mmdet3d, and mmseg.

```shell
pip install -U pip setuptools

cd nuscenes-devkit
pip install -r setup/requirements.txt
pip install -e setup/ -v
cd ..

cd mmcv
pip install -r requirements/optional.txt
pip install -e . -v
cd ..

cd mmengine
pip install -e . -v
cd ..

cd mmdetection
pip install -e . -v
cd ..

cd mmsegmentation
pip install -e . -v
cd ..

cd mmdetection3d
pip install -e . -v
cd ..
```

### 5. Install others.

```shell
pip install -r requirements.txt
```

### 6. Download code and backbone pretrain weight.

```shell
git clone https://github.com/DanielMing123/OCCFusion.git
cd OCCFusion
mkdir ckpt
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```

### 7. Download Fixed param [here](https://drive.google.com/drive/folders/15riDPe25gVZ79jGeamfftBrzRBbcfQjP?usp=sharing). The OCCFusion repo core structure should be like the following

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
