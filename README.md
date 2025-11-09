# ${C}^{3}$-GS

<div align="center">
    <a href="https://yuhsihu.github.io" target='_blank'>Yuxi Hu</a>, 
    <a href="https://halajun.github.io/" target='_blank'>Jun Zhang</a>,
    <a href="https://easonchen99.github.io/Homepage/" target='_blank'>Kuangyi Chen</a>,  
    <a href="https://www.doublez.site" target='_blank'>Zhe Zhang</a>,    
    <a href="https://www.tugraz.at/institute/icg/research/team-fraundorfer/people/friedrich-fraundorfer/" target='_blank'>Friedrich Fraundorfer</a>*
</div>

<br />

<div align="center">

![Publication](https://img.shields.io/badge/2025-BMVC-440099)
[![Paper](http://img.shields.io/badge/arxiv-arxiv.2508.20754-B31B1B?logo=arXiv&logoColor=green)](https://arxiv.org/abs/2508.20754)
![Pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)

</div>

## 📌 Introduction
This repository contains the official implementation of our BMVC 2025 paper: ${C}^{3}$-GS: Learning Context-aware, Cross-dimension, Cross-scale Feature for Generalizable Gaussian Splatting.

<!-- > **TODO: The code is under preparation for release. Stay tuned for updates!** -->

## 🔧 Setup
### 1.1 Requirements
Use the following commands to create a conda environment and install the required packages:
```bash
conda create -n c3gs python=3.7.13
conda activate c3gs
pip install -r requirements.txt
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 -f https://download.pytorch.org/whl/torch_stable.html
```
Install Gaussian Splatting renderer. We are using the [MVPGS](https://github.com/zezeaaa/MVPGS) implementation which returns both rendered image and depth:
```bash
git clone https://github.com/zezeaaa/MVPGS.git --recursive
pip install MVPGS/submodules/diff-gaussian-rasterization
pip install MVPGS/submodules/simple-knn
```

### 1.2 Datasets
- DTU

  Download [DTU data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) and [Depth raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip). Unzip and organize them as:
  ```
  mvs_training
      ├── dtu                   
          ├── Cameras                
          ├── Depths   
          ├── Depths_raw
          └── Rectified
  ```

- Download [NeRF Synthetic](https://drive.google.com/drive/folders/1WAeA7-Ktr9-sFDmoNYgmL3wt8Ltm7-Ys?usp=sharing), [Real Forward-facing](https://drive.google.com/drive/folders/1rciqkjLQEBnoT3lrXWfsJW3s3dHdrV9e?usp=sharing), and [Tanks and Temples](https://drive.google.com/drive/folders/15Q-N5SrD96i3YmQv0EgmzwJj80IBeYhQ?usp=sharing) datasets.

## 🧠 Training
This implementation is built upon the [MVSGaussian](https://github.com/TQTQliu/MVSGaussian) framework, with our modules and improvements integrated into its existing pipeline.

To maintain compatibility, we preserve the original directory and command structure (e.g., paths under `mvsgs/...`).
### 2.1 Training on DTU
To train a generalizable model from scratch on DTU, specify ``data_root`` in ``configs/mvsgs/dtu_pretrain.yaml`` first and then run:
```bash
python train_net.py --cfg_file configs/mvsgs/dtu_pretrain.yaml train.batch_size 4
```
More details can be found in the [MVSGaussian](https://github.com/TQTQliu/MVSGaussian) codebase.

### 2.2 Per-scene optimization
One strategy is to optimize only the initial Gaussian point cloud provided by the generalizable model.
```bash
bash scripts/mvsgs/llff_ft.sh
bash scripts/mvsgs/nerf_ft.sh
bash scripts/mvsgs/tnt_ft.sh
```
More details can be found in the [MVSGaussian](https://github.com/TQTQliu/MVSGaussian) codebase.

## 📊 Testing
### 3.1 Evaluation on DTU
Use the following command to evaluate the model on DTU:
```bash
python run.py --type evaluate --cfg_file configs/mvsgs/dtu_pretrain.yaml mvsgs.cas_config.render_if False,True mvsgs.cas_config.volume_planes 48,8 mvsgs.eval_depth True
```
The rendered images will be saved in ```result/mvsgs/dtu_pretrain```. 

### 3.2 Evaluation on Real Forward-facing
```bash
python run.py --type evaluate --cfg_file configs/mvsgs/llff_eval.yaml
```

### 3.3 Evaluation on NeRF Synthetic
```bash
python run.py --type evaluate --cfg_file configs/mvsgs/nerf_eval.yaml
```

### 3.4 Evaluation on Tanks and Temples
```bash
python run.py --type evaluate --cfg_file configs/mvsgs/tnt_eval.yaml
```

## 📜 Citation
If you find this work useful in your research, please cite:
```bibtex
@inproceedings{hu2025c3gs,
  title     = {{$C^3$-GS: Learning Context-aware, Cross-dimension, Cross-scale Feature for Generalizable Gaussian Splatting}},
  author    = {Hu, Yuxi and Zhang, Jun and Chen, Kuangyi and Zhang, Zhe and Fraundorfer, Friedrich},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2025}
}
```

## ⚙️ Notice
To support practical usage on a single GPU (e.g., RTX 3090/4090), the released code applies the Transformer only at the low-resolution level.

Compared to the original paper setting (Transformer at both low and high resolution), this variant trades off a small drop in accuracy for significantly better efficiency and scalability.

## ❤️ Acknowledgements
This repository builds on the excellent works of [MVSGaussian](https://github.com/TQTQliu/MVSGaussian), [MVSplat](https://github.com/donydchen/mvsplat). We sincerely thank the authors for their contributions to the community.