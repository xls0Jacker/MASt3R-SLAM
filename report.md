[2412.12392v2.pdf](https://leedong25.yuque.com/attachments/yuque/0/2025/pdf/45861457/1760956780764-50e38c08-829b-4468-b52f-fa196fa6e05b.pdf)_[2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)]_

**GitHub**：[https://github.com/rmurai0610/MASt3R-SLAM?tab=readme-ov-file](https://github.com/rmurai0610/MASt3R-SLAM?tab=readme-ov-file)

**DeepWiki**：[https://deepwiki.com/rmurai0610/MASt3R-SLAM?tab=readme-ov-file](https://deepwiki.com/rmurai0610/MASt3R-SLAM?tab=readme-ov-file)

# 论文内容概述
## 研究背景与动机
视觉同步定位与地图构建（SLAM）是当今机器人学和增强现实产品的基础构建模块。通过精心设计集成的硬件与软件栈，现已能实现鲁棒且准确的视觉SLAM。然而，SLAM 非即插即用算法，因其需要硬件专业知识和校准。<u>对于无额外传感器（如惯性测量单元，IMU）的最小单相机配置，目前尚不存在能同时提供精确位姿与一致稠密地图的野外同步定位与地图构建方案</u>。<font style="color:#DF2A3F;">（研究背景）</font>实现这种可靠的稠密 SLAM 系统将为空间智能开辟新的研究途径。<font style="color:#DF2A3F;">（研究动机）、</font>

## 相关工作
为了获得精确的位姿估计，稀疏单目 SLAM 专注于联合求解相机位姿和选定数量的无偏三维路标点。利用优化的稀疏性和精心的图构建的算法进步, 实现了大规模场景的实时位姿估计和稀疏重建。<u>尽管</u>**<u>稀疏单目 SLAM</u>**<u> 在给定足够特征和视差的情况下非常精确，但它</u>**<u>缺乏一个稠密场景模型</u>**<u>，而该模型对于鲁棒跟踪和更显式的几何推理都很有用</u>。

### 稠密单目 SLAM
早期稠密单目SLA M系统展示了**交替优化位姿与稠密深度**的方案，辅以 DTAM[19] **手工正则化**。

由于这些系统局限于受控环境，最近的工作尝试**将数据驱动先验与后端优化**相结合。虽然**单视图几何预测**（如 [11, 15, 31, 53] 深度和 [1, 51] 表面法线）已取得显著进展，但其在 SLAM 中的应用仍受限。

单视图几何预测存在歧义性，会导致有偏且不一致的三维几何。因此 SLAM 研究聚焦于通过潜在空间 [2, 6], 子空间 [41], 局部基元 [24], 和 [8, 9] 分布等形式，**在可能深度的假设空间上预测先验**。

<u>尽管这些先验的灵活性可提升一致性，但跨多视图的鲁棒对应关系至关重要</u>。

### 多视图先验
以 **多视图立体视觉（MVS）**[20, 33, 55] 和**光流** [43] 为代表的多视图先验方法，则专注于从两个或多个视角学习对应关系作为获取几何的手段。

然而，两者都需要额外信息：MVS 固定位姿以实现对应关系，而光流是运动与几何的耦合观测，受限于前文所述的_退化问 题_。

> _退化问题_：尽管多视图先验（如光流）能减少模糊性，但解耦运动与几何仍具挑战——因为像素运动同时取决于外参和相机模型。尽管这些根本原因可能随时间或不同观察者而变化，但 3D 场景在跨视角中保持不变。因此，从图像中求解位姿、相机模型和稠密几何所需的统一先验，应建立在共同坐标系的三维几何空间之上。
>

### 体积表示
体积表示已展现出一致重建的潜力，因其几何参数在渲染过程中相互耦合。多种 SLAM 系统采用神经场 [25] 和高斯泼溅 [18] 中的可微分渲染技术，适用于单目和 RGB‐D 相机。<u>然而相较于替代方案，这些方法的实时性能滞后，且需要深度、额外二维先验或缓慢相机运动来约束解。</u>面向通用场景重建的3D先验最早将二维特征融合为三维体素网格，再解码为表面几何 [27, 40]。此类方法假设已知融合位姿，故不适用于联合跟踪与地图构建，且体积表示需消耗大量内存并依赖预定义分辨率。

### 两视图三维重建先验
最近，DUSt3R 引入了一种新颖的双视图三维重建先验，可在共同坐标系中输出两幅图像的稠密三维点云。相较于先前解决任务子问题的先验方法，DUSt3R 通过隐式推理对应关系、位姿、相机模型和稠密几何，直接提供双视图三维场景的伪测量。

后续方法 MASt3R[21] 预测额外的逐像素特征以改进定位和运动恢复结构的像素匹配[10]。然而与所有先验方法类似，其预测在三维几何中仍可能存在不一致性和相关误差。因此 DUSt3R 和 MASt3R‐SfM 需通过大规模优化确保全局一致性，但时间复杂度无法随图像数量良好扩展。Spann3R[49] 通过微调 DUSt3R 直接将点云图流预测到全局坐标系，从而放弃后端优化，但必须维持有限的 token 内存，这可能导致大场景中的漂移。

<u>在本研究中，我们构建了一个围绕“两视图三维重建先验”的稠密 SLAM 系统。系统仅需通用的中心相机模型，无需任何内参先验；通过高效的点图匹配、跟踪与融合、回环检测以及全局优化，实时地把成对预测拉成大规模全局一致的稠密地图</u>。

## 创新点归纳
+ 首个使用双视图三维重建先验MASt3R [21]作为基础的实时 SLAM系统。
+ 用于点云图匹配、跟踪与局部融合、图构建与闭环检测以及二阶全局优化的高效技术。
+ 一种能够处理通用的时变相机模型的最先进密集 SLAM 系统。

# 项目部署与测试
## 项目部署
### 创建环境 && 安装依赖
```python
conda create -n mast3r-slam python=3.11
conda activate mast3r-slam
```

```python
nvcc --version
```

```python
# CUDA 11.8
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# CUDA 12.4
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

```python
git clone https://github.com/rmurai0610/MASt3R-SLAM.git --recursive
cd MASt3R-SLAM/

# 服务器docker环境无法翻墙，可考虑先在本地下载后将文件传输过去

pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .
 

# Optionally install torchcodec for faster mp4 loading
pip install torchcodec==0.1
```

这一步 pip install --no-build-isolation -e . 中有 "lietorch @ git+[https://github.com/princeton-vl/lietorch.git"](https://github.com/princeton-vl/lietorch.git")，由于 Docker 环境无法翻墙，需要源码安装 lietorch：  
**Step1 下载源码**：[https://github.com/princeton-vl/lietorch](https://github.com/princeton-vl/lietorch)，存放位置不指定，只需要 conda 环境正确即可

**Step2 安装 lietorch 所需依赖**：

```python
# install requirements（前面安装过可以跳过）
pip install torch torchvision torchaudio wheel

# optional: specify GPU architectures
export TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9;9.0"

# install lietorch
pip install --no-build-isolation .
```

**Step3 重新运行指令**：运行之前需要将 "lietorch @ git+[https://github.com/princeton-vl/lietorch.git"](https://github.com/princeton-vl/lietorch.git") 这一行注释掉。

```python
pip install --no-build-isolation -e . 
```

```python
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```

## 数据集下载
scripts 里面的 bash 脚本链接过时，需到官网手动下载，以 TUM RGBD 为例：

在官网 [https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) 下载 scripts/download_tum.sh 中所需数据集并解压，注意保持路径为 datasets/tum

