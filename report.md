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

# 论文公式对应代码
![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1761272569124-5612d316-9299-4584-be5a-d37718120a32.png)

```python
# MASt3R-SLAM/mast3r_slam/backend/src/gn_kernels.cu L813-L1140
/**
 * @brief 点对齐核函数：计算点对点匹配的Hessian矩阵和梯度
 * @param Twc 相机位姿 [num_poses, 8]，世界到相机的变换
 * @param Xs 3D点坐标 [num_poses, num_points, 3]
 * @param Cs 置信度 [num_poses, num_points, 1]
 * @param ii 边的起始节点索引 [num_edges]
 * @param jj 边的终止节点索引 [num_edges]
 * @param idx_ii2_jj 从j到i的匹配索引 [num_edges, num_points]
 * @param valid_match 匹配有效性标志 [num_edges, num_points, 1]
 * @param Q 匹配质量分数 [num_edges, num_points, 1]
 * @param Hs 输出Hessian块矩阵 [4, num_edges, 7, 7]
 * @param gs 输出梯度向量 [2, num_edges, 7]
 * @param sigma_point 点距离的标准差（信息矩阵权重）
 * @param C_thresh 置信度阈值
 * @param Q_thresh 匹配质量阈值
 * 
 * 每个block处理一条边，每个线程处理多个点
 * 计算点对点对齐的残差、Jacobian、Hessian和梯度
 */
__global__ void point_align_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Twc,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Xs,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Cs,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx_ii2_jj,
    const torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> valid_match,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Q,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gs,
    const float sigma_point,
    const float C_thresh,
    const float Q_thresh)
{
 
  // Twc和Xs的第一维是位姿数量
  // ii, jj, Q的第一维是边数量
 
  const int block_id = blockIdx.x;  // 当前block处理的边ID
  const int thread_id = threadIdx.x;  // 线程ID
 
  const int num_points = Xs.size(1);  // 每个位姿的点数
 
  int ix = static_cast<int>(ii[block_id]);  // 边的起始位姿索引
  int jx = static_cast<int>(jj[block_id]);  // 边的终止位姿索引
 
  // 共享内存存储位姿，减少全局内存访问
  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];
  __shared__ float si[1], sj[1], sij[1];
 
  __syncthreads();
 
  // 从全局内存加载位姿到共享内存
  if (thread_id < 3) {
    ti[thread_id] = Twc[ix][thread_id];
    tj[thread_id] = Twc[jx][thread_id];
  }
 
  if (thread_id < 4) {
    qi[thread_id] = Twc[ix][thread_id+3];
    qj[thread_id] = Twc[jx][thread_id+3];
  }
 
  if (thread_id < 1) {
    si[thread_id] = Twc[ix][thread_id+7];
    sj[thread_id] = Twc[jx][thread_id+7];
  }
 
  __syncthreads();
 
  // 计算相对位姿 Tij = Ti^(-1) * Tj
  if (thread_id == 0) {
    relSim3(ti, qi, si, tj, qj, sj, tij, qij, sij);
  }
 
  __syncthreads();
 
  // 局部变量（每个线程独立）
  float Xi[3];  // 位姿i处的点
  float Xj[3];  // 位姿j处的点
  float Xj_Ci[3];  // 将Xj变换到相机i坐标系
 
  // 残差
  float err[3];
  float w[3];  // 权重
 
  // Jacobian矩阵（14 = 7+7，对应两个位姿）
  float Jx[14];
  // float Jz;
 
  float* Ji = &Jx[0];  // 对位姿i的Jacobian
  float* Jj = &Jx[7];  // 对位姿j的Jacobian
 
  // Hessian矩阵（上三角存储，14x14矩阵需要14*(14+1)/2=105个元素）
  const int h_dim = 14*(14+1)/2;
  float hij[h_dim];
 
  float vi[7], vj[7];  // 梯度向量
 
  int l; // 稍后在Hessian填充时重用此变量
  // 初始化Hessian为0
  for (l=0; l<h_dim; l++) {
    hij[l] = 0;
  }
 
  // 初始化梯度为0
  for (int n=0; n<7; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }
 
  // 参数
  const float sigma_point_inv = 1.0/sigma_point;  // 信息矩阵权重
 
  __syncthreads();
 
  // 每个线程遍历部分点
  GPU_1D_KERNEL_LOOP(k, num_points) {
 
    // 获取点对应关系
    const bool valid_match_ind = valid_match[block_id][k][0];  // 匹配是否有效
    const int64_t ind_Xi = valid_match_ind ? idx_ii2_jj[block_id][k] : 0;  // j中的点k对应i中的索引

    // 读取位姿i处的点
    Xi[0] = Xs[ix][ind_Xi][0];
    Xi[1] = Xs[ix][ind_Xi][1];
    Xi[2] = Xs[ix][ind_Xi][2];
 
    // 读取位姿j处的点
    Xj[0] = Xs[jx][k][0];
    Xj[1] = Xs[jx][k][1];
    Xj[2] = Xs[jx][k][2];
 
    // 将点Xj变换到相机i的坐标系：Xj_Ci = Tij * Xj
    actSim3(tij, qij, sij, Xj, Xj_Ci);
 
    // 计算残差（点之间的3D差异）
    err[0] = Xj_Ci[0] - Xi[0];
    err[1] = Xj_Ci[1] - Xi[1];
    err[2] = Xj_Ci[2] - Xi[2];
 
    // 计算权重（基于置信度和Huber核）
    const float q = Q[block_id][k][0];  // 匹配质量
    const float ci = Cs[ix][ind_Xi][0];  // 点i的置信度
    const float cj = Cs[jx][k][0];  // 点j的置信度
    const bool valid = 
      valid_match_ind
      & (q > Q_thresh)
      & (ci > C_thresh)
      & (cj > C_thresh);

    // 使用置信度加权
    const float conf_weight = q;
    // const float conf_weight = q * ci * cj;  // 可选：使用所有置信度的乘积
    
    // 计算sqrt(weight)，用于鲁棒核函数
    const float sqrt_w_point = valid ? sigma_point_inv * sqrtf(conf_weight) : 0;
 
    // 应用Huber鲁棒权重
    w[0] = huber(sqrt_w_point * err[0]);
    w[1] = huber(sqrt_w_point * err[1]);
    w[2] = huber(sqrt_w_point * err[2]);
    
    // 将sigma权重加回（完整权重 = sigma^2 * Huber权重）
    const float w_const_point = sqrt_w_point * sqrt_w_point;
    w[0] *= w_const_point;
    w[1] *= w_const_point;
    w[2] *= w_const_point;
 
    // 计算Jacobian矩阵
    // 残差对位姿的导数：d(err)/d(T) = d(Xj_Ci - Xi)/d(T)
    
    // X坐标的Jacobian
    // 对位姿j的导数（在相机i坐标系中的表示）
    Ji[0] = 1.0;  // 对tx的导数
    Ji[1] = 0.0;  // 对ty的导数
    Ji[2] = 0.0;  // 对tz的导数
    Ji[3] = 0.0;  // 对旋转wx的导数
    Ji[4] = Xj_Ci[2];  // 对旋转wy的导数：z
    Ji[5] = -Xj_Ci[1];  // 对旋转wz的导数：-y
    Ji[6] = Xj_Ci[0];  // 对缩放s的导数：x

    // 转换到全局坐标系，得到对位姿i的Jacobian
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];  // Ji = -Jj（负号因为是对Ti求导）

    // 累加Hessian矩阵：H += J^T * w * J
    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[0] * Jx[n] * Jx[m];  // 上三角存储
        l++;
      }
    }
 
    // 累加梯度向量：g += J^T * w * err
    for (int n=0; n<7; n++) {
      vi[n] += w[0] * err[0] * Ji[n];
      vj[n] += w[0] * err[0] * Jj[n];
    }
 
    // Y坐标的Jacobian
    Ji[0] = 0.0;
    Ji[1] = 1.0;  // 对ty的导数
    Ji[2] = 0.0;
    Ji[3] = -Xj_Ci[2];  // 对旋转wx的导数：-z
    Ji[4] = 0;  // 对旋转wy的导数
    Ji[5] = Xj_Ci[0];  // 对旋转wz的导数：x
    Ji[6] = Xj_Ci[1];  // 对缩放s的导数：y
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    // 累加Hessian
    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[1] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    // 累加梯度
    for (int n=0; n<7; n++) {
      vi[n] += w[1] * err[1] * Ji[n];
      vj[n] += w[1] * err[1] * Jj[n];
    }
 
    // Z坐标的Jacobian
    Ji[0] = 0.0;
    Ji[1] = 0.0;
    Ji[2] = 1.0;  // 对tz的导数
    Ji[3] = Xj_Ci[1];  // 对旋转wx的导数：y
    Ji[4] = -Xj_Ci[0];  // 对旋转wy的导数：-x
    Ji[5] = 0;  // 对旋转wz的导数
    Ji[6] = Xj_Ci[2];  // 对缩放s的导数：z
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    // 累加Hessian
    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[2] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    // 累加梯度
    for (int n=0; n<7; n++) {
      vi[n] += w[2] * err[2] * Ji[n];
      vj[n] += w[2] * err[2] * Jj[n];
    }
 
 
  }  // 结束点循环
 
  __syncthreads();
 
  // 使用block归约将所有线程的结果汇总
  __shared__ float sdata[THREADS];
  // 归约梯度向量
  for (int n=0; n<7; n++) {
    // 归约位姿i的梯度
    sdata[threadIdx.x] = vi[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[0][block_id][n] = sdata[0];  // 线程0写回结果
    }
 
    __syncthreads();
 
    // 归约位姿j的梯度
    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[1][block_id][n] = sdata[0];
    }
 
  }
 
  // 归约Hessian矩阵（14x14，分为4个7x7块）
  l=0;
  for (int n=0; n<14; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);
 
      if (threadIdx.x == 0) {
        // 根据索引将Hessian写入对应的块
        if (n<7 && m<7) {
          // Hii块（左上）
          Hs[0][block_id][n][m] = sdata[0];
          Hs[0][block_id][m][n] = sdata[0];  // 对称矩阵
        }
        else if (n >=7 && m<7) {
          // Hij和Hji块（非对角）
          Hs[1][block_id][m][n-7] = sdata[0];
          Hs[2][block_id][n-7][m] = sdata[0];
        }
        else {
          // Hjj块（右下）
          Hs[3][block_id][n-7][m-7] = sdata[0];
          Hs[3][block_id][m-7][n-7] = sdata[0];  // 对称矩阵
        }
      }
 
      l++;
    }
  }
}

```

```python
# MASt3R-SLAM/mast3r_slam/backend/src/gn_kernels.cu L1140-L1228
/**
 * @brief 基于点对点匹配的Gauss-Newton优化（CUDA实现）
 * @param Twc 相机位姿 [num_poses, 8]
 * @param Xs 3D点坐标 [num_poses, num_points, 3]
 * @param Cs 置信度 [num_poses, num_points, 1]
 * @param ii 边的起始节点索引
 * @param jj 边的终止节点索引
 * @param idx_ii2jj 匹配索引
 * @param valid_match 匹配有效性
 * @param Q 匹配质量
 * @param sigma_point 点距离标准差
 * @param C_thresh 置信度阈值
 * @param Q_thresh 质量阈值
 * @param max_iter 最大迭代次数
 * @param delta_thresh 收敛阈值
 * @return 最后一次迭代的增量（用于调试）
 * 
 * 迭代优化相机位姿，最小化点对点的3D距离
 */
std::vector<torch::Tensor> gauss_newton_points_cuda(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const float sigma_point,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh)
{
  auto opts = Twc.options();  // 获取tensor的设备和类型选项
  const int num_edges = ii.size(0);  // 边数量
  const int num_poses = Xs.size(0);  // 位姿数量
  const int n = Xs.size(1);  // 点数量

  const int num_fix = 1;  // 固定的位姿数量（通常是第一帧）

  // 设置索引
  torch::Tensor unique_kf_idx = get_unique_kf_idx(ii, jj);  // 获取所有唯一的关键帧索引
  // 用于边构建的索引
  std::vector<torch::Tensor> inds = create_inds(unique_kf_idx, 0, ii, jj);
  torch::Tensor ii_edge = inds[0];
  torch::Tensor jj_edge = inds[1];
  // 用于线性系统索引（固定第一帧）
  std::vector<torch::Tensor> inds_opt = create_inds(unique_kf_idx, num_fix, ii, jj);
  torch::Tensor ii_opt = inds_opt[0];
  torch::Tensor jj_opt = inds_opt[1];

  const int pose_dim = 7;  // Sim(3)的维度

  // 初始化缓冲区
  torch::Tensor Hs = torch::zeros({4, num_edges, pose_dim, pose_dim}, opts);  // Hessian块矩阵
  torch::Tensor gs = torch::zeros({2, num_edges, pose_dim}, opts);  // 梯度向量

  // 用于调试输出
  torch::Tensor dx;

  torch::Tensor delta_norm;  // 增量范数

  // Gauss-Newton迭代
  for (int itr=0; itr<max_iter; itr++) {

    // 调用kernel计算Hessian和梯度
    point_align_kernel<<<num_edges, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      Xs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Cs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      ii_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      idx_ii2jj.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
      valid_match.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
      Q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      gs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      sigma_point, C_thresh, Q_thresh
    );


    // 构建稀疏线性系统：位姿×位姿块
    SparseBlock A(num_poses - num_fix, pose_dim);

    // 更新Hessian矩阵（左端项）
    A.update_lhs(Hs.reshape({-1, pose_dim, pose_dim}), 
        torch::cat({ii_opt, ii_opt, jj_opt, jj_opt}), 
        torch::cat({ii_opt, jj_opt, ii_opt, jj_opt}));

    // 更新梯度向量（右端项）
    A.update_rhs(gs.reshape({-1, pose_dim}), 
        torch::cat({ii_opt, jj_opt}));

    // 求解线性系统：A*dx = -b
    // 注意：这里考虑了负号，因为求解的是下降方向
    dx = -A.solve();
    
    // 在Sim(3)流形上应用增量
    pose_retr_kernel<<<1, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      num_fix);

    // 检查终止条件
    // 需要指定第二个参数，否则函数调用会有歧义
    delta_norm = torch::linalg::linalg_norm(dx, std::optional<c10::Scalar>(), {}, false, {});
    if (delta_norm.item<float>() < delta_thresh) {
      break;  // 增量足够小，收敛
    }
        

  }

  return {dx};  // 返回最后一次迭代的增量（用于调试）
}
```

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1761275583865-be6d85e6-3468-4bc5-9834-dc5114f2c56e.png)

```python
# MASt3R-SLAM/mast3r_slam/backend/src/gn_kernels.cu L455-L723
/**
 * @brief 射线对齐核函数：计算归一化射线匹配的Hessian矩阵和梯度
 * @param Twc 相机位姿 [num_poses, 8]
 * @param Xs 3D点坐标 [num_poses, num_points, 3]
 * @param Cs 置信度 [num_poses, num_points, 1]
 * @param ii 边的起始节点索引
 * @param jj 边的终止节点索引
 * @param idx_ii2_jj 匹配索引
 * @param valid_match 匹配有效性
 * @param Q 匹配质量
 * @param Hs 输出Hessian块矩阵
 * @param gs 输出梯度向量
 * @param sigma_ray 射线方向的标准差
 * @param sigma_dist 距离的标准差
 * @param C_thresh 置信度阈值
 * @param Q_thresh 匹配质量阈值
 * 
 * 优化射线方向（归一化）和距离，而不是直接优化3D点位置
 * 这对于尺度模糊的场景更加鲁棒
 */
__global__ void ray_align_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Twc,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Xs,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Cs,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx_ii2_jj,
    const torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> valid_match,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Q,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gs,
    const float sigma_ray,
    const float sigma_dist,
    const float C_thresh,
    const float Q_thresh)
{
 
  // Twc和Xs的第一维是位姿数量
  // ii, jj, Q的第一维是边数量
 
  const int block_id = blockIdx.x;  // 当前block处理的边ID
  const int thread_id = threadIdx.x;  // 线程ID
 
  const int num_points = Xs.size(1);  // 每个位姿的点数
 
  int ix = static_cast<int>(ii[block_id]);  // 边的起始位姿索引
  int jx = static_cast<int>(jj[block_id]);  // 边的终止位姿索引
 
  // 共享内存存储位姿
  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];
  __shared__ float si[1], sj[1], sij[1];
 
  __syncthreads();
 
  // 从全局内存加载位姿到共享内存
  if (thread_id < 3) {
    ti[thread_id] = Twc[ix][thread_id];
    tj[thread_id] = Twc[jx][thread_id];
  }
 
  if (thread_id < 4) {
    qi[thread_id] = Twc[ix][thread_id+3];
    qj[thread_id] = Twc[jx][thread_id+3];
  }
 
  if (thread_id < 1) {
    si[thread_id] = Twc[ix][thread_id+7];
    sj[thread_id] = Twc[jx][thread_id+7];
  }
 
  __syncthreads();
 
  // 计算相对位姿
  if (thread_id == 0) {
    relSim3(ti, qi, si, tj, qj, sj, tij, qij, sij);
  }
 
  __syncthreads();
 
  // 局部变量
  float Xi[3];  // 位姿i处的点
  float Xj[3];  // 位姿j处的点
  float Xj_Ci[3];  // 变换到相机i坐标系的点
 
  // 残差（4维：3维射线方向 + 1维距离）
  float err[4];
  float w[4];  // 权重
 
  // Jacobian矩阵
  float Jx[14];
  // float Jz;
 
  float* Ji = &Jx[0];  // 对位姿i的Jacobian
  float* Jj = &Jx[7];  // 对位姿j的Jacobian
 
  // Hessian矩阵（上三角存储）
  const int h_dim = 14*(14+1)/2;
  float hij[h_dim];
 
  float vi[7], vj[7];  // 梯度向量
 
  int l; // 稍后在Hessian填充时重用
  // 初始化Hessian为0
  for (l=0; l<h_dim; l++) {
    hij[l] = 0;
  }
 
  // 初始化梯度为0
  for (int n=0; n<7; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }
 
  // 参数
  const float sigma_ray_inv = 1.0/sigma_ray;  // 射线方向的信息矩阵权重
  const float sigma_dist_inv = 1.0/sigma_dist;  // 距离的信息矩阵权重
 
  __syncthreads();
 
  // 每个线程遍历部分点
  GPU_1D_KERNEL_LOOP(k, num_points) {
 
    // 获取点对应关系
    const bool valid_match_ind = valid_match[block_id][k][0]; 
    const int64_t ind_Xi = valid_match_ind ? idx_ii2_jj[block_id][k] : 0;

    // 读取位姿i处的点
    Xi[0] = Xs[ix][ind_Xi][0];
    Xi[1] = Xs[ix][ind_Xi][1];
    Xi[2] = Xs[ix][ind_Xi][2];
 
    // 读取位姿j处的点
    Xj[0] = Xs[jx][k][0];
    Xj[1] = Xs[jx][k][1];
    Xj[2] = Xs[jx][k][2];
 
    // 归一化测量点（计算单位方向向量）
    const float norm2_i = squared_norm3(Xi);  // ||Xi||^2
    const float norm1_i = sqrtf(norm2_i);  // ||Xi||
    const float norm1_i_inv = 1.0/norm1_i;    
    
    // 归一化射线：ri = Xi / ||Xi||
    float ri[3];
    for (int i=0; i<3; i++) ri[i] = norm1_i_inv * Xi[i];
 
    // 将点Xj变换到相机i的坐标系
    actSim3(tij, qij, sij, Xj, Xj_Ci);
 
    // 计算预测点的范数
    const float norm2_j = squared_norm3(Xj_Ci);  // ||Xj_Ci||^2
    const float norm1_j = sqrtf(norm2_j);  // ||Xj_Ci||
    const float norm1_j_inv = 1.0/norm1_j;

    // 归一化预测射线：rj_Ci = Xj_Ci / ||Xj_Ci||
    float rj_Ci[3];
    for (int i=0; i<3; i++) rj_Ci[i] = norm1_j_inv * Xj_Ci[i];
 
    // 计算残差（射线方向差异 + 距离差异）
    err[0] = rj_Ci[0] - ri[0];  // x方向射线误差
    err[1] = rj_Ci[1] - ri[1];  // y方向射线误差
    err[2] = rj_Ci[2] - ri[2];  // z方向射线误差
    err[3] = norm1_j - norm1_i;  // 距离误差
 
    // 计算权重
    const float q = Q[block_id][k][0];  // 匹配质量
    const float ci = Cs[ix][ind_Xi][0];  // 点i的置信度
    const float cj = Cs[jx][k][0];  // 点j的置信度
    const bool valid = 
      valid_match_ind
      & (q > Q_thresh)
      & (ci > C_thresh)
      & (cj > C_thresh);

    // 使用置信度加权
    const float conf_weight = q;
    // const float conf_weight = q * ci * cj;  // 可选：使用所有置信度的乘积
    
    // 计算sqrt(weight)
    const float sqrt_w_ray = valid ? sigma_ray_inv * sqrtf(conf_weight) : 0;
    const float sqrt_w_dist = valid ? sigma_dist_inv * sqrtf(conf_weight) : 0;
 
    // 应用Huber鲁棒权重
    w[0] = huber(sqrt_w_ray * err[0]);
    w[1] = huber(sqrt_w_ray * err[1]);
    w[2] = huber(sqrt_w_ray * err[2]);
    w[3] = huber(sqrt_w_dist * err[3]);
    
    // 将sigma权重加回
    const float w_const_ray = sqrt_w_ray * sqrt_w_ray;
    const float w_const_dist = sqrt_w_dist * sqrt_w_dist;
    w[0] *= w_const_ray;
    w[1] *= w_const_ray;
    w[2] *= w_const_ray;
    w[3] *= w_const_dist;
 
    // 计算Jacobian矩阵
    // 归一化操作的导数：d(P/||P||)/dP = (I - P*P^T/||P||^2) / ||P||
    const float norm3_j_inv = norm1_j_inv / norm2_j;  // 1 / ||P||^3
    const float drx_dPx = norm1_j_inv - Xj_Ci[0]*Xj_Ci[0]*norm3_j_inv;  // d(rx)/d(Px)
    const float dry_dPy = norm1_j_inv - Xj_Ci[1]*Xj_Ci[1]*norm3_j_inv;  // d(ry)/d(Py)
    const float drz_dPz = norm1_j_inv - Xj_Ci[2]*Xj_Ci[2]*norm3_j_inv;  // d(rz)/d(Pz)
    const float drx_dPy = - Xj_Ci[0]*Xj_Ci[1]*norm3_j_inv;  // d(rx)/d(Py)
    const float drx_dPz = - Xj_Ci[0]*Xj_Ci[2]*norm3_j_inv;  // d(rx)/d(Pz)
    const float dry_dPz = - Xj_Ci[1]*Xj_Ci[2]*norm3_j_inv;  // d(ry)/d(Pz)
 
    // rx坐标的Jacobian
    Ji[0] = drx_dPx;  // 对tx的导数
    Ji[1] = drx_dPy;  // 对ty的导数
    Ji[2] = drx_dPz;  // 对tz的导数
    Ji[3] = 0.0;  // 对旋转wx的导数
    Ji[4] = rj_Ci[2];  // 对旋转wy的导数：z
    Ji[5] = -rj_Ci[1];  // 对旋转wz的导数：-y
    Ji[6] = 0.0;  // 对缩放的导数（归一化后与缩放无关）

    // 转换到全局坐标系
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    // 累加Hessian矩阵
    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[0] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    // 累加梯度向量
    for (int n=0; n<7; n++) {
      vi[n] += w[0] * err[0] * Ji[n];
      vj[n] += w[0] * err[0] * Jj[n];
    }
 
    // ry坐标的Jacobian
    Ji[0] = drx_dPy;  // d(ry)/d(Px) = d(rx)/d(Py)
    Ji[1] = dry_dPy;  // 对ty的导数
    Ji[2] = dry_dPz;  // 对tz的导数
    Ji[3] = -rj_Ci[2];  // 对旋转wx的导数：-z
    Ji[4] = 0.0;  // 对旋转wy的导数
    Ji[5] = rj_Ci[0];  // 对旋转wz的导数：x
    Ji[6] = 0.0;  // 对缩放的导数
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    // 累加Hessian
    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[1] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    // 累加梯度
    for (int n=0; n<7; n++) {
      vi[n] += w[1] * err[1] * Ji[n];
      vj[n] += w[1] * err[1] * Jj[n];
    }
 
    // rz坐标的Jacobian
    Ji[0] = drx_dPz;  // d(rz)/d(Px) = d(rx)/d(Pz)
    Ji[1] = dry_dPz;  // d(rz)/d(Py) = d(ry)/d(Pz)
    Ji[2] = drz_dPz;  // 对tz的导数
    Ji[3] = rj_Ci[1];  // 对旋转wx的导数：y
    Ji[4] = -rj_Ci[0];  // 对旋转wy的导数：-x
    Ji[5] = 0.0;  // 对旋转wz的导数
    Ji[6] = 0.0;  // 对缩放的导数
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    // 累加Hessian
    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[2] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    // 累加梯度
    for (int n=0; n<7; n++) {
      vi[n] += w[2] * err[2] * Ji[n];
      vj[n] += w[2] * err[2] * Jj[n];
    }


    // 距离坐标的Jacobian
    // d(||P||)/dP = P / ||P|| = 归一化向量
    Ji[0] = rj_Ci[0];  // 对tx的导数
    Ji[1] = rj_Ci[1];  // 对ty的导数
    Ji[2] = rj_Ci[2];  // 对tz的导数
    Ji[3] = 0.0;  // 对旋转wx的导数（距离与旋转无关）
    Ji[4] = 0.0;  // 对旋转wy的导数
    Ji[5] = 0.0;  // 对旋转wz的导数
    Ji[6] = norm1_j;  // 对缩放s的导数：||P|| * ds = ||P||
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    // 累加Hessian
    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[3] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    // 累加梯度
    for (int n=0; n<7; n++) {
      vi[n] += w[3] * err[3] * Ji[n];
      vj[n] += w[3] * err[3] * Jj[n];
    }
 
 
  }  // 结束点循环
 
  __syncthreads();
 
  // 使用block归约将所有线程的结果汇总
  __shared__ float sdata[THREADS];
  // 归约梯度向量
  for (int n=0; n<7; n++) {
    // 归约位姿i的梯度
    sdata[threadIdx.x] = vi[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[0][block_id][n] = sdata[0];
    }
 
    __syncthreads();
 
    // 归约位姿j的梯度
    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[1][block_id][n] = sdata[0];
    }
 
  }
 
  l=0;
  for (int n=0; n<14; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);
 
      if (threadIdx.x == 0) {
        if (n<7 && m<7) {
          Hs[0][block_id][n][m] = sdata[0];
          Hs[0][block_id][m][n] = sdata[0];
        }
        else if (n >=7 && m<7) {
          Hs[1][block_id][m][n-7] = sdata[0];
          Hs[2][block_id][n-7][m] = sdata[0];
        }
        else {
          Hs[3][block_id][n-7][m-7] = sdata[0];
          Hs[3][block_id][m-7][n-7] = sdata[0];
        }
      }
 
      l++;
    }
  }
}
```

```python
# MASt3R-SLAM/mast3r_slam/backend/src/gn_kernels.cu L725-L811
/**
 * @brief 基于射线对齐的Gauss-Newton优化（CUDA实现）
 * @param Twc 相机位姿
 * @param Xs 3D点坐标
 * @param Cs 置信度
 * @param ii 边的起始节点索引
 * @param jj 边的终止节点索引
 * @param idx_ii2jj 匹配索引
 * @param valid_match 匹配有效性
 * @param Q 匹配质量
 * @param sigma_ray 射线方向标准差
 * @param sigma_dist 距离标准差
 * @param C_thresh 置信度阈值
 * @param Q_thresh 质量阈值
 * @param max_iter 最大迭代次数
 * @param delta_thresh 收敛阈值
 * @return 最后一次迭代的增量
 * 
 * 优化归一化射线方向和距离，适用于尺度模糊场景
 */
std::vector<torch::Tensor> gauss_newton_rays_cuda(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const float sigma_ray,
  const float sigma_dist,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh)
{
  auto opts = Twc.options();  // 获取tensor选项
  const int num_edges = ii.size(0);  // 边数量
  const int num_poses = Xs.size(0);  // 位姿数量
  const int n = Xs.size(1);  // 点数量

  const int num_fix = 1;  // 固定位姿数量

  // 设置索引
  torch::Tensor unique_kf_idx = get_unique_kf_idx(ii, jj);
  // For edge construction
  std::vector<torch::Tensor> inds = create_inds(unique_kf_idx, 0, ii, jj);
  torch::Tensor ii_edge = inds[0];
  torch::Tensor jj_edge = inds[1];
  // For linear system indexing (pin=2 because fixing first two poses)
  std::vector<torch::Tensor> inds_opt = create_inds(unique_kf_idx, num_fix, ii, jj);
  torch::Tensor ii_opt = inds_opt[0];
  torch::Tensor jj_opt = inds_opt[1];

  const int pose_dim = 7; // sim3

  // initialize buffers
  torch::Tensor Hs = torch::zeros({4, num_edges, pose_dim, pose_dim}, opts);
  torch::Tensor gs = torch::zeros({2, num_edges, pose_dim}, opts);

  // For debugging outputs
  torch::Tensor dx;

  torch::Tensor delta_norm;

  for (int itr=0; itr<max_iter; itr++) {

    ray_align_kernel<<<num_edges, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      Xs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Cs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      ii_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      idx_ii2jj.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
      valid_match.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
      Q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      gs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      sigma_ray, sigma_dist, C_thresh, Q_thresh
    );


    // pose x pose block
    SparseBlock A(num_poses - num_fix, pose_dim);

    A.update_lhs(Hs.reshape({-1, pose_dim, pose_dim}), 
        torch::cat({ii_opt, ii_opt, jj_opt, jj_opt}), 
        torch::cat({ii_opt, jj_opt, ii_opt, jj_opt}));

    A.update_rhs(gs.reshape({-1, pose_dim}), 
        torch::cat({ii_opt, jj_opt}));

    // NOTE: Accounting for negative here!
    dx = -A.solve();

    //
    pose_retr_kernel<<<1, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      num_fix);

    // Termination criteria
    // Need to specify this second argument otherwise ambiguous function call...
    delta_norm = torch::linalg::linalg_norm(dx, std::optional<c10::Scalar>(), {}, false, {});
    if (delta_norm.item<float>() < delta_thresh) {
      break;
    }
        

  }

  return {dx}; // 返回调试信息
}

```

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1761276026051-4ac77389-617f-41de-9352-f996cb1d2a88.png)（已知标定）

```python
# MASt3R-SLAM/mast3r_slam/backend/src/gn_kernels.cu L1231-L1543
/**
 * @brief 标定投影核函数：计算像素投影残差的Hessian矩阵和梯度
 * @param Twc 相机位姿
 * @param Xs 3D点坐标
 * @param Cs 置信度
 * @param K 相机内参矩阵 [3, 3]
 * @param ii 边的起始节点索引
 * @param jj 边的终止节点索引
 * @param idx_ii2_jj 匹配索引
 * @param valid_match 匹配有效性
 * @param Q 匹配质量
 * @param Hs 输出Hessian块矩阵
 * @param gs 输出梯度向量
 * @param height 图像高度
 * @param width 图像宽度
 * @param pixel_border 像素边界（忽略图像边缘的像素）
 * @param z_eps 深度阈值（避免负深度）
 * @param sigma_pixel 像素误差标准差
 * @param sigma_depth 深度误差标准差
 * @param C_thresh 置信度阈值
 * @param Q_thresh 质量阈值
 * 
 * 使用已知相机内参优化位姿，最小化像素重投影误差和深度误差
 */
__global__ void calib_proj_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Twc,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Xs,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Cs,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> K,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx_ii2_jj,
    const torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> valid_match,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Q,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gs,
    const int height,
    const int width,
    const int pixel_border,
    const float z_eps,
    const float sigma_pixel,
    const float sigma_depth,
    const float C_thresh,
    const float Q_thresh)
{
 
  // Twc和Xs的第一维是位姿数量
  // ii, jj, Q的第一维是边数量
 
  const int block_id = blockIdx.x;  // 当前block处理的边ID
  const int thread_id = threadIdx.x;  // 线程ID
 
  const int num_points = Xs.size(1);  // 每个位姿的点数
 
  int ix = static_cast<int>(ii[block_id]);  // 边的起始位姿索引
  int jx = static_cast<int>(jj[block_id]);  // 边的终止位姿索引

  // 共享内存存储相机内参
  __shared__ float fx;  // 焦距x
  __shared__ float fy;  // 焦距y
  __shared__ float cx;  // 主点x
  __shared__ float cy;  // 主点y
 
  // 共享内存存储位姿
  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];
  __shared__ float si[1], sj[1], sij[1];

  // 从全局内存加载相机内参到共享内存
  if (thread_id == 0) {
    fx = K[0][0];
    fy = K[1][1];
    cx = K[0][2];
    cy = K[1][2];
  }
 
  __syncthreads();
 
  // 从全局内存加载位姿到共享内存
  if (thread_id < 3) {
    ti[thread_id] = Twc[ix][thread_id];
    tj[thread_id] = Twc[jx][thread_id];
  }
 
  if (thread_id < 4) {
    qi[thread_id] = Twc[ix][thread_id+3];
    qj[thread_id] = Twc[jx][thread_id+3];
  }
 
  if (thread_id < 1) {
    si[thread_id] = Twc[ix][thread_id+7];
    sj[thread_id] = Twc[jx][thread_id+7];
  }
 
  __syncthreads();
 
  // 计算相对位姿
  if (thread_id == 0) {
    relSim3(ti, qi, si, tj, qj, sj, tij, qij, sij);
  }
 
  __syncthreads();
 
  // 局部变量
  float Xi[3];  // 位姿i处的点
  float Xj[3];  // 位姿j处的点
  float Xj_Ci[3];  // 变换到相机i坐标系的点
 
  // 残差（3维：2维像素误差 + 1维深度误差）
  float err[3];
  float w[3];  // 权重
 
  // Jacobian矩阵
  float Jx[14];
  // float Jz;
 
  float* Ji = &Jx[0];  // 对位姿i的Jacobian
  float* Jj = &Jx[7];  // 对位姿j的Jacobian
 
  // Hessian矩阵（上三角存储）
  const int h_dim = 14*(14+1)/2;
  float hij[h_dim];
 
  float vi[7], vj[7];  // 梯度向量
 
  int l; // 稍后在Hessian填充时重用
  // 初始化Hessian为0
  for (l=0; l<h_dim; l++) {
    hij[l] = 0;
  }
 
  // 初始化梯度为0
  for (int n=0; n<7; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }
 
  // 参数
  const float sigma_pixel_inv = 1.0/sigma_pixel;  // 像素误差的信息矩阵权重
  const float sigma_depth_inv = 1.0/sigma_depth;  // 深度误差的信息矩阵权重
 
  __syncthreads();
 
  // 每个线程遍历部分点
  GPU_1D_KERNEL_LOOP(k, num_points) {
 
    // 获取点对应关系
    const bool valid_match_ind = valid_match[block_id][k][0]; 
    const int64_t ind_Xi = valid_match_ind ? idx_ii2_jj[block_id][k] : 0;

    // 读取位姿i处的点
    Xi[0] = Xs[ix][ind_Xi][0];
    Xi[1] = Xs[ix][ind_Xi][1];
    Xi[2] = Xs[ix][ind_Xi][2];
 
    // 读取位姿j处的点
    Xj[0] = Xs[jx][k][0];
    Xj[1] = Xs[jx][k][1];
    Xj[2] = Xs[jx][k][2];

    // 获取测量像素坐标（从线性索引转换）
    const int u_target = ind_Xi % width;  // 目标像素u
    const int v_target = ind_Xi / width;  // 目标像素v
 
    // 将点Xj变换到相机i的坐标系
    actSim3(tij, qij, sij, Xj, Xj_Ci);

    // 检查点是否在相机前方（深度为正）
    const bool valid_z = ((Xj_Ci[2] > z_eps) && (Xi[2] > z_eps));

    // 处理深度相关变量（避免除零）
    const float zj_inv = valid_z ? 1.0/Xj_Ci[2] : 0.0;  // 深度倒数
    const float zj_log = valid_z ? logf(Xj_Ci[2]) : 0.0;  // log深度（预测）
    const float zi_log = valid_z ? logf(Xi[2]) : 0.0;  // log深度（测量）

    // 将3D点投影到像素平面
    const float x_div_z = Xj_Ci[0] * zj_inv;  // X/Z
    const float y_div_z = Xj_Ci[1] * zj_inv;  // Y/Z
    const float u = fx * x_div_z + cx;  // u = fx * X/Z + cx
    const float v = fy * y_div_z + cy;  // v = fy * Y/Z + cy

    // 检查投影是否在图像范围内（排除边界）
    const bool valid_u = ((u > pixel_border) && (u < width - 1 - pixel_border));
    const bool valid_v = ((v > pixel_border) && (v < height - 1 - pixel_border));

    // 计算残差（像素重投影误差 + 对数深度误差）
    err[0] = u - u_target;  // u方向像素误差
    err[1] = v - v_target;  // v方向像素误差
    err[2] = zj_log - zi_log;  // log深度误差（相对深度）

    // 计算权重
    const float q = Q[block_id][k][0];  // 匹配质量
    const float ci = Cs[ix][ind_Xi][0];  // 点i的置信度
    const float cj = Cs[jx][k][0];  // 点j的置信度
    const bool valid =
      valid_match_ind
      & (q > Q_thresh)
      & (ci > C_thresh)
      & (cj > C_thresh)
      & valid_u & valid_v & valid_z;  // 检查图像和深度有效性
    
    // 使用置信度加权
    const float conf_weight = q;
    
    // 计算sqrt(weight)
    const float sqrt_w_pixel = valid ? sigma_pixel_inv * sqrtf(conf_weight) : 0;
    const float sqrt_w_depth = valid ? sigma_depth_inv * sqrtf(conf_weight) : 0;

    // 应用Huber鲁棒权重
    w[0] = huber(sqrt_w_pixel * err[0]);
    w[1] = huber(sqrt_w_pixel * err[1]);
    w[2] = huber(sqrt_w_depth * err[2]);
    
    // 将sigma权重加回
    const float w_const_pixel = sqrt_w_pixel * sqrt_w_pixel;
    const float w_const_depth = sqrt_w_depth * sqrt_w_depth;
    w[0] *= w_const_pixel;
    w[1] *= w_const_pixel;
    w[2] *= w_const_depth;

    // 计算Jacobian矩阵
    // 投影函数的导数：d(u,v)/d(X,Y,Z)

    // u坐标的Jacobian（像素u对3D点的导数）
    Ji[0] = fx * zj_inv;  // du/dX = fx/Z
    Ji[1] = 0.0;  // du/dY = 0
    Ji[2] = -fx * x_div_z * zj_inv;  // du/dZ = -fx*X/Z^2
    Ji[3] = -fx * x_div_z * y_div_z;  // du/dωx（旋转）
    Ji[4] = fx * (1 + x_div_z*x_div_z);  // du/dωy
    Ji[5] = -fx * y_div_z;  // du/dωz
    Ji[6] = 0.0;  // du/ds（缩放，投影后与缩放无关）

    // 转换到全局坐标系，得到对位姿i的Jacobian
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];  // Ji = -Jj（负号因为是对Ti求导）


    // 累加Hessian矩阵：H += J^T * w * J（上三角存储）
    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[0] * Jx[n] * Jx[m];  // 权重w[0]对应u方向误差
        l++;
      }
    }

    // 累加梯度向量：g += J^T * w * err
    for (int n=0; n<7; n++) {
      vi[n] += w[0] * err[0] * Ji[n];  // 位姿i的梯度
      vj[n] += w[0] * err[0] * Jj[n];  // 位姿j的梯度
    }

    // v坐标的Jacobian（像素v对3D点的导数）
    Ji[0] = 0.0;  // dv/dX = 0
    Ji[1] = fy * zj_inv;  // dv/dY = fy/Z
    Ji[2] = -fy * y_div_z * zj_inv;  // dv/dZ = -fy*Y/Z^2
    Ji[3] = -fy * (1 + y_div_z*y_div_z);  // dv/dωx（旋转）
    Ji[4] = fy * x_div_z * y_div_z;  // dv/dωy
    Ji[5] = fy * x_div_z;  // dv/dωz
    Ji[6] = 0.0;  // dv/ds

    // 转换到全局坐标系
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    // 累加Hessian（v方向）
    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[1] * Jx[n] * Jx[m];  // 权重w[1]对应v方向误差
        l++;
      }
    }

    // 累加梯度（v方向）
    for (int n=0; n<7; n++) {
      vi[n] += w[1] * err[1] * Ji[n];
      vj[n] += w[1] * err[1] * Jj[n];
    }

    // 深度坐标的Jacobian（log深度对3D点的导数）
    // d(log(Z))/d(X,Y,Z) = (0, 0, 1/Z)
    Ji[0] = 0.0;  // d(log Z)/dX = 0
    Ji[1] = 0.0;  // d(log Z)/dY = 0
    Ji[2] = zj_inv;  // d(log Z)/dZ = 1/Z
    Ji[3] = y_div_z;  // d(log Z)/dωx：Y/Z
    Ji[4] = -x_div_z;  // d(log Z)/dωy：-X/Z
    Ji[5] = 0.0;  // d(log Z)/dωz
    Ji[6] = 1.0;  // d(log Z)/ds：log深度随缩放线性变化

    // 转换到全局坐标系
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    // 累加Hessian（深度方向）
    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[2] * Jx[n] * Jx[m];  // 权重w[2]对应深度误差
        l++;
      }
    }

    // 累加梯度（深度方向）
    for (int n=0; n<7; n++) {
      vi[n] += w[2] * err[2] * Ji[n];
      vj[n] += w[2] * err[2] * Jj[n];
    }

  }  // 结束点循环
 
  __syncthreads();
 
  // 使用block归约将所有线程的结果汇总
  __shared__ float sdata[THREADS];
  // 归约梯度向量
  for (int n=0; n<7; n++) {
    // 归约位姿i的梯度
    sdata[threadIdx.x] = vi[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[0][block_id][n] = sdata[0];  // 线程0写回结果
    }
 
    __syncthreads();
 
    // 归约位姿j的梯度
    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[1][block_id][n] = sdata[0];
    }
 
  }
 
  // 归约Hessian矩阵（14x14，分为4个7x7块）
  l=0;
  for (int n=0; n<14; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);
 
      if (threadIdx.x == 0) {
        // 根据索引将Hessian写入对应的块
        if (n<7 && m<7) {
          // Hii块（左上）
          Hs[0][block_id][n][m] = sdata[0];
          Hs[0][block_id][m][n] = sdata[0];  // 对称矩阵
        }
        else if (n >=7 && m<7) {
          // Hij和Hji块（非对角）
          Hs[1][block_id][m][n-7] = sdata[0];
          Hs[2][block_id][n-7][m] = sdata[0];
        }
        else {
          // Hjj块（右下）
          Hs[3][block_id][n-7][m-7] = sdata[0];
          Hs[3][block_id][m-7][n-7] = sdata[0];  // 对称矩阵
        }
      }
 
      l++;
    }
  }
}

```

```python
# MASt3R-SLAM/mast3r_slam/backend/src/gn_kernels.cu L1546-L1638
/**
 * @brief 基于标定投影的Gauss-Newton优化（CUDA实现）
 * @param Twc 相机位姿
 * @param Xs 3D点坐标
 * @param Cs 置信度
 * @param K 相机内参矩阵
 * @param ii 边的起始节点索引
 * @param jj 边的终止节点索引
 * @param idx_ii2jj 匹配索引
 * @param valid_match 匹配有效性
 * @param Q 匹配质量
 * @param height 图像高度
 * @param width 图像宽度
 * @param pixel_border 像素边界
 * @param z_eps 深度阈值
 * @param sigma_pixel 像素标准差
 * @param sigma_depth 深度标准差
 * @param C_thresh 置信度阈值
 * @param Q_thresh 质量阈值
 * @param max_iter 最大迭代次数
 * @param delta_thresh 收敛阈值
 * @return 最后一次迭代的增量
 * 
 * 使用已知相机内参优化位姿，适用于已标定相机的SLAM系统
 */
std::vector<torch::Tensor> gauss_newton_calib_cuda(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor K,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const int height, const int width,
  const int pixel_border,
  const float z_eps,
  const float sigma_pixel, const float sigma_depth,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh)
{
  auto opts = Twc.options();  // 获取tensor选项
  const int num_edges = ii.size(0);  // 边数量
  const int num_poses = Xs.size(0);  // 位姿数量
  const int n = Xs.size(1);  // 点数量

  const int num_fix = 1;  // 固定位姿数量（通常固定第一帧作为参考）

  // 设置索引映射
  // 获取边中涉及的所有唯一关键帧
  torch::Tensor unique_kf_idx = get_unique_kf_idx(ii, jj);
  
  // 为边构建创建索引（pin=0，包含所有位姿）
  std::vector<torch::Tensor> inds = create_inds(unique_kf_idx, 0, ii, jj);
  torch::Tensor ii_edge = inds[0];  // 重新索引后的起始节点
  torch::Tensor jj_edge = inds[1];  // 重新索引后的终止节点
  
  // 为线性系统索引创建索引（pin=num_fix，固定前num_fix个位姿）
  // 这样可以将全局索引映射到优化变量的局部索引
  std::vector<torch::Tensor> inds_opt = create_inds(unique_kf_idx, num_fix, ii, jj);
  torch::Tensor ii_opt = inds_opt[0];  // 优化用的起始节点索引
  torch::Tensor jj_opt = inds_opt[1];  // 优化用的终止节点索引

  const int pose_dim = 7;  // Sim(3)位姿的维度（3平移+4旋转）

  // 初始化缓冲区
  // Hs存储4个7x7块矩阵：[Hii, Hij, Hji, Hjj]，每条边一组
  torch::Tensor Hs = torch::zeros({4, num_edges, pose_dim, pose_dim}, opts);
  // gs存储2个梯度向量：[gi, gj]，每条边一组
  torch::Tensor gs = torch::zeros({2, num_edges, pose_dim}, opts);

  // 用于调试的输出变量
  torch::Tensor dx;  // 位姿增量

  torch::Tensor delta_norm;  // 增量范数，用于判断收敛

  // Gauss-Newton迭代主循环
  for (int itr=0; itr<max_iter; itr++) {

    // 步骤1：调用CUDA kernel计算Hessian矩阵和梯度向量
    // 每条边启动一个block，每个block有THREADS个线程
    calib_proj_kernel<<<num_edges, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),  // 当前位姿估计
      Xs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   // 3D点云
      Cs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   // 置信度
      K.packed_accessor32<float,2,torch::RestrictPtrTraits>(),    // 相机内参
      ii_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),     // 边的起始节点
      jj_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),     // 边的终止节点
      idx_ii2jj.packed_accessor32<long,2,torch::RestrictPtrTraits>(),   // 点匹配索引
      valid_match.packed_accessor32<bool,3,torch::RestrictPtrTraits>(), // 匹配有效性
      Q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),          // 匹配质量
      Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),         // 输出：Hessian
      gs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),         // 输出：梯度
      height, width, pixel_border, z_eps, sigma_pixel, sigma_depth, C_thresh, Q_thresh
    );


    // 步骤2：构建稀疏线性系统 H*dx = -g
    SparseBlock A(num_poses - num_fix, pose_dim);  // 创建稀疏块矩阵

    // 将各条边的Hessian块填充到大的稀疏矩阵中
    // 需要指定每个块在大矩阵中的位置(ii, jj)
    A.update_lhs(Hs.reshape({-1, pose_dim, pose_dim}),  // 将4个块展平
        torch::cat({ii_opt, ii_opt, jj_opt, jj_opt}),   // 行索引：[ii, ii, jj, jj]
        torch::cat({ii_opt, jj_opt, ii_opt, jj_opt}));  // 列索引：[ii, jj, ii, jj]

    // 将各条边的梯度向量累加到大的梯度向量中
    A.update_rhs(gs.reshape({-1, pose_dim}),  // 将2个梯度展平
        torch::cat({ii_opt, jj_opt}));  // 节点索引：[ii, jj]

    // 步骤3：求解线性系统 H*dx = -g
    // 注意：这里加负号是因为我们要沿着梯度下降方向（-∇f）
    dx = -A.solve();

    
    // 步骤4：在Sim(3)流形上应用增量更新
    // Pose_new = exp(dx) * Pose_old
    pose_retr_kernel<<<1, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),  // 位姿（输入输出）
      dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),   // 增量
      num_fix);  // 跳过前num_fix个固定位姿

    // 步骤5：检查终止条件
    // 计算增量的L2范数：||dx||
    // 需要明确指定第二个参数，否则会有函数重载歧义
    delta_norm = torch::linalg::linalg_norm(dx, std::optional<c10::Scalar>(), {}, false, {});
    if (delta_norm.item<float>() < delta_thresh) {
      break;  // 如果增量足够小，认为已收敛，提前终止迭代
    }
        

  }  // 结束迭代循环

  return {dx};  // 返回最后一次迭代的增量（用于调试和分析）
}
```

