# HH-Former for Wheat Leaf Disease Classification

> 官方 PyTorch 实现 | 论文处于投刊阶段，标题：《Hierarchical Hybrid Attention Transformer for Wheat Leaf Disease Classification》  
> 提出分层混合注意力Transformer（HH-Former）模型，实现小麦叶片常见病害与健康叶片的高精度分类，辅助农业病害快速诊断。


## 1. 研究背景与模型定位
小麦作为全球核心粮食作物，其叶片病害（如白粉病、条锈病、叶锈病）易导致光合效率骤降、产量损失达10%-30%，传统人工检测存在耗时久、主观性强、田间覆盖范围有限的问题。  

本文提出**分层混合注意力Transformer（HH-Former）**，通过创新的注意力机制融合与分层金字塔结构，解决小麦叶片病害“局部病灶特征模糊（如早期微小锈孢子堆）、全局纹理建模效率低、相似病害区分难”的核心问题，最终在自建小麦叶片病害数据集（WPLDD）上实现99.93%的分类准确率，为小麦病害田间自动化诊断提供高效、可靠的技术方案。


## 2. HH-Former 核心创新点
1. **三层注意力机制融合**：  
   - 浅层：采用窗口注意力（Window Attention），精准捕捉小麦叶片局部病变特征（如白粉病的白色霉层、条锈病的黄褐色条状孢子堆、叶锈病的红褐色圆形病斑）；  
   - 中层：应用线性注意力（Linear Attention），在降低计算量的同时，高效建模叶片全局纹理分布（如健康小麦叶片的平行叶脉与病害区域的纹理差异）；  
   - 深层：使用标准注意力（Standard Attention），对前两层特征进行精细融合，强化不同病害类别的特征区分度（如区分形态相似的条锈病与叶锈病）。  

2. **分层金字塔特征结构**：  
   通过逐步下采样特征图，构建多尺度特征表示，适配小麦叶片不同大小的病害区域（从早期毫米级微小病斑到后期厘米级大面积发病区域）。  

3. **效率与精度平衡**：  
   相较于传统Transformer模型，计算量降低20%以上，在保证高精度的同时提升推理速度，可适配农业田间边缘检测设备（如便携式诊断终端、无人机载检测模块）。


## 3. 实验数据集：WPLDD
### 3.1 数据集概况
本研究基于**自建小麦叶片常见病害检测数据集（Wheat Leaf Disease Detection Dataset, WPLDD）**，数据集已随项目上传至仓库的 `WPLDD/` 文件夹，无需额外下载。  

| 数据集名称 | 包含类别                | 图像总数 | 图像分辨率 | 数据分布（训练:测试） |
|------------|-------------------------|----------|------------|-----------------------|
| WPLDD      | 白粉病、条锈病、叶锈病 + 健康叶片 | 详见数据集说明文件 | 统一 resize 至 256×256 | 3:1（通过代码自动划分） |

### 3.2 数据集结构
仓库中 `WPLDD/` 文件夹组织如下，图像均采集于小麦主产区田间（涵盖不同生育期、光照与拍摄角度），可直接用于模型训练/测试：
```
WPLDD/
├── 白粉病/       # 小麦白粉病叶片图像（病斑特征：白色粉末状霉层，多分布于叶片正面）
├── 叶锈病/       # 小麦叶锈病叶片图像（病斑特征：红褐色圆形或椭圆形孢子堆，随机分布）
└── 健康叶片/     # 健康小麦叶片图像（特征：叶脉清晰、叶色浓绿均匀，无异常斑点）
```


## 4. 实验环境配置
### 4.1 依赖安装
推荐使用Anaconda创建虚拟环境，确保依赖版本匹配（避免兼容性问题，尤其适配PyTorch 2.7.1）：
```bash
# 1. 创建并激活虚拟环境
conda create -n hh-former-wheat python=3.10
conda activate hh-former-wheat

# 2. 安装PyTorch与TorchVision（需适配CUDA版本，示例为CUDA 12.1；CPU用户可替换为cpu版本）
pip3 install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装其他依赖库（数据处理、可视化、模型工具等）
pip install numpy~=2.0.2 matplotlib~=3.9.4 opencv-python~=4.12.0.88
pip install pandas~=2.3.1 pillow~=11.2.1 torchviz~=0.0.3 xlwt~=1.3.0
pip install tqdm~=4.67.1 timm~=1.0.15
```

### 4.2 硬件要求
- **GPU**：推荐 NVIDIA GPU（显存≥8GB，如RTX 3060/4060，支持CUDA 11.8+），训练50轮耗时约2-3小时，显存占用峰值≤6GB；  
- **CPU**：支持推理测试（单张图像推理耗时约0.5-1秒），但训练耗时显著增加（约20-25小时），不推荐用于完整训练流程。


## 5. 实验结果
### 5.1 核心指标对比（WPLDD数据集）
HH-Former 与主流深度学习模型在小麦叶片病害分类任务上的性能对比如下，模型在精度、计算效率上均表现更优，尤其对相似病害（条锈病/叶锈病）的区分能力显著提升：

| 模型               | 分类准确率（Accuracy） | 计算量（FLOPs） | 参数量（M） |
|--------------------|------------------------|-----------------|-------------|
| Vision Transformer（ViT-Base） | 98.83%              | 17.6G           | 86.8        |
| EfficientNet-B0    | 97.93%              | 0.39G           | 5.3         |
| Swin Transformer（Tiny） | 97.09%           | 4.5G            | 28.2        |
| **HH-Former（本文）**    | **99.93%**         | **14.1G**       | **16.5**    |

> 注：1. 计算量降低20%以上是相对于ViT-Base模型的对比结果；2. 准确率为5次重复实验的平均值（排除随机种子影响），标准差≤0.05%，结果稳定性高；3. EfficientNet-B0对小麦条锈病与叶锈病的混淆率达3.5%，而HH-Former混淆率仅0.02%。


## 6. 代码使用说明
### 6.1 模型训练
运行 `train.py` 脚本启动训练，支持通过参数调整训练配置，示例命令（适配WPLDD数据集）：
```bash
python train.py \
  --data_dir ./WPLDD \
  --epochs 50 \
  --batch_size 16 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --save_dir ./weights \
  --device cuda:0 \
  --log_interval 10  # 每10个batch打印一次训练日志
```

#### 关键参数说明：
| 参数名          | 含义                                  | 默认值       |
|-----------------|---------------------------------------|-------------|
| `--data_dir`    | WPLDD数据集根目录路径                 | `./WPLDD`   |
| `--epochs`      | 训练轮数                              | 50          |
| `--batch_size`  | 批次大小（根据GPU显存调整，8/16/32）  | 16          |
| `--lr`          | 初始学习率                            | 1e-4        |
| `--save_dir`    | 训练权重保存目录                      | `./weights` |
| `--device`      | 训练设备（`cuda:0` 或 `cpu`）         | `cuda:0`    |

#### 训练输出：
- 训练过程中，模型会自动保存**验证集准确率最高**的权重至 `--save_dir` 目录，文件名为 `best-model.pth`；  
- 训练日志（损失值、准确率）会实时打印，并保存至 `train_log.txt`。

### 6.2 模型预测
使用训练好的权重进行单张小麦叶片图像预测，运行 `predict.py` 脚本，示例命令：
```bash
python predict.py \
  --image_path ./examples/wheat_stripe_rust.jpg \  # 输入图像路径（示例图像存于examples/）
  --weight_path ./weights/best_hh_former.pth \  # 预训练权重路径
  --device cuda:0
```

#### 预测输出示例：
```
输入图像路径：./examples/wheat_stripe_rust.jpg
预测类别：小麦条锈病
置信度：0.9978
```

### 6.3 预训练权重
提供基于 WPLDD 数据集训练完成的最优权重，可直接用于预测或微调。除随项目仓库附带的权重外，也可通过百度网盘获取完整权重文件：

百度网盘分享：
链接: https://pan.baidu.com/s/1OG8uLUL0_OQL-BaDWNEhmA
提取码: 4ycx
（复制这段内容后打开百度网盘手机 App，操作更方便）
本地权重文件：weights/best_hh_former.pth（若仓库内权重存在大小限制，可通过上述网盘链接获取完整版本）；
适用场景：仅针对小麦叶片的 “白粉病、条锈病、叶锈病、健康叶片” 四类分类，若需扩展其他小麦病害（如叶枯病），建议基于此权重微调（冻结浅层注意力模块，仅训练分类头与深层特征融合层，可减少 50% 以上训练数据量）。


## 7. 项目文件结构
```
hh-former-for-wheat-leaf-disease/
├── WPLDD/                # 自建小麦叶片病害数据集（含白粉病、条锈病、叶锈病、健康叶片）
├── examples/             # 预测示例图像（如wheat_powdery_mildew.jpg、wheat_stripe_rust.jpg）
├── models/               # 模型定义文件夹
│   └── hh_former.py      # HH-Former核心代码（含分层注意力、金字塔结构）
├── dataset/              # 数据处理文件夹
│   ├my_dataset.py      # 对图像进行预处理（按索引返回单张图像及其标签，供 DataLoader 批量加载数据）
│   ├── data_loader.py    # WPLDD数据集加载与预处理（自动划分训练/测试集+数据增强）
│   ├── wheat_gan.py   # 使用GAN生成部分病害图像数据集
├── train.py              # 模型训练脚本
├── predict.py            # 模型预测脚本
└── README.md             # 项目说明文档（本文档）
```


## 8. 已知问题与注意事项
1. **数据集适配**：当前模型与权重仅针对 `WPLDD/` 中的“斑枯病、白粉病、叶枯病、叶锈病、健康叶片”五类，若新增病害类别，需补充对应数据集并重新训练（建议每类样本量≥500张，确保模型泛化性）；  
2. **图像分辨率**：输入图像会自动resize至256×256，若原始图像分辨率过低（<128×128），可能导致早期微小病斑特征丢失，建议输入图像分辨率≥256×256；  
3. **CUDA版本问题**：若安装PyTorch时出现CUDA不兼容，可替换为CPU版本（需将所有脚本的`--device`改为`cpu`），但训练效率会大幅下降；  
4. **田间场景适配**：若用于实际田间检测，建议先通过 `dataset/data_loader.py` 中的数据增强模块扩充数据集，提升模型对田间复杂环境的适应能力。


## 9. 引用与联系方式
### 9.1 引用方式
论文处于投刊阶段，正式发表后将更新BibTeX引用格式，当前可临时引用：
```bibtex
@article{hh_former_wheat_disease,
  title={Hierarchical Hybrid Attention Transformer for Wheat Leaf Disease Classification},
  author={[作者姓名，待发表时补充]},
  journal={[期刊名称，待录用后补充]},
  year={2025},
  note={Manuscript submitted for publication}
}
```

### 9.2 联系方式
若遇到代码运行问题或学术交流需求，请联系：  
- 邮箱：changyibu@huuc.edu.cn  
- GitHub Issue：直接在本仓库提交Issue，会在1-3个工作日内回复。