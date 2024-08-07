# YourGPT
![封面图片](https://github.com/Rainbowkv/YourGPT/blob/main/images/Demo.gif)

&emsp;&emsp;欢迎来到YourGPT，这是一个为大语言模型学习者设计的项目，通过实践帮助您从零开始使用PyTorch构建自己的GPT模型。本项目的目标是完成大语言模型预训练阶段，您将会完整地进行大模型的构建、训练、预测和评估过程。

## 目录
- [项目介绍](#项目介绍)
- [特色](#特色)
- [环境搭建](#环境搭建)
   - [克隆项目](#1-克隆项目)
   - [安装依赖](#2-安装依赖)
   - [预训练模型下载(可选)](#3-预训练模型下载可选)
- [快速开始](#快速开始)
   - [训练自己的gpt](#1-训练自己的gpt)
   - [体验预训练模型推理](#2-体验预训练模型推理)
   - [计算模型参数](#3-计算模型参数)
   - [ddp单机多卡训练](#4-ddp单机多卡训练)
- [贡献](#贡献)
- [许可证](#许可证)
- [致谢](#致谢)

## 项目介绍

&emsp;&emsp;YourGPT项目基于莎士比亚的作品集作为训练数据，从零使用pytorch写出GPT的核心结构，您将学会如何构建和预训练一个GPT语言模型。

&emsp;&emsp;项目的主要特色是一个包含Transformer解码器结构的仅200+行文件[transformer_4d.py](https://github.com/Rainbowkv/YourGPT/blob/main/transformer_4d.py)，其中不仅完整包含模型的构建，还包含训练代码。这意味着，完成[环境搭建](#环境搭建)后，只需要通过`python transformer_4d.py`运行这一个单独的文件，您就可以完成模型的训练和保存，无需依赖项目中的其他文件。

---
*项目结构*
```.
├─ bigram_demo     # 二元模型
├─ checkpoint      # 模型参数
├─ data            # 数据集
├─ images          # 图片
├─ models          # 模型类
├─ outcome         # 预测结果
├─ tokenizer       # 词编码
└─ utils           # 小工具
```

## 特色

- **完整的Transformer解码器实现：** 项目中包含一个完整的Transformer解码器结构实现，使得您可以深入理解这一现代NLP模型的核心。
- **独立文件设计：** 模型构建和训练需要的代码均在一个文件中，简化了学习过程，便于您快速上手。
- **模块化的设计：** 项目同样有模块化的设计，models/目录下有不同参数量的模型类，供您在[train.py](https://github.com/Rainbowkv/YourGPT/blob/main/train.py)、[estimate.py](https://github.com/Rainbowkv/YourGPT/blob/main/estimate.py)、[terminal_demo.py](https://github.com/Rainbowkv/YourGPT/blob/main/terminal_demo.py)中直接使用，您也可以修改这些模型的结构。
- **手动计算参数量：** [快速开始：3.计算模型参数](#caculate_num_params)会带您计算自己所设计模型的参数量，查看模型的结构细节。
- **实践导向：** 通过实际操作构建和训练模型，加深对大语言模型和Transformer架构的理解。

## 环境搭建

### 1. **克隆项目：**
   打开终端，运行以下命令将项目代码克隆到本地：

   `git clone https://github.com/Rainbowkv/YourGPT.git`

### 2. **安装依赖：**
   在项目目录下运行以下命令安装所需的第三方库：

   `cd YourGPT/`

   `pip install -r requirements.txt`

### 3. **预训练模型下载（可选）：**
   您需要通过git lfs工具来拉取本项目的checkpoints目录，模型类与参数文件对应关系如下：
   | 模型类 | 参数文件 | 参数量 |
   |--------|---------|-------|
   | models.UltimateModel | checkpoint/2024-05-12-23-23-08-params-10873409.pth | 0.01B |
   | models.HugeModel | checkpoint/2024-05-12-05-16-50-params-42783809.pth | 0.04B |

## 快速开始

### 1. **训练自己的GPT：**

   这里直接使用项目特色文件[transformer_4d.py](https://github.com/Rainbowkv/YourGPT/blob/main/transformer_4d.py)演示，文件头部区域，调整模型的超参数：
   
   | 模型参数 | 默认值 |
   |----------|-------|
   | n_embd | 384 |
   | block_size | 256 |
   | dropout | 0.2 |
   | n_blocks | 2 |
   | num_heads | 4 |
   |---|---|
   | 参数量 | 3691073 |

   ![transformers_4d_1.png](https://github.com/Rainbowkv/YourGPT/blob/main/images/transformers_4d_1.png)

   在您设置好参数后，直接执行下面的训练命令（推荐这样做）：

   `python transformer_4d.py`

   ![train_demo.png](https://github.com/Rainbowkv/YourGPT/blob/main/images/train_demo.png)
   
   
   如果您需要知道模型的参数量以防止OOM，修改[utils/num_params_caculate.py](https://github.com/Rainbowkv/YourGPT/blob/main/utils/num_params_caculate.py)文件相应参数，运行`python utils/num_params_caculate.py`即可进行计算。

<a id="try_predic"></a>

### 2. **体验预训练模型推理：**

   需要通过lfs下载checkpoints目录，运行以下命令体验本项目预训练的GPT(大约0.01B参数量)模型（脚本依赖文件checkpoint/2024-05-12-23-23-08-params-10873409.pth的存在）：

   `python terminal_demo.py`

   此脚本优先使用GPU推理，显存占用0.7GB，GPU不可用时自动切换回CPU。

<a id="caculate_num_params"></a>

### 3. **计算模型参数：**

   [penetrateModel.py](https://github.com/Rainbowkv/YourGPT/blob/main/penetrateModel.py)可以用来观察模型的结构，结合[手算模型参数量.txt](https://github.com/Rainbowkv/YourGPT/blob/main/手算模型参数量.txt)这个文件，您可以对模型的细节更加清晰。

### 4. **DDP单机多卡训练：**

   linux下执行命令示例：
   ```
   CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.run \
   --nproc_per_node=4 \
   --nnodes=1 \
   --node_rank=0 \
   --master_addr=localhost \
   --master_port=12345 \
   DDP_use_run.py
   ```

   windows下执行命令示例：
   ```
   set CUDA_VISIBLE_DEVICES=0,1,2,3 & ^
   python -m torch.distributed.run ^
   --nproc_per_node=1 ^
   --nnodes=1 ^
   --node_rank=0 ^
   --master_addr=localhost ^
   --master_port=12345 ^
   DDP_use_run.py
   ```

## 贡献

&emsp;&emsp;欢迎您任何形式的贡献，无论是通过发起Pull Request来修正错误，还是提出新的特性和想法。如果有任何问题，也欢迎通过Issues提出。

## 许可证

&emsp;&emsp;本项目采用MIT许可证，详细信息请参阅[LICENSE](https://github.com/Rainbowkv/YourGPT/blob/main/LICENSE)文件。

## 致谢

&emsp;&emsp;本项目受到了[nanoGPT](https://github.com/karpathy/nanoGPT)项目的启发，在其基础上进行了扩展与改进。对Andrej Karpathy及其贡献者们表示深深的感激和敬意。