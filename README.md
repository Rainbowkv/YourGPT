# YourGPT
![封面图片](https://github.com/Rainbowkv/YourGPT/blob/main/images/Demo.png)

&emsp;&emsp;欢迎来到YourGPT，这是一个为大语言模型学习者设计的项目，通过实践帮助您从零开始使用PyTorch构建自己的迷你GPT模型。本项目的目标是提供一个简单的入门级示例，使得对大语言模型感兴趣的人群能够快速上手并了解模型的基本结构和训练过程。

## 目录
- [项目介绍](#项目介绍)
- [特色](#特色)
- [环境搭建](#环境搭建)
- [快速开始](#快速开始)
- [贡献](#贡献)
- [许可证](#许可证)

## 项目介绍

&emsp;&emsp;YourGPT项目基于莎士比亚的作品集作为训练数据，从零使用pytorch写出GPT的核心结构，您将学会如何构建和预训练一个GPT语言模型。
&emsp;&emsp;项目的特色是一个包含Transformer解码器结构的仅200+行文件[transformer_4d.py](https://github.com/Rainbowkv/YourGPT/blob/main/transformer_4d.py)，其中不仅完整包含模型的构建，还包含训练代码。这意味着，完成[环境搭建](#环境搭建)后，只需要通过`python transformer_4d.py`运行这一个单独的文件，您就可以完成模型的训练和保存，无需依赖项目中的其他文件。

## 特色

- **完整的Transformer实现：** 项目中包含一个完整的Transformer结构实现，使得您可以深入理解这一现代NLP模型的核心。
- **独立文件设计：** 模型构建和训练需要的代码均在一个文件中，简化了学习过程，便于您快速上手。
- **模块化的设计：** 项目同样有模块化的设计，models/目录下有不同参数量的模型类，供您在[train.py](https://github.com/Rainbowkv/YourGPT/blob/main/train.py)、[estimate.py](https://github.com/Rainbowkv/YourGPT/blob/main/estimate.py)、[terminal_demo.py](https://github.com/Rainbowkv/YourGPT/blob/main/terminal_demo.py)中直接使用，您也可以修改这些模型的结构。
- **实践导向：** 通过实际操作构建和训练模型，加深对大语言模型和Transformer架构的理解。

## 环境搭建

1. **克隆项目：**
   打开终端，运行以下命令将项目代码克隆到本地：

   `git clone https://github.com/Rainbowkv/YourGPT.git`

2. **安装依赖：**
   在项目目录下运行以下命令安装所需的第三方库：

   `cd YourGPT/`

   `pip install -r requirements.txt`

## 快速开始
1. **体验预训练模型推理：**
   一切准备就绪后，运行以下命令体验本项目预训练的GPT(大约0.01B参数量)模型：

   `python terminal_demo.py`

   此脚本优先使用GPU推理，显存占用0.7GB，GPU不可用时自动切换回CPU。

2. **训练自己的GPT**
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

   在考虑GPU显存的情况下，您可能需要提前预知模型的参数以防止OOM，修改[utils/num_params_caculate.py](https://github.com/Rainbowkv/YourGPT/blob/main/utils/num_params_caculate.py)文件相应参数，运行`python utils/num_params_caculate.py`即可进行计算。

   ---

   设置好参数后，执行`python transformer_4d.py`开启训练：
   ![train_demo.png](https://github.com/Rainbowkv/YourGPT/blob/main/images/train_demo.png)

## 贡献

&emsp;&emsp;欢迎您任何形式的贡献，无论是通过发起Pull Request来修正错误，还是提出新的特性和想法。如果您有任何问题，也欢迎通过Issues提出。

## 许可证

&emsp;&emsp;本项目采用MIT许可证，详细信息请参阅[LICENSE](https://github.com/Rainbowkv/YourGPT/blob/main/LICENSE)文件。