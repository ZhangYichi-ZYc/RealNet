# RealNet

**💡 这是论文 "RealNet: A Feature Selection Network with Realistic Synthetic Anomaly for Anomaly Detection (CVPR 2024)" 的官方实现 [[arxiv]](https://arxiv.org/abs/2403.05897)**

RealNet 是一个简单而高效的框架，包含三个关键创新：首先，我们提出了强度可控的扩散异常合成（SDAS），这是一种基于扩散过程的合成策略，能够生成具有不同异常强度的样本，模拟真实异常样本的分布。其次，我们开发了异常感知特征选择（AFS），这是一种选择具有代表性和区分性的预训练特征子集的方法，以提高异常检测性能，同时控制计算成本。第三，我们引入了重构残差选择（RRS），这是一种自适应选择区分性残差的策略，用于跨多个粒度层次全面识别异常区域。

<div align=center><img width="850" src="assets/pipeline.JPG"/></div>

### 🏆 异常合成
我们使用扩散模型进行异常合成，为四个数据集（MVTec-AD、MPDD、BTAD 和 VisA）的异常检测模型训练提供了 `36万` 张异常图像。[[下载]](https://drive.google.com/drive/folders/12B1SMmdsVc6UPDoLP6cctn4YvQfk8cS6?usp=drive_link)
<div align=center><img width="800" src="assets/anomaly_synthesis.jpg"/></div>

### 🏆 扩散模型检查点
扩散模型 [[下载]](https://drive.google.com/drive/folders/1kQCuAc0Tlf-XZosJLgKleUYvPJ33zyBK?usp=drive_link) 和指导分类器（可选）[[下载]](https://drive.google.com/drive/folders/1x-TSOXVYSQgvub1de5m8R_FF4m8pD1ow?usp=drive_link)，在 MVTec-AD、MPDD、BTAD 和 VisA 数据集上训练。

### 🏆 基于特征重构的方法

|      | 图像 AUROC     | 像素 AUROC     |
| :----------: | :----------: | :----------: |
| MVTec-AD | 99.6 | 99.0 |
| MPDD | 96.3 | 98.2 |
| BTAD | 96.1 | 97.9 |
| VisA | 97.8 | 98.8 |

## 🔧 安装

要运行实验，首先克隆存储库并安装 `requirements.txt`。

```
$ git clone https://github.com/cnulab/RealNet.git
$ cd RealNet
$ pip install -r requirements.txt
```
### 数据准备
下载以下数据集：
- **MVTec-AD [[官方]](https://www.mvtec.com/company/research/datasets/mvtec-ad/) 或 [[我们的链接]](https://drive.google.com/file/d/1jo8kYau8U-Z2OzAgb4wjqnCecKK-6N11/view?usp=drive_link)**
- **MPDD [[官方]](https://github.com/stepanje/mpdd) 或 [[我们的链接]](https://drive.google.com/file/d/1GVHC2lCt3QUBkVbMCGQ47jwZ3NkQSUgk/view?usp=drive_link)**
- **BTAD [[官方]](http://avires.dimi.uniud.it/papers/btad/btad.zip) 或 [[我们的链接]](https://drive.google.com/file/d/1_J2b3yEr4VUqRZEL6V0wK8r8hkHaV77c/view?usp=drive_link)**
- **VisA [[官方]](https://github.com/amazon-science/spot-diff) 或 [[我们的链接]](https://drive.google.com/file/d/1xl46seYQkjC2B3mLxBhfaXSJTydOAfoo/view?usp=drive_link)**

**对于 `VisA` 数据集，我们进行了格式处理以确保一致性。强烈建议您从我们的链接下载。**

如果使用 `DTD`（可选）数据集进行异常合成，请下载：

- **DTD [[官方]](https://www.robots.ox.ac.uk/~vgg/data/dtd/) 或 [[我们的链接]](https://drive.google.com/file/d/1omufc0m67sPmthvralN8K40n8TWl4z8e/view?usp=drive_link)**

将它们解压到 `data` 文件夹中。请参阅 [data/README](data/README.md)。

## 🚀 实验
### 🌞 训练扩散模型

我们加载在 ImageNet 上预训练的扩散模型权重，如下所示：

- **预训练扩散模型 [[官方]](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt) 或 [[我们的链接]](https://drive.google.com/file/d/1OyiUJOWBdrFUumiO5TBn0-APYj5DpblD/view?usp=drive_link)**

我们使用指导分类器来提高图像质量（可选）：

- **预训练指导分类器 [[官方]](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt) 或 [[我们的链接]](https://drive.google.com/file/d/1ZdzR3rXPzyzC67kHJmOQL38P9zbBbZja/view?usp=drive_link)**

下载它们并放置在 `pretrain` 文件夹中。

在 MVTec-AD 数据集上训练扩散模型：
```
$ python -m torch.distributed.launch --nproc_per_node=4  train_diffusion.py --dataset MVTec-AD
```

在 MVTec-AD 数据集上训练指导分类器：
```
$ python -m torch.distributed.launch --nproc_per_node=2  train_classifier.py --dataset MVTec-AD
```

我们使用 `4*A40 GPUs`，训练扩散模型需要 48 小时，使用 `2*RTX3090 GPUs`，训练指导分类器需要 3 小时。

我们提供以下检查点：
- **MVTec-AD: [[扩散模型]](https://drive.google.com/file/d/1cl2w5eCFrmbOEWlcqPakignI4CHZdm_d/view?usp=drive_link), [[指导分类器]](https://drive.google.com/file/d/1-geYTTmeDD9yZzEtstbgjwlWFfgNhld8/view?usp=drive_link)**
- **MPDD: [[扩散模型]](https://drive.google.com/file/d/1GYUdxObhgu-kWIwBf6gumsMOY3IZDF6o/view?usp=drive_link), [[指导分类器]](https://drive.google.com/file/d/1pLEOk4D5o80Yzq7RDeSSBv2HaF76Fxf5/view?usp=drive_link)**
- **BTAD: [[扩散模型]](https://drive.google.com/file/d/1IYktXaXIOCv3otIVTmvj2Ck5DOeLHXiN/view?usp=drive_link), [[指导分类器]](https://drive.google.com/file/d/1ASS70U72VOVcAqaN4AK-EZlgEGqfj1p3/view?usp=drive_link)**
- **VisA: [[扩散模型]](https://drive.google.com/file/d/1FzgW5xRz-TtPBkbMBbSoAJDq5gkO6Yd_/view?usp=drive_link), [[指导分类器]](https://drive.google.com/file/d/15bdOwBdO_bd74p2rcIKt9pTMDzxgjoJW/view?usp=drive_link)**

下载它们到 `experiments` 文件夹中。请参阅 [experiments/README](experiments/README.md)。

### 🌞 强度可控的扩散异常合成

使用 `1*RTX3090 GPU` 采样异常图像：
```
$ python -m torch.distributed.launch --nproc_per_node=1  sample.py --dataset MVTec-AD
```

我们为每个类别提供了 `10k` 张分辨率为 `256*256` 的采样异常图像，可以通过以下链接下载：
- **MVTec-AD [[下载]](https://drive.google.com/file/d/1Rs6XRb6v3WdSidiFsMHMK9tALqaSMY3u/view?usp=drive_link)**
- **MPDD [[下载]](https://drive.google.com/file/d/1SdRNyoaG0FrBp79UrdMT0L15jigZMveW/view?usp=drive_link)**
- **BTAD [[下载]](https://drive.google.com/file/d/1r4HlORHzgyz9nHr2QTvU2odPZytco-Y2/view?usp=drive_link)**
- **VisA [[下载]](https://drive.google.com/file/d/1Dq75NOUWIUdt_DV6JiVhwwYAKR7EeVJC/view?usp=drive_link)**

### 🌞 训练 RealNet

使用 `1*RTX3090 GPU` 训练 RealNet：
```
$ python -m torch.distributed.launch --nproc_per_node=1  train_realnet.py --dataset MVTec-AD --class_name bottle
```

[realnet.yaml](experiments/MVTec-AD/realnet.yaml) 提供了训练期间的各种配置。

更多命令可以在 [run.sh](run.sh) 中找到。

### 🌞 评估 RealNet

计算图像 AUROC、像素 AUROC 和 PRO，并生成异常定位的定性结果：
```
$ python  evaluation_realnet.py --dataset MVTec-AD --class_name bottle
```
<div align=center><img width="850" src="assets/results.jpg"/></div>

## ✈️ 其他

我们还为每个类别提供了一些生成的 `正常` 图像（在论文中将异常强度设置为0），可以通过以下链接下载：
- **MVTec-AD [[下载]](https://drive.google.com/file/d/1e4A4cGJkCYD4KCD0GHutSleaJqHM5fNb/view?usp=drive_link)**
- **BTAD [[下载]](https://drive.google.com/file/d/1MXRqcY0yfbsOY59ZJ4p4rlmmDo4qS6dK/view?usp=drive_link)**
- **VisA [[下载]](https://drive.google.com/file/d/10r1moi4LW1DrlujY-1-aVYjVRcFJUSO_/view?usp=drive_link)**

该存储库的附加文件目录：
- **[[Google Drive]](https://drive.google.com/drive/folders/1DwAR6jS7x4PcXP8ygDnd5cEsHP0BsKdU?usp=drive_link)**
- **[[百度云]](https://pan.baidu.com/s/1Aqc1TwTMXTemlR3-TjyuaA?pwd=6789) (密码 6789)**

代码参考：**[UniAD](https://github.com/zhiyuanyou/UniAD)** 和 **[BeatGans](https://github.com/openai/guided-diffusion)**。

## 🔗 引用

如果这项工作对您有帮助，请引用：
```
@inproceedings{zhang2024realnet,
      title={RealNet: A Feature Selection Network with Realistic Synthetic Anomaly for Anomaly Detection}, 
      author={Ximiao Zhang, Min Xu, and Xiuzhuang Zhou},
      year={2024},
      eprint={2403.05897},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
