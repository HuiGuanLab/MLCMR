# 多语言文本-视频跨模态检索的新基线模型

"多语言文本-视频跨模态检索的新基线模型"论文源代码

![framework](framework.png)

## 内容列表

- [环境](#环境)
- [数据准备](#数据准备)
- [使用VATEX训练MLCMR](#使用VATEX训练MLCMR)
  - [平行多语言场景](#平行多语言场景)
    - [模型训练与评估](#模型训练与评估)
    - [预期表现](#预期表现)
  - [伪平行多语言场景](#伪平行多语言场景)
    - [模型训练与评估](#模型训练与评估-1)
    - [预期表现](#预期表现-1)
  - [不平行多语言场景](#不平行多语言场景)
    - [模型训练与评估](#模型训练与评估-2)
    - [预期表现](#预期表现-2)
- [使用MSRVTT训练MLCMR](#使用MSRVTT训练MLCMR)
  - [伪平行多语言场景](#伪平行多语言场景)
    - [模型训练与评估](#模型训练与评估-3)
    - [预期表现](#预期表现-3)

## 环境

- CUDA 10.1
- Python 3.8
- PyTorch 1.5.1

我们使用Anaconda设置了一个支持PyTorch的深度学习工作区，请运行以下脚本以安装所需的程序包。

```shell
conda create --name mlcmr python=3.8
conda activate mlcmr
git clone https://github.com/HuiGuanLab/MLCMR.git
cd mlcmr
pip install -r requirements.txt
conda deactivate
```

## 数据准备

我们使用两种公开数据集: VATEX, MSR-VTT. 预训练提取的特征请放置在  `$HOME/VisualSearch/`.

我们已经在项目文件的 `VisualSearch` 里准备好了所有的训练文本文件

对应的视频特征可通过下方获取

| Dataset    | feature                                                      |
| ---------- | ------------------------------------------------------------ |
| VATEX      | [vatex-i3d.tar.gz, pwd:p3p0](https://pan.baidu.com/s/1lg23K93lVwgdYs5qnTuMFg?pwd=p3p0) | 
| MSR-VTT | [msrvtt10k-resnext101_resnet152.tar.gz, pwd:p3p0](https://pan.baidu.com/s/1lg23K93lVwgdYs5qnTuMFg?pwd=p3p0) |


```shell
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH

请组织这些文件成下面的形式:
# 下载VATEX数据[英语, 中文]
VisualSearch/VATEX/
	FeatureData/
		i3d_kinetics/
			feature.bin
			id.txt
			shape.txt
			video2frames.txt
	TextData/
		xx.txt

# 下载MSR-VTT数据[英语, 中文]
VisualSearch/msrvtt/
	FeatureData/
		resnext101-resnet152/
			feature.bin
			id.txt
			shape.txt
			video2frames.txt
	TextData/
		xx.txt

```

## 使用VATEX训练MLCMR

### 平行多语言场景

#### 模型训练与评估

运行以下脚本来训练和评估“MLCMR”网络。具体而言，它将训练“MLCMR”网络，并选择在验证集上表现最好的检查点作为最终模型。请注意，我们只在验证集上保存性能最好的检查点，以节省磁盘空间。

```shell
ROOTPATH=$HOME/VisualSearch

conda activate mlcmr

# 例子:
# 训练 平行多语言 MLCMR 以验证中文性能 
./do_all.sh vatex i3d_kinetics parallel human_label zh $ROOTPATH
# 可以通过修改训练完毕后产生的do_testxxx.py的target_language参数为en，直接验证对应的英语性能

# 训练 平行多语言 MLCMR 以验证英文性能 
./do_all.sh vatex i3d_kinetics parallel human_label en $ROOTPATH

```

#### 预期性能

| Type  | Text-to-Video Retrieval | Video-to-Text Retrieval | SumR |
| ----- | ----------------------- | ----------------------- | ---- |
| R@1   | R@5                     | R@10                    | MedR |
| en2cn | 30.8                    | 64.4                    | 74.6 |


### 伪平行多语言场景

#### 模型训练与评估

运行以下脚本来训练和评估伪平行多语言场景下“MLCMR”网络。

```shell
ROOTPATH=$HOME/VisualSearch

conda activate mlcmr

# 训练 伪平行多语言 MLCMR 以验证中文性能 
./do_all.sh vatex i3d_kinetics parallel translate zh $ROOTPATH

# 训练 伪平行多语言 MLCMR 以验证英文性能 
./do_all.sh vatex i3d_kinetics parallel translate en $ROOTPATH
```


#### 预期性能

| Type  | Text-to-Video Retrieval | Video-to-Text Retrieval | SumR |
| ----- | ----------------------- | ----------------------- | ---- |
| R@1   | R@5                     | R@10                    | MedR |
| en2cn | 28.9                    | 56.3                    | 67.3 |

### 不平行多语言场景

#### 模型训练与评估

运行以下脚本来训练和评估不平行多语言场景下“MLCMR”网络。

```shell
ROOTPATH=$HOME/VisualSearch

conda activate mlcmr

# 训练 不平行多语言 MLCMR 以验证中文性能 
./do_all.sh vatex i3d_kinetics unparallel human_label zh $ROOTPATH

# 训练 不平行多语言 MLCMR 以验证英文性能 
./do_all.sh vatex i3d_kinetics unparallel human_label en $ROOTPATH
```


#### 预期性能

| Type  | Text-to-Video Retrieval | Video-to-Text Retrieval | SumR |
| ----- | ----------------------- | ----------------------- | ---- |
| R@1   | R@5                     | R@10                    | MedR |
| en2cn | 28.9                    | 56.3                    | 67.3 |


## 使用MSRVTT训练MLCMR

由于MSRVTT不具备多语言特性，因此仅验证起伪平行多语言场景下的性能

### 伪平行多语言场景

#### 模型训练与评估

运行以下脚本来训练和评估“MLCMR”网络。

```shell
ROOTPATH=$HOME/VisualSearch

conda activate mlcmr

# 例子:
# 训练 伪平行多语言 MLCMR 以验证中文性能 
./do_all.sh msrvtt10k resnext101-resnet152 parallel translate zh $ROOTPATH

# 训练 伪平行多语言 MLCMR 以验证英文性能 
./do_all.sh msrvtt10k resnext101-resnet152 parallel translate en $ROOTPATH

```

#### 预期性能

| Type  | Text-to-Video Retrieval | Video-to-Text Retrieval | SumR |
| ----- | ----------------------- | ----------------------- | ---- |
| R@1   | R@5                     | R@10                    | MedR |
| en2cn | 28.9                    | 56.3                    | 67.3 |
