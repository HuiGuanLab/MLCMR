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

我们已经在项目文件的 `VisualSearch` 里准备好了所需的文本文件

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
VisualSearch/msrvtt10K/
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
# 使用 VATEX 训练 平行多语言 MLCMR 以验证中文性能 
./do_all.sh vatex i3d_kinetics parallel human_label zh $ROOTPATH
# 可以通过修改训练完毕后产生的do_testxxx.py的target_language参数为en，直接验证对应的英语性能

# 使用 VATEX 训练 平行多语言 MLCMR 以验证英文性能 
./do_all.sh vatex i3d_kinetics parallel human_label en $ROOTPATH

```

#### 预期表现

参考论文中做出的实验，VATEX上平行多语言场景下的MLCMR预期性能如下：

<table>
    <tr>
        <th rowspan='2'>Dateset</th><th colspan='5'>Text-to-Video Retrieval</th> <th colspan='5'>Video-to-Text Retrieval</th>  <th rowspan='2'>SumR</th>
    </tr>
    <tr>
        <th> R@1 </th> <th> R@5 </th> <th> R@10 </th> <th> MedR </th> <th>	mAP </th> <th> R@1 </th> <th> R@5 </th> <th> R@10 </th> <th> MedR </th> <th>	mAP </th>
    </tr>
    <tr>  
    	<td>Parllel_VATEX_Chinese</td>
		<td>36.3</td><td>71.3</td><td>81.2</td><td>2.0</td><td>51.69</td> 
    	<td>49.9</td><td>79.8</td><td>88.2</td><td>2.0</td><td>39.26</td> 
    	<td>406.6</td> 
    </tr>
    <tr>  
    	<td>Parllel_VATEX_English</td>
		<td>38.2</td><td>74.8</td><td>83.9</td><td>/</td><td>/</td> 
    	<td>50.3</td><td>78.0</td><td>86.3</td><td>/</td><td>/</td> 
    	<td>411.4</td> 
    </tr>
</table>


### 伪平行多语言场景

#### 模型训练与评估

运行以下脚本来训练和评估伪平行多语言场景下“MLCMR”网络。

```shell
ROOTPATH=$HOME/VisualSearch

conda activate mlcmr

# 使用 VATEX 训练 伪平行多语言 MLCMR 以验证中文性能 
./do_all.sh vatex i3d_kinetics parallel translate zh $ROOTPATH
# 请注意，这里不能通过修改对应的do_testxxx.py文件验证对应的英文性能，因为两者的训练数据是不同的

# 使用 VATEX 训练 伪平行多语言 MLCMR 以验证英文性能 
./do_all.sh vatex i3d_kinetics parallel translate en $ROOTPATH
```


#### 预期表现

参考论文中做出的实验，VATEX上伪平行多语言场景下的MLCMR预期性能如下：

<table>
    <tr>
        <th rowspan='2'>Dataset</th><th colspan='5'>Text-to-Video Retrieval</th> <th colspan='5'>Video-to-Text Retrieval</th>  <th rowspan='2'>SumR</th>
    </tr>
    <tr>
        <th> R@1 </th> <th> R@5 </th> <th> R@10 </th> <th> MedR </th> <th>	mAP </th> <th> R@1 </th> <th> R@5 </th> <th> R@10 </th> <th> MedR </th> <th>	mAP </th>
    </tr>
    <tr> 
    	<td>Parllel_VATEX_Translate_Chinese</td>
		<td>33.1</td><td>67.1</td><td>77.1</td><td>3.0</td><td>48.18</td> 
    	<td>46.7</td><td>76.6</td><td>85.9</td><td>2.0</td><td>35.40</td> 
    	<td>386.5</td> 
    </tr>
    <tr>  
    	<td>Parllel_VATEX_Translate_English</td>
		<td>38.3</td><td>74.0</td><td>82.9</td><td>/</td><td>/</td> 
    	<td>50.2</td><td>78.5</td><td>86.7</td><td>/</td><td>/</td> 
    	<td>410.7</td> 
    </tr>
</table>

### 不平行多语言场景

#### 模型训练与评估

运行以下脚本来训练和评估不平行多语言场景下“MLCMR”网络。

```shell
ROOTPATH=$HOME/VisualSearch

conda activate mlcmr

# 使用 VATEX 训练 不平行多语言 MLCMR 以验证中文性能 
./do_all.sh vatex i3d_kinetics unparallel human_label zh $ROOTPATH
# 可以通过修改训练完毕后产生的do_testxxx.py的target_language参数为en，直接验证对应的英语性能

# 使用 VATEX 训练 不平行多语言 MLCMR 以验证英文性能 
./do_all.sh vatex i3d_kinetics unparallel human_label en $ROOTPATH
```


#### 预期表现

参考论文中做出的实验，VATEX上不平行多语言场景下的MLCMR预期性能如下：

<table>
    <tr>
        <th rowspan='2'>Dataset</th><th colspan='3'>Text-to-Video Retrieval</th> <th colspan='3'>Video-to-Text Retrieval</th>  <th rowspan='2'>SumR</th>
    </tr>
    <tr>
        <th> R@1 </th> <th> R@5 </th> <th> R@10 </th> <th> R@1 </th> <th> R@5 </th> <th> R@10 </th> 
    </tr>
    <tr> 
    	<td>Unparllel_VATEX_Chinese</td>
		<td>31.6</td><td>64.7</td><td>76.0</td>
    	<td>44.9</td><td>75.5</td><td>84.5</td>
    	<td>377.1</td> 
    </tr>
    <tr>  
    	<td>Unparllel_VATEX_English</td>
		<td>32.8</td><td>66.7</td><td>77.4</td>
    	<td>44.5</td><td>74.4</td><td>84.3</td>
    	<td>380.1</td> 
    </tr>
</table>



## 使用MSRVTT训练MLCMR

由于MSRVTT不具备多语言特性，因此仅验证伪平行多语言场景下的性能

### 伪平行多语言场景

#### 模型训练与评估

运行以下脚本来训练和评估“MLCMR”网络。

```shell
ROOTPATH=$HOME/VisualSearch

conda activate mlcmr

# 例子:
# 使用 MSRVTT 训练 伪平行多语言 MLCMR 以验证中文性能 
./do_all.sh msrvtt10k resnext101-resnet152 parallel translate zh $ROOTPATH
# 请注意，这里不能通过修改对应的do_testxxx.py文件验证对应的英文性能，因为两者的训练数据是不同的

# 使用 MSRVTT 训练 伪平行多语言 MLCMR 以验证英文性能 
./do_all.sh msrvtt10k resnext101-resnet152 parallel translate en $ROOTPATH

```

#### 预期表现

参考论文中做出的实验，MSRVTT上伪平行多语言场景下的MLCMR预期性能如下：

<table>
    <tr>
        <th rowspan='2'>Dataset</th><th colspan='5'>Text-to-Video Retrieval</th> <th colspan='5'>Video-to-Text Retrieval</th>  <th rowspan='2'>SumR</th>
    </tr>
    <tr>
        <th> R@1 </th> <th> R@5 </th> <th> R@10 </th> <th> MedR </th> <th>	mAP </th> <th> R@1 </th> <th> R@5 </th> <th> R@10 </th> <th> MedR </th> <th>	mAP </th>
    </tr>
    <tr> 
    	<td>Parllel_MSRVTT_Translate_Chinese</td>
		<td>30.5</td><td>55.1</td><td>67.3</td><td>4.0</td><td>42.34</td> 
    	<td>30.3</td><td>56.9</td><td>68.1</td><td>4.0</td><td>42.64</td> 
    	<td>308.2</td> 
    </tr>
    <tr>  
    	<td>Parllel_MSRVTT_Translate_English</td>
		<td>23.7</td><td>49.4</td><td>61.6</td><td>6.0</td><td>35.79</td> 
    	<td>23.0</td><td>50.1</td><td>62.1</td><td>5.0</td><td>36.07</td> 
    	<td>269.9</td> 
    </tr>
</table>
