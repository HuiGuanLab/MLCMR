# 多语言文本-视频跨模态检索的新基线模型

多语言文本-视频跨模态检索的新基线模型论文源代码

![framework](framework.png)

## Table of Contents

- [Environments](#environments)
- [Required Data](#required-data)
- [MLCMR on VATEX](#MLCMR-on-VATEX)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [Evaluation using Provided Checkpoints](#Evaluation-using-Provided-Checkpoints)
  - [Expected Performance](#Expected-Performance)
- [MLCMR on MSRVTT](#NRCCR-on-MSRVTT10K-CN)
  - [Model Training and Evaluation](#model-training-and-evaluation-1)
  - [Evaluation using Provided Checkpoints](#Evaluation-using-Provided-Checkpoints-1)
  - [Expected Performance](#Expected-Performance-1)
- [Reference](#Reference)

## Environments

- CUDA 10.1
- Python 3.8
- PyTorch 1.5.1

We used Anaconda to setup a deep learning workspace that supports PyTorch. Run the following script to install the required packages.

```shell
conda create --name mlcmr python=3.8
conda activate mlcmr
git clone https://github.com/LiJiaBei-7/nrccr.git
cd nrccr
pip install -r requirements.txt
conda deactivate
```

## Required Data

We use TWO public datasets: VATEX, MSR-VTT. The extracted feature is placed  in `$HOME/VisualSearch/`.

For Multi-30K, we have provided translation version (from Google Translate) of Task1 and Task2, respectively.  [Task1: Applied to translation tasks. Task2: Applied to captioning tasks.].

In addition, we also provide MSCOCO dataset here, and corresponding performance below.  The validation and test set on Japanese from [STAIR Captions](https://stair-lab-cit.github.io/STAIR-captions-web/), and that on Chinese from [COCO-CN](https://github.com/li-xirong/coco-cn).

Training set:

	source(en) + translation(en2xx) + back-translation(en2xx2en)

Validation set and test set:

	target(xx) + translation(xx2en)

| Dataset    | feature                                                      | caption                                                      |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| VATEX      | [vatex-i3d.tar.gz, pwd:p3p0](https://pan.baidu.com/s/1lg23K93lVwgdYs5qnTuMFg?pwd=p3p0) | [vatex_caption, pwd:oy27](https://www.aliyundrive.com/s/xDrzCDNEHWP) |
| MSR-VTT-CN | [msrvtt10k-resnext101_resnet152.tar.gz, pwd:p3p0](https://pan.baidu.com/s/1lg23K93lVwgdYs5qnTuMFg?pwd=p3p0) | [cn_caption, pwd:oy27](https://www.aliyundrive.com/s/3sBNJqfTxcp) |
| Multi-30K  | [multi30k-resnet152.tar.gz, pwd:5khe](https://pan.baidu.com/s/1AzTN6rFyabirACVkVEVKCQ) | [multi30k_caption, pwd:oy27](https://www.aliyundrive.com/s/zGEbQAvqHGy) |
| MSCOCO     |                                                              | [mscoco_caption, pwd:13dc](https://www.aliyundrive.com/s/PxToUYryguz) |


```shell
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH

Organize these files like this:
# download the data of VATEX[English, Chinese]
VisualSearch/VATEX/
	FeatureData/
		i3d_kinetics/
			feature.bin
			id.txt
			shape.txt
			video2frames.txt
	TextData/
		xx.txt

# download the data of MSR-VTT-CN[English, Chinese]
VisualSearch/msrvttcn/
	FeatureData/
		resnext101-resnet152/
			feature.bin
			id.txt
			shape.txt
			video2frames.txt
	TextData/
		xx.txt

# download the data of Multi-30K[Englich, German, French, Czech]
# For Task2, the training set was translated from Flickr30K, which contains five captions per image, while for task1, each image corresponds to one caption.
# The validation and test set on French and Czech are same in both tasks.
VisualSearch/multi30k/
	FeatureData/
		train_id.txt
		val_id.txt
		test_id_2016.txt

	resnet_152[optional]/
		train-resnet_152-avgpool.npy
		val-resnet_152-avgpool.npy
		test_2016_flickr-resnet_152-avgpool.npy	
	TextData/
		xx.txt	
	flickr30k-images/
		xx.jpg

# download the data of MSCOCO[English, Chinese, Japanese]
VisualSearch/mscoco/
	FeatureData/
		train_id.txt
		ja_val_id.txt
		zh_val_id.txt
		ja_test_id.txt
		zh_test_id.txt
	TextData/
		xx.txt
	all_pics/
		xx.jpg
		
	image_ids.txt
```

## MLCMR on VATEX

### Model Training and Evaluation

Run the following script to train and evaluate `NRCCR` network. Specifically, it will train `NRCCR` network and select a checkpoint that performs best on the validation set as the final model. Notice that we only save the best-performing checkpoint on the validation set to save disk space.

```shell
ROOTPATH=$HOME/VisualSearch

conda activate nrccr_env

# To train the model on the MSR-VTT, which the feature is resnext-101_resnet152-13k 
# Template:
./do_all_vatex.sh $ROOTPATH <gpu-id>

# Example:
# Train NRCCR 
./do_all_vatex.sh $ROOTPATH 0
```

`<gpu-id>` is the index of the GPU where we train on.

### Evaluation using Provided Checkpoints

Download trained checkpoint on VATEX from Baidu pan ([url](https://pan.baidu.com/s/1QPPBZq_fN8D4tnf_dhfQKA),  pwd:ise6) and run the following script to evaluate it.

```shell
ROOTPATH=$HOME/VisualSearch/

tar zxf $ROOTPATH/<best_model>.pth.tar -C $ROOTPATH

./do_test_vatex.sh $ROOTPATH $MODELDIR <gpu-id>
# $MODELDIR is the path of checkpoints, $ROOTPATH/.../runs_0
```

### Expected Performance

| Type  | Text-to-Video Retrieval | Video-to-Text Retrieval | SumR |
| ----- | ----------------------- | ----------------------- | ---- |
| R@1   | R@5                     | R@10                    | MedR |
| en2cn | 30.8                    | 64.4                    | 74.6 |


## MLCMR on MSR-VTT

### Model Training and Evaluation

Run the following script to train and evaluate `NRCCR` network on MSR-VTT-CN.

```shell
ROOTPATH=$HOME/VisualSearch

conda activate nrccr_env

# To train the model on the VATEX
./do_all_msrvttcn.sh $ROOTPATH <gpu-id>
```

### Evaluation using Provided Checkpoints

Download trained checkpoint on MSR-VTT-CN from Baidu pan ([url](https://pan.baidu.com/s/1QPPBZq_fN8D4tnf_dhfQKA),  pwd:ise6) and run the following script to evaluate it.

```shell
ROOTPATH=$HOME/VisualSearch/

tar zxf $ROOTPATH/<best_model>.pth.tar -C $ROOTPATH

./do_test_msrvttcn.sh $ROOTPATH $MODELDIR <gpu-id>
# $MODELDIR is the path of checkpoints, $ROOTPATH/.../runs_0
```

### Expected Performance

| Type  | Text-to-Video Retrieval | Video-to-Text Retrieval | SumR |
| ----- | ----------------------- | ----------------------- | ---- |
| R@1   | R@5                     | R@10                    | MedR |
| en2cn | 28.9                    | 56.3                    | 67.3 |


## Reference

If you find the package useful, please consider citing our paper:

```
@inproceedings{wang2022cross,
  title={Cross-Lingual Cross-Modal Retrieval with Noise-Robust Learning},
  author={Yabing Wang and Jianfeng Dong and Tianxiang Liang and Minsong Zhang and Rui Cai and Xun Wang},
  journal={In Proceedings of the 30th ACM international conference on Multimedia},
  year={2022}
}
```