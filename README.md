# Semi-UIE

## Preparation

### Install

We test the code on PyTorch 1.13.1 + CUDA 11.6

1. Create a new conda environment

   ```
   conda create -n semi-UIE python=3.7
   conda activate semi-UIE
   ```

   

2. Install dependencies

```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
pip install -r requirements.txt
```

### Download

You can download the pretrained models on [BaiduPan](https://pan.baidu.com/s/1008_cJKSY0EGkOnRv1osBA?pwd=7mrv 
 ) (7mrv )

and datasets on  [BaiduPan](https://pan.baidu.com/s/1xrJOg0JfgiDJJGYML5G2rg?)(5pvq)
and the result of paper on [BaiduPan](链接: https://pan.baidu.com/s/13Yh37pVlY7j1QZHPqk8Gwg?)(j6au)

The final file path should be the same as the following:

```
┬─ Semi-UIE
    ├─ model
    │   ├─ ckpt
    │ 	│	├─ best_in_evaluation.pth
    │ 	│	├─ best_in_psnr.pth
    │ 	│	└─ best_in_NR.pth
    │   └─ log
    |
	├─ data
	    	├─ UIEBD
	    	│   ├─ Labeled
	    	│   │   ├─ GT
	    	│   │   │   └─ ... (image filename)
	    	|   |   ├─ input
	    	│   │   │   └─ ... (image filename)
	   	│   │   └─ LA
	    	│   │   │   └─ ... (image filename)
	    	│   ├─ unlabeled
	    	│   │   ├─ condidate
	    	│   │   │   └─ ... (image filename)
	    	|   |   ├─ input
	    	│   │   │   └─ ... (image filename)
	   	│   │   └─ LA
	    	│   │   │   └─ ... (image filename)
	    	│   └─ val
	    	│   │   ├─ GT
	    	│   │   │   └─ ... (image filename)
	    	|   |   ├─ input
	    	│   │   │   └─ ... (image filename)
	   	│   │   └─ LA
	    	│   │   │   └─ ... (image filename)
	    	└─ ... (dataset name)
```

### Before starting training or testing, please confirm that the environment has been configured and that the paths to all files are correct.

## Train

```
python train.py
```

## Test

```
python test.py
```
## Evaluate

The various indicators mentioned in the paper are MUSIQ, URanker, UIQM, and UCIQE. For MUSIQ and URanker, please confirm the file path and then use the 'evaluation.py' file to calculate. For UIQM and UCIQE, please use matlab to call the 'UIQM' folder. Compute.m is used for calculation. compute.m will generate a csv file. To avoid trouble, please calculate MUSIQ and URanker first.There may be slight differences in the calculation results, but they are generally similar.

If you only need the results of this paper, please use [BaiduPan](链接: https://pan.baidu.com/s/13Yh37pVlY7j1QZHPqk8Gwg?)(j6au) to download the results

## Acknowledgement

The training code architecture is based on the [Semi-UIR](https://github.com/Huang-ShiRui/Semi-UIR?tab=readme-ov-file) and [MLLE](https://github.com/Li-Chongyi/MMLE_code) and thanks for their work. We also thank for the following repositories: [IQA-Pytorch](https://github.com/chaofengc/IQA-PyTorch),  [AECR-Net](https://github.com/GlassyWu/AECR-Net/blob/main/models/CR.py), [UIEB](https://li-chongyi.github.io/proj_benchmark.html), [EUVP](https://irvlab.cs.umn.edu/resources/euvp-dataset), [UWCNN](https://li-chongyi.github.io/proj_underwater_image_synthesis.html),  [RUIE](https://github.com/dlut-dimt/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark),  [Ucolor](https://github.com/Li-Chongyi/Ucolor), [CWR](https://github.com/JunlinHan/CWR),[WWPE](https://github.com/Li-Chongyi/WWPF_code),[SGUIE](https://github.com/trentqq/SGUIE-Net_Simple),[URanker](https://github.com/Li-Chongyi/li-chongyi.github.io/tree/master/URanker_files),[RAUNE](https://github.com/fansuregrin/RAUNE-Net) and so on.

