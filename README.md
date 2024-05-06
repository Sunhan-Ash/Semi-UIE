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

You can download the pretrained models on [BaiduPan](https://pan.baidu.com/s/1RaOnqsu5ssqtb_VCLirVxw? ) (stxh)

and datasets on  [BaiduPan](https://pan.baidu.com/s/1xrJOg0JfgiDJJGYML5G2rg?)(5pvq)

Because the consent of the LACC author was not obtained, this paper cannot open source it. However, good results can be obtained by removing this module. We provide this version of the pre-trained model.

The final file path should be the same as the following:

```
┬─ Semi-UIE
    ├─ model
    │   ├─ ckpt
  	│ 	│ 	└─ best_in_NR.pth
    │   └─ log
    |
	├─ data
    	├─ UIEBD
    	│   ├─ Labeled
    	│   │   ├─ GT
    	│   │   │   └─ ... (image filename)
    	|	|	├─ input
    	│   │   │   └─ ... (image filename)
   		│   │   └─ LA
    	│   │   │   └─ ... (image filename)
    	│   ├─ unlabeled
    	│   │   ├─ condidate
    	│   │   │   └─ ... (image filename)
    	|	|	├─ input
    	│   │   │   └─ ... (image filename)
   		│   │   └─ LA
    	│   │   │   └─ ... (image filename)
    	│   └─ val
    	│   │   ├─ GT
    	│   │   │   └─ ... (image filename)
    	|	|	├─ input
    	│   │   │   └─ ... (image filename)
   		│   │   └─ LA
    	│   │   │   └─ ... (image filename)
    	└─ ... (dataset name)
```



## Train

```
python train.py
```

## Test

```
python test.py
```



The code of this paper has not been sorted out yet. The team will sort out the code and publish it soon.
=======
