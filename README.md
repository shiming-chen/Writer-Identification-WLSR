# Semi-Supervised-Learning-for-Writer-Identification
AusAI 2018

# Overview 
- [Dependences](##dependences)
- [Line_Segmentation](##Line_Segmentation)
- [Feature_Extraction](##Feature_Extraction)
- [Contact](##Contact)

The codes of this repository are for papper "Semi-Supervised Feature Learning for Off-Line Writer Identification"

## Dependences 
- Matlab, Matconvnet, Opencv, NVIDIA GPU
- **(Note that I have included my Matconvnet in this repo, so you do not need to download it again. I has changed some codes comparing with the original version. For example, one of the difference is in `/matlab/+dagnn/@DagNN/initParams.m`. If one layer has params, I will not initialize it again, especially for pretrained model.)**

	You just need to uncomment and modify some lines in `compile.m` and run it in Matlab. Try it~
	(The code does not support cudnn 6.0. You may just turn off the Enablecudnn or try cudnn5.1)

	If you fail in compilation, you may refer to http://www.vlfeat.org/matconvnet/install/

## Line_Segmentation
At first, you segment the document to lines with statistical line segmentation. You can refer to guideline (https://github.com/KiM55/DLS-CNN/).

## Feature_Extraction

### Train
1. Make a dir called `data` by typing `mkdir ./data`.

2. Download [ResNet-50 model](http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat) pretrained on Imagenet. Put it in the `data` dir. 

3. Add your dataset path into `prepare_data.m` and run it. Make sure the code outputs the right image path.

4.  Run `train_id_net_res_market_wlsr.m` for training the proposed method.

### Test

Run `test/feature_extraction.m` to extract the features of images in the gallery and query set. They will store in a .mat file. Then you can use it to do evaluation.

### Evaluation
Run `evaluation/evaluation.m` for evaluation.

## Contact
If you run into any problems with this code, please submit a bug report on the Github site of the project. For another inquries pleace contact with me: gchenshiming@gmail.com or g_shmchen@163.com




