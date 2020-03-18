## Pruning_PJ3
Carlos Santillana and Glen Meyerowitz's pruning implementation.
**PASCAL VOC related code from https://github.com/xuzheyuan624/yolov3-pytorch**

### PASCAL VOC
**Code from https://github.com/xuzheyuan624/yolov3-pytorch**

Change to pascal_voc directory
```
cd pascal_voc
```
Create "weights" directory and download pretrained backbone weights for darknet53 to "weights" folder from https://drive.google.com/file/d/1zoGUFk9Tfoll0vCteSJRGa9I1yEAfU-a/view.
```
cd data
sh get_voc_dataset.sh
python voc_label.py
```
Change to "data" direcotry and download pascal voc dataset. Uncompress the tar file and you should find "VOCdevkit" under "data" directory. Meanwhile, check image path names in xx_val.txt and xx_train.txt to make sure training scripts can find them.
```
cd ..
```
Return to pascal_voc direcotry and run following commands below for training/evaluation. No need to load darknet53 backbone weights with "--load" argument, since it is automatically loaded every time you run the script. Use "--load" to optionally specify the path for your own checkpoint.
```
python3 main.py train --load PRETRAINED_PTH --name=voc --gpu False
python3 main.py eval --load PRETRAINED_PTH --name=voc --gpu False
```
Pruning funcions can be implemented in prune_utils.py.

**Notice** Evaluation can be very slow for not-well-trained model due to too many predicted bounding boxes. So train several epoches then evaluate.

**Notice** I have uploaded pretrained weights for yolov3 on PASCAL VOC on google drive https://drive.google.com/file/d/1PnhVkGkjiBalNK_gBNS0bw9SN39eLcXu/view?usp=sharing. It has been trained for 27 epoches and achieves  mAP(IoU=0.5) as 73.2, which can be used as a starting point.

**Notice** I have uploaded pretrained **pruned weights** for yolov3 on PASCAL VOC on google drive https://drive.google.com/open?id=1IJ4Zy_P9FxnoZwKPqsQaFBn5jhYJyvlP if you want to run eval on these weights.  The link will be taken down 3/27/20 as they take up a lot of memory.
