# 使用SSD对X-ray数据集进行图片检测

## 数据集文件目录：
```
SSD
|__ data
	|__ sixray
		|__ core_3000
		    |_ Annotation
		    |_ Image
		|__ coreless_3000
		    |_ Annotation
		    |_ Image
		|__ sixray_eval
		    |_ Annotation
		    |_ Image
		|__ train.txt
		|__ test.txt
```
## 中断后继续训练：
```
python train.py --resume 'filename.pth' --start_iter num_of_newiter
```
## 训练结果：
使用SSD512，90%数据集进行训练，mAP最高达到0.9181。40000步时mAP即达到0.9119。

![](https://github.com/wangx26/SSD_SIXray_homework/blob/master/weights/result.png)