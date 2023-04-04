# ColorNet
本工程主要实现自监督学习下使模型学习到图像之间密集的对应性关系，进而实现半监督式的视频目标分割。

本工程基于[Dense Unsupervised Learning for Video Segmentation](https://github.com/visinf/dense-ulearn-vos)的官方实现代码。


运行环境为pytorch-1.11.


* * *
### 数据
配置数据集路径，本方法训练集为YouTube-VOS和MS COCO、测试集为DAVIS 2017 val。
假设数据存放目录为$DATAROOT，默认为./data，存放方式如下：

```
YouTube-VOS : 
$DATAROOT/JPEGImages/*/*.jpg

DAVIS 2017 : 
$DATAROOT/davis2017/JPEGImages/480p/*/*.jpg
$DATAROOT/davis2017/Annotations/480p/*/*.png

MS COCO : 
$DATAROOT/COCO/train2017/*.jpg
```
数据也可以存放于其他路径，但必须将filelists拷贝至存放数据的路径下，并且修改./configs/ytvos.yaml的DATASET->ROOT为存放路径。

* * *

### 训练
```
python trainColorNetDUL_benchmark.py --exp 实验ID --train_config 配置文件路径
```
如果使用COCO数据集训练，在最后追加--coco。
```
python trainColorNetDUL_benchmark.py --exp 实验ID --train_config 配置文件路径 --coco
```
#### 配置文件
配置文件默认为./data/configs/train_config_default_M.json，如果设置的配置文件不存在，训练将直接退出。

```
{
    "exp":"benchmark_M",                    实验ID
    "first_kernal_size":3,                  第一层卷积的卷积核尺寸
    "color_mode":"LAB",                     色彩空间模式，可选RGB和LAB
    "is_dul":true,                          是否引入DUL方法
    "is_aug":true,                          是否使用常规数据增强方法
    "is_shadow":true,                       是否生成阴影
    "is_liquid":true,                       是否使用随机图像扭曲
    "is_edge":true,                         是否使用边缘权重图
    "is_neg":true,                          是否使用负样本
    "space_consistency":true,               是否使用空间一致性
    "is_labclip": true,                     是否使用Lab空间数值截断
    "ohem_range":[0,1.0],                   Loss计算时选取的百分比
    "model_size":"M",                       模型规格
    "TEMP":25,                              图像重构时求特征距离的缩放系数┏
    "is_cos_lr":true,                       学习率是否使用warmup和余弦衰减
    "video_len": 5                          切片大小
}
```

* * *
### 测试
#### 预测
特征类型分为color、cls、merge三种，color：局部特征；cls：全局特征；merge：融合特征。预测后的结果将保存在./output目录下。

**注：只有在配置文件中is_dul为true时才能使用全局特征和融合特征。**
```
python infer_vos_dul.py   --cfg configs/ytvos.yaml --exp 0001 --run final --infer-list filelists/val_davis2017_test --mask-output-dir ./output --seed 0 --set TEST.KEY 特征类型 --resume 模型路径
```

#### 评估
使用[davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation)官方提供的代码进行预测。
```
python evaluation_method.py --task semi-supervised --davis_path DAVIS数据集存放路径 --results_path 预测结果路径
```


| 结构 | 参数量 | FLOPs(256x256) | J&F(局部\全局\融合)      | 下载 |
| --- | --- | --- |--------------------| --- |
| Model-L(仅局部) | 11.5M | 12.3G | 70.9 / - / -        | [download](https://1drv.ms/u/s!AjYPLlUeVYc7nOo--DrUio6S5Pojyw?e=MNaS2i) |
| Model-M | 14.6M | 15.2G | 70.0 / 70.6 / 72.1  | [download](https://1drv.ms/u/s!AjYPLlUeVYc7nOpA_gjifo3YeHS53Q?e=P62mCF) |
| Model-S | 4.0M | 4.3G | 68.1 / 69.7 / 70.9 | [download](https://1drv.ms/u/s!AjYPLlUeVYc7nOo_3s_7J7ZmAJTeKQ?e=5nr3dq) |