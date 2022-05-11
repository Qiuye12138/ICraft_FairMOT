# `Icraft`版`FairMOT`

## 一、背景

#### 问题1：

截止到`V1.1`版本，`Icraft`暂时只原生支持`ReLU`和`LeakyReLU`两种激活函数，而`FairMOT`使用了`SiLU`激活函数

可行的解决方法有以下几种：

1. 等待`IcraftV2.0`上线。该版本新增了对若干中激活函数的原生支持，包括`SiLU`
2. 使用`CustomOp`软算子。`Icraft`支持自定义软算子，但由于使用`CPU`计算，且激活函数出现频繁，导致数据传输时间过长

4. 修改模型，使用`LeakyReLU`替换`SiLU`作为激活函数，需要重新训练

本工程提供方案三的参考代码

#### 问题2：

由于`Icraft`不支持`DCN`模块，所以本工程将使用`backbone`为`YoloV5`的模型

`FairMOT`的`backbone`使用的`YoloV5`过于古老，使用切片式的`Focus`作为输入层

截止到`V1.1`版本，`Icraft`暂时不支持`Python`中的切片操作

本工程将其改为了卷积层，与最新的`YoloV5`保持一致

#### 问题3：

`FairMot`未提供模型导出代码

本工程提供了模型导出功能，以便`Icraft`编译

导出模型时删除掉最后一层卷积后的操作，它们将在后处理代码中实现

#### 问题4：

`Icraft`提供了硬算子加速检测网络阈值筛选的功能，但截止到`V1.1`版本，最多支持一个检测层含有`3`个特征图

`FairMot`共输出四个特征图

本工程将`hm`和`wh`两个特征图拼接，实现只输出三个特征图

#### 问题5：

截止到`V1.1`版本，`Icraft`不支持最后一层是拼接算子

本工程将`问题4`中添加的拼接算子后又添加了一个卷积层，该层输出恒等于输入

#### 问题6：

截止到`V1.1`版本，`Icraft`的阈值筛选加速硬算子要求阈值必须在每个检测层的第`1`个特征图内

本工程导出模型时调整计算顺序，先计算含`hm`的卷积层

模型导出和`Icraft`加载模型，均以计算发生的顺序为准



## 二、使用步骤
### 2.1、安装

在`Windows`下直接安装`cython_bbox`会失败，可以用以下方法替换：

```powershell
python -m pip install git+https://github.com/yanfengliu/cython_bbox.git
```

### 2.2、准备数据集

下载[MOT17](https://motchallenge.net/data/MOT17/)数据集

修改`src/lib/cfg/mot17_half.json`内的`root`

修改`src/gen_labels_16.py`内的`seq_root`和`label_root`

使用以下脚本生成`MOT17`的标签

```powershell
python gen_labels_16.py
```

> `MOT17`与`MOT16`脚本相同

### 2.3、训练

下载预训练模型`fairmot_lite.pth`

使用以下代码即可开始训练

```bash
python train.py mot --exp_id mot17_half_yolov5s --data_cfg '../src/lib/cfg/mot17_half.json' --lr 5e-4 --batch_size 8 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --load_model '../fairmot_lite.pth' --gpus 0
```

使用以下代码即可恢复最近一次训练

```bash
python train.py mot --exp_id mot17_half_yolov5s --data_cfg '../src/lib/cfg/mot17_half.json' --lr 5e-4 --batch_size 8 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --load_model '../fairmot_lite.pth' --gpus 0 --resume
```

### 2.4、确认推理正确

使用以下代码即可推理

```bash
python demo.py mot --test_mot17 True --load_model ../exp/mot/mot17_half_yolov5s/model_last.pth --conf_thres 0.4 --arch 'yolo' --gpus 0

```

### 2.5、测试精度

使用以下代码即可测试精度，需要将`--data_dir`修改为实际数据集路径

```bash
python track_half.py mot --exp_id mot17_half_yolov5s --data_cfg '../src/lib/cfg/mot17_half.json' --data_dir D:\Dataset\MOT17 --load_model ../exp/mot/mot17_half_yolov5s/model_last.pth --conf_thres 0.4 --val_mot17 True --arch 'yolo'
```

本次重训练精度为：

```bash
              IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT ML   FP    FN IDs    FM  MOTA  MOTP IDt IDa IDm
MOT17-02-SDP 44.6% 59.0% 35.9% 52.4% 86.0%  53  14  25 14  846  4720 171   280 42.1% 0.228  86  55  11
MOT17-04-SDP 86.2% 90.1% 82.7% 88.6% 96.6%  69  53  16  0  745  2750  59   294 85.3% 0.194  18  31   3
MOT17-05-SDP 62.2% 74.7% 53.2% 67.6% 94.9%  71  22  34 15  122  1087  54   112 62.4% 0.197  36  28  18
MOT17-09-SDP 67.1% 78.9% 58.4% 73.1% 98.8%  22  11   9  2   25   777  27    46 71.3% 0.193  19  11   4
MOT17-10-SDP 76.3% 90.4% 65.9% 68.9% 94.4%  36  13  21  2  244  1852  39   233 64.1% 0.242  11  18   3
MOT17-11-SDP 68.5% 78.5% 60.7% 72.1% 93.2%  44  17  14 13  237  1264  46    65 65.8% 0.152  18  21   4
MOT17-13-SDP 71.9% 81.6% 64.2% 70.9% 90.0%  44  17  21  6  249   922  52   151 61.3% 0.239  16  31   4
OVERALL      73.6% 82.9% 66.2% 75.2% 94.3% 339 147 140 52 2468 13372 448  1181 69.9% 0.202 204 195  47
```

### 2.6、保存模型

使用以下代码即可保存模型

```bash
python demo.py mot --test_mot17 True --load_model ../exp/mot/mot17_half_yolov5s/model_last.pth --conf_thres 0.4 --arch yolo --gpus -1 --export True
```



## 三、修改部分

`src\gen_labels_16.py`：替换`seq_root`、 `label_root`

`src\lib\cfg\mot17_half.json`：替换`root`

`src\lib\models\common.py`：将`SiLU`替换为`LeakyReLU`，删除`DCN`相关

`src\lib\models\model.py`：删除`DCN`相关

`src\lib\models\networks\config\yolov5s.yaml`：将`Focus`替换为卷积

`src\lib\models\yolo.py`：将`SiLU`替换为`LeakyReLU`；训练、推理、导出时执行不同代码

`src\lib\opts.py`：添加`export`命令行参数

`src\lib\tracker\multitracker.py`：推理、导出时执行不同代码



