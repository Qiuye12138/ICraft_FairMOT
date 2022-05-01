# `Icraft`版`FairMOT`

## 一、背景

#### 问题1：

截止到`V1.1`版本，`Icraft`暂时只原生支持`ReLU`和`LeakyReLU`两种激活函数，而`FairMOT`使用了`SiLU`激活函数。

可行的解决方法有以下几种：

1. 等待`IcraftV1.2`上线。该版本新增了对若干中激活函数的原生支持，包括`SiLU`。
2. 使用`CustomOp`软算子。`Icraft`支持自定义软算子，但由于使用`CPU`计算，且激活函数出现频繁，导致数据传输时间过长。

3. 使用`CustomOp`硬算子。`Icraft`支持自定义硬算子，但开发硬件`RTL`周期长、成本高。

4. 修改模型，使用`LeakyReLU`替换`SiLU`作为激活函数，需要重新训练。

本工程提供方案四的参考代码。

#### 问题2：

由于`Icraft`不支持`DCN`模块，所以本工程将使用`backbone`为`YoloV5`的模型。

`FairMOT`的`backbone`使用的`YoloV5`过于古老，使用切片式的`Focus`作为输入层。

截止到`V1.1`版本，`Icraft`暂时不支持`Python`中的切片操作。

本工程将其改为了卷积层，与最新的`YoloV5`保持一致。

#### 问题3：

`FairMot`未提供模型导出代码。

本工程提供了模型导出功能，以便`Icraft`编译。

#### 问题4：

`Icraft`提供了硬算子加速检测网络阈值筛选的功能，但截止到`V1.1`版本，最多支持一个检测层含有`3`个特征图。

`FairMot`共输出四个特征图。

本工程将两个特征图拼接，实现只输出三个特征图。

#### 问题5：

截止到`V1.1`版本，`Icraft`不支持最后一层是拼接算子

本工程将`问题4`中添加的拼接算子后又添加了一个卷积层，该层输出恒等于输入

#### 问题6：

截止到`V1.1`版本，`Icraft`的阈值筛选加速硬算子要求阈值必须在每个检测层的第`1`个特征图内。

本工程导出模型时调整计算顺序，先计算含`hm`的卷积层。

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

使用以下代码即可开始训练

```bash
python train.py mot --exp_id mot17_half_yolov5s --data_cfg '../src/lib/cfg/mot17_half.json' --lr 5e-4 --batch_size 8 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --gpus 0
```

使用以下代码即可恢复最近一次训练

```bash
python train.py mot --exp_id mot17_half_yolov5s --data_cfg '../src/lib/cfg/mot17_half.json' --lr 5e-4 --batch_size 8 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --gpus 0 --resume
```

### 2.4、确认推理正确

使用以下代码即可推理

```bash
python demo.py mot --test_mot17 True --load_model ../exp/mot/mot17_half_yolov5s/model_last.pth --conf_thres 0.4 --arch yolo --gpus 0
```

### 2.5、保存模型

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



