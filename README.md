[toc]

# MMDetection’s documentation

- 本仓库为mmdetection 2.25.0版本，其中`git remote`连接两个远端
  - `mmlab`原版代码仓
  - `origin`个人代码仓

## Get Started
- 环境需求：Linux, Windows and macOS. Python 3.6+, CUDA 9.2+ and PyTorch 1.5+.
```bash
# Prerequisites
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision -c pytorch
# Installation
pip install -U openmim
mim install mmcv-full
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# pip install mmdet # use mmdet as a dependency or third-party
# Verify
mim download mmdet --config yolov3_mobilenetv2_320_300e_coco --dest .
python demo/image_demo.py demo/demo.jpg yolov3_mobilenetv2_320_300e_coco.py yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth --device cpu --out-file result.jpg
```

- 其他事项
  - GeForce 30 series需要CUDA 11，cuda版本与GPU驱动版本[查看](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)
  - docker安装`docker build -t mmdetection docker/`
  - docker运行`docker run --gpus all -it -v {DATA_DIR}:/data mmdetection`
  - 不使用`mim`安装`mmcv`，详见[文档](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)

```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

- 预训练权重下载[链接](https://github.com/open-mmlab/mmcv/blob/master/mmcv/model_zoo/open_mmlab.json)，注意区分各模型的`img_norm_cfg`
  - `TorchVision`的ResNet50, ResNet101，`mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True`
  - `Pycls`的RegNetX，`mean=[103.530, 116.280, 123.675], std=[57.375, 57.12, 58.395], to_rgb=False`
  - `MSRA`的Caffe Type，`mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False`
  - `Caffe2`的ResNext101_32x8d，`mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False`

- 对比`Detectron2`，mmdet/detectron2

| Type         | Performance | Training(s/iter) | Inference(img/s) | memory  |
| ------------ | ----------- | ---------------- | ---------------- | ------- |
| Faster R-CNN | 38.0/37.9   | 0.216/0.210      | 22.2/25.6        | 3.8/3.0 |
| Mask R-CNN   | 38.8/38.6   | 0.265/0.261      | 19.6/22.5        | 3.9/3.4 |
| Retinanet    | 37.0/36.5   | 0.205/0.200      | 20.6/17.8        | 3.4/3.9 |

## Quick Run

### Inference with existing models demo

```python
# 推理
from mmdet.apis import init_detector, inference_detector
import mmcv
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
model.show_result(img, result)
model.show_result(img, result, out_file='result.jpg')
# video
video = mmcv.VideoReader('video.mp4')
for frame in video:
    result = inference_detector(model, frame)
    model.show_result(frame, result, wait_time=1)
# ===========================================================================
# 异步推理Asynchronous
import asyncio
import torch
from mmdet.apis import init_detector, async_inference_detector
from mmdet.utils.contextmanagers import concurrent
async def main():
    config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    device = 'cuda:0'
    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)
    streamqueue = asyncio.Queue()
    streamqueue_size = 3
    for _ in range(streamqueue_size):
        streamqueue.put_nowait(torch.cuda.Stream(device=device))
    img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
    async with concurrent(streamqueue):
        result = await async_inference_detector(model, img)
    model.show_result(img, result)
    model.show_result(img, result, out_file='result.jpg')
asyncio.run(main())
# ===========================================================================
# Batch Inference
data = dict(train=dict(...), val=dict(...), test=dict(samples_per_gpu=2, ...))
# or pass option
--cfg-options data.test.samples_per_gpu=2
```

### Test existing models on standard datasets

```bash
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] \
    [--eval ${EVAL_METRICS}] [--show]
# multi-gpu testing
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} \
    [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
# 不指定RESULT_FILE时不会保存文件
# EVAL_METRICS需被数据集支持，如bbox，segm，mAP，recall
# --show用于可视化，--show-dir用于保存可视化结果
./tools/dist_test.sh configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \
    checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
    8 \
    --out results.pkl \
    --eval bbox segm \
    --options "classwise=True"
```

### Train predefined models on standard datasets

- 学习率自动调整：默认学习率是适用于 8 GPU，2 sample per gpu，即batch size = 8 * 2 = 16，通过设置`auto_scale_lr.base_batch_size`调整，或传参`--auto-scale-lr`，依据[论文](https://arxiv.org/abs/1706.02677)
- 训练参数
  - `--work-dir ${WORK_DIR}`指定文件保存目录
  - `--resume-from ${CHECKPOINT_FILE}`从保存的权重恢复训练
  - `resume-from` and `load-from`：`resume`包括权重、训练状态等，而load只加载权重将从0开始训练

```bash
# Training on a single GPU
python tools/train.py ${CONFIG_FILE} [optional arguments]
# Training on multiple GPUs
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
# Single machine multi tasks need different ports (29500 by default)
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
# Train with multiple machines
# On the first machine:
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh tools/dist_train.sh $CONFIG $GPUS
# On the second machine:
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh tools/dist_train.sh $CONFIG $GPUS
```

###  Customized Datasets

- Three ways to support a new dataset
  - 转为`COCO format`，当前评估 `mask AP` 仅支持coco
  - 转为mmdetection的 `middle format`
  - 自定义实现
