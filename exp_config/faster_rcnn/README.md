[toc]

# Faster-RCNN

- BDD100K COCO mAP

|         | AP@.5:.95 | AP@.5 | AP@small | AP@medium | AP@large | AR@.5:.95 | AR@small | AR@medium | AR@large |
| ------- | --------- | ----- | -------- | --------- | -------- | --------- | -------- | --------- | -------- |
| r50_fpn | 0.156     | 0.322 | 0.079    | 0.200     | 0.239    | 0.277     | 0.155    | 0.334     | 0.396    |
|         |           |       |          |           |          |           |          |           |          |
|         |           |       |          |           |          |           |          |           |          |



- BDD100K classwise mAP

|         | ped   | rider | car   | truck | bus   | moto  | bicycle | light | sign  |
| ------- | ----- | ----- | ----- | ----- | ----- | ----- | ------- | ----- | ----- |
| r50_fpn | 0.175 | 0.080 | 0.403 | 0.182 | 0.184 | 0.055 | 0.058   | 0.175 | 0.244 |
|         |       |       |       |       |       |       |         |       |       |
|         |       |       |       |       |       |       |         |       |       |



- NuImages







## detectors

```python
# mmdet/models/detectors/faster_rcnn.py
@DETECTORS.register_module()
class FasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
```

## backbone

- config

```python
backbone=dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=True,
    style='pytorch')
```

- def

```python
# mmdet/models/backbones/resnet.py
pass
```

## neck

- config

```python
neck=dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_outs=5)
```

- def

```python
# mmdet/models/necks/fpn.py
pass
```

## rpn_head

- config

```python
rpn_head=dict(
    type='RPNHead',
    in_channels=256,
    feat_channels=256,
    anchor_generator=dict(
        type='AnchorGenerator',
        scales=[8],
        ratios=[0.5, 1.0, 2.0],
        strides=[4, 8, 16, 32, 64]),
    bbox_coder=dict(
        type='DeltaXYWHBBoxCoder',
        target_means=[0.0, 0.0, 0.0, 0.0],
        target_stds=[1.0, 1.0, 1.0, 1.0]),
    loss_cls=dict(
        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    loss_bbox=dict(type='L1Loss', loss_weight=1.0))
```

- def

```python
# mmdet/models/dense_heads/rpn_head.py
# 输入：
#      in_channels(int), 输入几个特征层
#      init_cfg(dict), 初始化方式
#      num_convs(int), 卷基层数量





```

## roi_head

- config

```python
roi_head=dict(
    type='StandardRoIHead',
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='Shared2FCBBoxHead',
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=10,
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)))
```

- def

```python
```

## train_cfg





## test_cfg
