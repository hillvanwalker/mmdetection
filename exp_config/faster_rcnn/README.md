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



- NuImages数据集

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

