[toc]

# Faster-RCNN

- BDD100K数据集

|      | ped  | rider | car  | truck | bus  | train | moto | bicycle | light | sign | mAP@0.5 |
| ---- | ---- | ----- | ---- | ----- | ---- | ----- | ---- | ------- | ----- | ---- | ------- |
|      |      |       |      |       |      |       |      |         |       |      |         |
|      |      |       |      |       |      |       |      |         |       |      |         |
|      |      |       |      |       |      |       |      |         |       |      |         |

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

