[toc]

# 训练

- `tools/train.py`参数传递

```python
config             # config file
--work-dir         # dir to save logs and models
--resume-from      # checkpoint file
--auto-resume      # resume from the latest
--no-validate      # not to evaluate the checkpoint
--cfg-options      # override settings in config
--auto-scale-lr    # automatically scaling LR
```

- 读取config文件及设置、目录、日志等

```python
# mmcv/utils/config.py, class Config
cfg = Config.fromfile(args.config)
# ...
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
```

- 构建模型并初始化

```python
model = build_detector(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg'))
model.init_weights()
```

- 构建数据集

```python
datasets = [build_dataset(cfg.data.train)]
if len(cfg.workflow) == 2:
    val_dataset = copy.deepcopy(cfg.data.val)
    val_dataset.pipeline = cfg.data.train.pipeline
    datasets.append(build_dataset(val_dataset))
```

- 开始训练

```python
train_detector(
    model,
    datasets,
    cfg,
    distributed=distributed,
    validate=(not args.no_validate),
    timestamp=timestamp,
    meta=meta)
```

## build_detector

- 从config构建模型

```python
# mmdet/models/builder.py
def build_detector(cfg, train_cfg=None, test_cfg=None):
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
# mmcv/utils/registry.py
class Registry:
    """A registry to map strings to classes.
    Registered object could be built from registry.
    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> resnet = MODELS.build(dict(type='ResNet'))
    """
    def build(self, *args, **kwargs):
        return self.build_func(*args, **kwargs, registry=self)

# self.build_func will be set with the following priority:
# 1. build_func
# 2. parent.build_func
# 3. build_from_cfg
def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.
    """
# ------------------------------> mmdet/models/detectors/xxx.py
```

### BaseDetector

- 基本检测模型

```python
# mmdet/models/detectors/base.py
class BaseDetector(BaseModule, metaclass=ABCMeta):
    def __init__(self, init_cfg=None):
        super(BaseDetector, self).__init__(init_cfg)
        self.fp16_enabled = False
    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None
    @property
    def with_shared_head(self):
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head
    # with_bbox, with_mask, ...
    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass
    def extract_feats(self, imgs):
        """Extract features from multiple images.
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]
    def forward_train(self, imgs, img_metas, **kwargs):
        pass
    # simple_test, aug_test, forward_test, ...
    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)
    # _parse_losses, train_step, val_step, onnx_export
    # show_result
```

### TwoStageDetector

- 二阶段模型

```python
# mmdet/models/detectors/two_stage.py
@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        # 根据config中是否指定neck, rpn, roi, head等分别进行初始化 
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.rpn_head = build_head(rpn_head_)
        self.roi_head = build_head(roi_head)
    # 训练
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        # 前向计算，汇总loss
        return losses
```

### SingleStageDetector

- 一阶段模型

```python
@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
    # 其余函数基本类似
```

## build_dataset

- 几种数据集类别

```python
def build_dataset(cfg, default_args=None):
    from .dataset_wrappers import (ClassBalancedDataset, ConcatDataset,
                                   MultiImageMixDataset, RepeatDataset)
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'MultiImageMixDataset':
        cp_cfg = copy.deepcopy(cfg)
        cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'])
        cp_cfg.pop('type')
        dataset = MultiImageMixDataset(**cp_cfg)
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset
```

## train_detector

- mmdet api

```python
# mmdet/apis/train.py
def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    # 设置日志
    logger = get_root_logger(log_level=cfg.log_level)
    # 准备dataloaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']
    # train_dataloader_default_args = dict(...)
    # train_loader_cfg = {...}
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]
    # 将模型移到gpu，put model on gpus...ddp
    # 构建optimizer
    auto_scale_lr(cfg, distributed, logger)
    optimizer = build_optimizer(model, cfg.optimizer)
    # 构建runner
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))
    # 设置hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))
    # eval hooks
    if validate:
		pass
    # 设置resume
    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    # 启动run
    runner.run(data_loaders, cfg.workflow)
    # ------> 去各hooks执行
```
