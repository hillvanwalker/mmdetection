[toc]

# Faster-RCNN

- BDD100K数据集

|      | ped  | rider | car  | truck | bus  | train | moto | bicycle | light | sign | mAP@0.5 |
| ---- | ---- | ----- | ---- | ----- | ---- | ----- | ---- | ------- | ----- | ---- | ------- |
|      |      |       |      |       |      |       |      |         |       |      |         |
|      |      |       |      |       |      |       |      |         |       |      |         |
|      |      |       |      |       |      |       |      |         |       |      |         |

- NuImages数据集





## 训练

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

