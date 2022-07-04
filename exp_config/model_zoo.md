# mmdet/models/detectors/

| Detector                 | Struct                  | box      | mask     | fps  |
| ------------------------ | ----------------------- | -------- | -------- | ---- |
| **faster_rcnn**          | **R-50-FPN**            | **38.4** |          |      |
| + guided anchoring       | R-50-FPN(ga)            | 39.6     |          |      |
| + train from scratch     | R-50-FPN(scratch)       | 40.7     |          |      |
| + dynamic RCNN           | R-50-FPN(dy)            | 38.9(1)  |          |      |
|                          | R-101-FPN               | 39.8     |          |      |
|                          | R-101-FPN(ga)           | 41.5     |          |      |
| **mask_rcnn**            | **R-50-FPN**            | **39.2** | **35.4** |      |
| + group normalization    | R-50-FPN(gn)            | 40.5     | 36.7     |      |
| + weight standardization | R-50-FPN(gn+ws)         | 41.1     | 37.1     |      |
| + deformable convolution | R-50-FPN(dcn(c3-c5))    | 41.8     | 37.4     |      |
| + train from scratch     | R-50-FPN(scratch)       | 41.2     | 37.4     |      |
|                          | R-101-FPN               | 40.8     | 36.6     |      |
|                          | R-101-FPN(gn)           | 42.1     | 38.0     |      |
|                          | R-101-FPN(gn+ws)        | 43.1     | 38.6     |      |
|                          | R-101-FPN(dcn(c3-c5))   | 43.5     | 38.9     |      |
| **retinanet**            | **R-50-FPN**            | **37.4** |          |      |
| + free anchor            | R-50-FPN(fa)            | 38.7     |          |      |
|                          | R-101-FPN               | 38.9     |          |      |
|                          | R-101-FPN(fa)           | 40.3     |          |      |
| **cascade_rcnn**         | **R-50-FPN**            | **41.0** |          |      |
|                          | R-101-FPN               | 42.5     |          |      |
| **htc**                  | **R-50-FPN**            | **43.3** | **38.3** |      |
|                          | R-101-FPN               | 44.8     | 39.6     |      |
| **fcos**                 | **R-50(gn)**            | **36.6** |          |      |
|                          | R-101(gn)               | 39.1     |          |      |
| **foveabox**             | **R-50**                | **37.2** |          |      |
|                          | R-101                   | 40.0     |          |      |
| **atss**                 | **R-50**                | **39.4** |          |      |
|                          | R-101                   | 41.5     |          |      |
| **fsaf**                 | **R-50**                | **37.4** |          |      |
|                          | R-101                   | 39.3     |          |      |
| **detectors**            | **Cascade + ResNet-50** | **47.4** |          |      |
|                          | HTC + ResNet-50         | 49.1     | 42.6     |      |
|                          | HTC + ResNet-101        | 50.5     | 43.9     |      |
| **gfl**                  | **R-50**(ms)            | **42.9** |          |      |
| + multi-scale training   | R-101(ms)               | 44.7     |          |      |
| + deformable convolution | R-101(dcnv2)            | 47.1     |          |      |
|                          |                         |          |          |      |
| detr                     | R-50                    | 40.1     |          |      |
| deformable_detr          | R-50                    | 44.5     |          |      |
|                          |                         |          |          |      |
| yolox                    | s                       | 40.5     |          |      |
|                          | l                       | 49.4     |          |      |
|                          | x                       | 50.9     |          |      |
|                          |                         |          |          |      |
|                          |                         |          |          |      |

- 表中选取`2x Lr schd`结果，实际训练3x，以及采用multiscale训练策略等，还会有些许提升
- 更换backbone为X-101-64x4d等也会有较大提升
- Instaboost在实例分割中加入数据增广等方法
- 其他backbone：regnet，Res2Net，resnest，efficientnet
