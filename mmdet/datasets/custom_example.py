import mmcv
import numpy as np
from .builder import DATASETS
from .custom import CustomDataset
# 自定义数据集方法
# 1. 转为现有格式COCO or PASCAL VOC
# 2. 使用如下中间格式, 继承CustomDataset, 重写load_annotations(self, ann_file)和get_ann_info(self, idx)方法
'''
[
    {
        'filename': 'a.jpg',
        'width': 1280,
        'height': 720,
        'ann': {
            'bboxes': <np.ndarray, float32> (n, 4),
            'labels': <np.ndarray, int64> (n, ),
            'bboxes_ignore': <np.ndarray, float32> (k, 4),
            'labels_ignore': <np.ndarray, int64> (k, ) (optional field)
        }
    },
    ...
]
'''
# 3. 任意格式，txt为例：
'''
000001.jpg
1280 720
2
10 20 40 60 1
20 40 50 60 2
#
000002.jpg
1280 720
3
50 20 40 60 2
20 40 30 45 2
30 40 50 60 3
'''

@DATASETS.register_module()
class MyDataset(CustomDataset):
    # 自定义类别
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle') # 0,1,2,3
    # 重写数据加载部分，ann_file从config文件传入
    def load_annotations(self, ann_file):
        ann_list = mmcv.list_from_file(ann_file)
        # 组织为data_infos
        data_infos = []
        for i, ann_line in enumerate(ann_list):
            if ann_line != '#':
                continue
            img_shape = ann_list[i + 2].split(' ')
            width = int(img_shape[0])
            height = int(img_shape[1])
            bbox_number = int(ann_list[i + 3])
            anns = ann_line.split(' ')
            bboxes = []
            labels = []
            for anns in ann_list[i + 4:i + 4 + bbox_number]:
                bboxes.append([float(ann) for ann in anns[:4]])
                labels.append(int(anns[4]))
            data_infos.append(
                dict(
                    filename=ann_list[i + 1],
                    width=width,
                    height=height,
                    ann=dict(
                        bboxes=np.array(bboxes).astype(np.float32),
                        labels=np.array(labels).astype(np.int64))
                ))
        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']

# 在config中使用：
'''
dataset_A_train = dict(
    type='MyDataset',
    ann_file = 'image_list.txt',
    pipeline=train_pipeline
)
'''
# 其他
# RepeatDataset, 简单的重复采样数据集
'''
dataset_A_train = dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
'''
# ClassBalancedDataset，根据类别数量均衡采样，需要实现self.get_cat_ids(idx)
'''
dataset_A_train = dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )

'''
# ConcatDataset，堆叠相同数据格式多个标注文件
'''
dataset_A_train = dict(
    type='Dataset_A',
    ann_file = ['anno_file_1', 'anno_file_2'],
    pipeline=train_pipeline
)
'''
# 堆叠不同格式的数据集
'''
dataset_A_train = dict()
dataset_B_train = dict()

data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train = [
        dataset_A_train,
        dataset_B_train
    ],
    val = dataset_A_val,
    test = dataset_A_test
    )
'''
# 或
'''
dataset_A_val = dict()
dataset_B_val = dict()

data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dataset_A_train,
    val=dict(
        type='ConcatDataset',
        datasets=[dataset_A_val, dataset_B_val],
        separate_eval=False))

'''
# 过滤数据集的类别
'''
classes = ('person', 'bicycle', 'car')
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))
'''
















