# ImageNet-wIth-augmentation

## Requirements
This code has been tested with  
python 3.6.3  
pytorch 0.4.1  
torchvision 0.2.1  
numpy 1.41.3

## Examples
--model: ResNet50, ResNet101, ResNeXt101_32x4d
--augment: None, mixup, cutmix, autoaug

### Standard 

```
python imagenet_train.py --epoch 90 ---model ResNet50 --seed=20190901 --name baseline --data_dir xxx
```

### Mixup

```
python imagenet_train.py --epoch 200 --model ResNet50 --augment mixup --alpha 0.2 --seed=20190901 --name mixup_a0.2 --data_dir xxx
```

### Cutmix

```
python imagenet_train.py --epoch 300 --model ResNet50 --augment cutmix --seed=20190901 --name cutmix --data_dir xxx
```

### Auto augment

```
python imagenet_train.py --epoch 300 --model ResNet50 --augment autoaug --seed=20190901 --name autoaug --data_dir xxx
```
