# DeiT: Data-efficient Image Transformers Implementation

# Usage

Install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

```python
import torch
# check you have the right version of timm
import timm
assert timm.__version__ == "0.3.2"

# now load it with torchhub
model = torch.hub.load('path_to_weight_file', pretrained=True)
```

## Evaluation
To evaluate the pre-trained DeiT-tiny on Cifar 100 val with a single GPU run:
```
python main.py --eval --resume path/to/tiny/weight.pth --data-path /path/to/cifar100
```

For Deit-small, run:
```
python main.py --eval --resume path/to/small/weight.pth --model deit_small_patch16_224 --data-path /path/to/cifar100
```

For Deit-base, run:
```
python main.py --eval --model deit_base_distilled_patch16_224 --resume path/to/base/weight.pth
```

## Training
To train DeiT-small and Deit-tiny on Cifar100 on a single node with 4 gpus for 300 epochs run:

DeiT-small
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_small_patch16_224 --batch-size 256 --data-path /path/to/cifar100 --output_dir /path/to/save
```

DeiT-tiny
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path /path/to/cifar100 --output_dir /path/to/save
```

# Note: Due to size constraint only Deit-tiny weight file is provided
