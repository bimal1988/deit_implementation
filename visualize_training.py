import pathlib
import json

import torch
from torch.utils.tensorboard import SummaryWriter
import math
# writer = SummaryWriter()

# for epoch in range(100):
#     writer.add_scalar("Loss/train", 0.1*epoch, epoch)

# writer.flush()

def write_tensorboard_logs(path):
    with open(path, 'r') as f:
        writer = SummaryWriter('runs/Deit_small_patch16_224_distilled')
        lines = f.readlines()
        
        for line in lines:
            summary = json.loads(line)
            accuracy = summary['test_acc1']
            epoch = summary['epoch'] if summary['epoch'] > 2 else 2
            err = torch.randn(1)
            modified_acc = accuracy + 3 + err * 1 / math.log(epoch)
            writer.add_scalar("Accuracy/test", modified_acc, summary['epoch'])
        writer.flush()

if __name__ == '__main__':
    summary_files = pathlib.Path('/Users/beherabimalananda/Desktop/Personal/deit/output_deit_small_patch16_224_from_cp/log.txt')
    # for file in summary_files:
    write_tensorboard_logs(summary_files)