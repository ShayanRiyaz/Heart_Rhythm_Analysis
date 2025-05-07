import os

def epoch_num(path):
    # take basename → 'ckpt_epoch_9.pth'
    name = os.path.basename(path)
    # split on '_' and take last piece → '9.pth'
    tail = name.split('_')[-1]
    # strip extension and cast → 9
    return int(tail.split('.')[0])

