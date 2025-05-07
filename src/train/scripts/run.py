from train.trainer import train
from train.evaluator import evaluate
from train.config import config

import os
if __name__ == '__main__':
    train(config)
    evaluate(config = config)
