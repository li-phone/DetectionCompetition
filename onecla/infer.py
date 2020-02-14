import torch
from torchvision.models import resnet34, resnet101
import argparse
import numpy as np
import os
import torch.nn as nn
from torch.autograd import Variable
from torchnet import meter
from sklearn.metrics.classification import classification_report
import pandas as pd
from tqdm import tqdm
from torch.utils.data import *
from build_network import *
from utils import *
from data_reader import DataReader, collate_fn


def infer(model, cfg, cuda=True):
    img_dirs = [r['img_prefix'] for r in cfg.dataset['test']]
    data_reader = DataReader(None, img_dirs, transform=None)
    data_loader = DataLoader(data_reader, collate_fn=collate_fn, **cfg.val_data_loader)
    y_pred = []
    model.eval()
    for step, (data) in tqdm(enumerate(data_loader)):
        inputs = torch.stack(data)
        if cuda:
            inputs = inputs.cuda()
        with torch.no_grad():
            outputs = model(inputs)
        outs = nn.functional.softmax(outputs, dim=1)
        pred = torch.argmax(outs, dim=1)
        # pred = outs[:, 0]
        y_pred.extend(list(pred.cpu().detach().numpy()))
        # break
    model.train()
    ids = [os.path.basename(x) for x in data_reader.image_paths]
    ids = [x.split('.')[0] for x in ids]
    ids = [int(x) for i, x in enumerate(ids)]
    return pd.DataFrame(data=dict(ids=ids[:len(y_pred)], label=y_pred))


def main(cfg):
    mkdirs(cfg.work_dir)

    cfg.gpus = prase_gpus('1')
    model = build_network(**cfg.model_config, gpus=cfg.gpus)
    model, optimizer, lr_scheduler, last_epoch = resume_network(model, cfg)

    submit_df = infer(model, cfg)
    save_name = os.path.join(cfg.work_dir, '{}_epoch_{}_submit.csv'.format(cfg.model_config['name'], last_epoch))
    submit_df.to_csv(save_name, index=False, header=False)
    logger.info('infer successfully!')


if __name__ == '__main__':
    pcfg = import_module("cfg.py")
    dcfg = import_module(pcfg.dataset_cfg_path)
    main(dcfg)
