import argparse
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch_geometric.datasets import FAUST
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

from models import FeaStNet
from utils import run

parser = argparse.ArgumentParser(description='shape correspondence')
parser.add_argument('--dataset', type=str, default='FAUST')
parser.add_argument('--device_idx', type=int, default=0)
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--heads', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lr_decay', type=float, default=0.5)
parser.add_argument('--decay_step', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--t_inv', action='store_false', default=True)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

args.data_fp = osp.join(osp.dirname(osp.realpath(__file__)), 'data',
                        args.dataset)
device = torch.device('cuda', args.device_idx)
torch.set_num_threads(args.n_threads)

# deterministic
torch.manual_seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True


class Pre_Transform(object):
    def __call__(self, data):
        data.x = data.pos
        data = T.FaceToEdge()(data)
        data.pos = None
        return data


train_dataset = FAUST(args.data_fp, True, pre_transform=Pre_Transform())
test_dataset = FAUST(args.data_fp, False, pre_transform=Pre_Transform())
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)
d = train_dataset[0]
target = torch.arange(d.num_nodes, dtype=torch.long, device=device)
print(d)

model = FeaStNet(d.num_features, d.num_nodes, args.heads).to(device)
print(model)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      args.decay_step,
                                      gamma=args.lr_decay)

run(model, train_loader, test_loader, target, d.num_nodes, args.epochs,
    optimizer, scheduler, device)
