import os
import torch
from utils import *

unpruned_eb = sys.argv[1]
pct = float(sys.argv[2])
log_folder = sys.argv[4]
data_dir = sys.argv[3]
dataset = sys.argv[5]

weight_before_prune = fix_robustness_ckpt(torch.load(unpruned_eb))
try:
    os.mkdir('tmp')
except OSError as error:
    pass

if dataset == 'cifar10':
    model = resnet18(seed=0, num_classes=10)
    model.load_state_dict(weight_before_prune, strict=False)
    cfg = resprune(model.cuda(), pct)
    initial_weights, mask = get_resnet_pruned_init(model, cfg, pct, 'cifar10')
    torch.save(initial_weights.state_dict(), 'tmp/eb_reset.pt')
elif dataset == 'cifar100':
    model = resnet18(seed=0, num_classes=100)
    model.load_state_dict(weight_before_prune, strict=False)
    cfg = resprune(model.cuda(), pct)
    initial_weights, mask = get_resnet_pruned_init(model, cfg, pct, 'cifar100')
    torch.save(initial_weights.state_dict(), 'tmp/eb_reset.pt')  

print(initial_weights)

os.system(f'python "fast_adversarial-master/CIFAR10/train_fgsm.py" \
--cfg-dir=tmp/pruned.pth.tar \
--model="resnet" \
--cfg="{cfg}" \
--model-dir=tmp/eb_reset.pt \
--data-dir="{data_dir}" \
--epochs=110 \
--out-dir="store/{log_folder}" \
--dataset="{dataset}"'
)
