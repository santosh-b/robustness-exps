import os
import torch
from utils import *

unpruned_eb = sys.argv[1]
pct = float(sys.argv[2])
log_folder = sys.argv[4]
data_dir = sys.argv[3]
dataset = sys.argv[5]

weight_before_prune = fix_robustness_ckpt(torch.load(unpruned_eb))

if dataset == 'cifar10':
    os.system(f'python "Early-Bird-Tickets/vggprune.py" \
    --dataset cifar10 \
    --test-batch-size 128 \
    --depth 16 \
    --percent {pct} \
    --model "{unpruned_eb}" \
    --save "tmp" \
    --gpu_ids 0')

    cfg = torch.load('tmp/pruned.pth.tar')['cfg']

    model = vgg(16, seed=0)
    model.load_state_dict(weight_before_prune, strict=False)
    initial_weights, mask = get_pruned_init(model, cfg, pct, 'cifar10')
    torch.save(initial_weights.state_dict(), 'tmp/eb_reset.pt')
elif dataset == 'cifar100':

    os.system(f'python "Early-Bird-Tickets/vggprune.py" \
    --dataset cifar100 \
    --test-batch-size 128 \
    --depth 16 \
    --percent {pct} \
    --model "{unpruned_eb}" \
    --save "tmp" \
    --gpu_ids 0')

    cfg = torch.load('tmp/pruned.pth.tar')['cfg']

    model = vgg(16, dataset='cifar100', seed=0)
    model.load_state_dict(weight_before_prune, strict=False)
    initial_weights, mask = get_pruned_init(model, cfg, pct, 'cifar100')
    torch.save(initial_weights.state_dict(), 'tmp/eb_reset.pt')  

print(initial_weights)

os.system(f'python "fast_adversarial-master/CIFAR10/train_fgsm.py" \
--cfg-dir=tmp/pruned.pth.tar \
--model-dir=tmp/eb_reset.pt \
--data-dir="{data_dir}" \
--epochs=110 \
--out-dir="store/{log_folder}" \
--dataset="{dataset}"'
)