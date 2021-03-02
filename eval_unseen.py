from utils import *
import torch
import robustness

import os
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms


import torch.nn.functional as F
import inspect
import torch.nn as nn

unpruned_eb = sys.argv[1]
final_weights = sys.argv[2]
pct = sys.argv[3]
log_folder = sys.argv[4]
dataset = sys.argv[5]
model_type = sys.argv[6]
is_pruned = sys.argv[7]

is_pruned = True if is_pruned=='True' else False
pct = float(pct)

if is_pruned:
    weight_before_prune = fix_robustness_ckpt(torch.load(unpruned_eb))
    if model_type == 'resnet':
        if dataset == 'cifar10':
            model = resnet18(seed=0, num_classes=10)
            model.load_state_dict(weight_before_prune, strict=False)
            cfg = resprune(model.cuda(), pct)
        elif dataset == 'cifar100':
            model = resnet18(seed=0, num_classes=100)
            model.load_state_dict(weight_before_prune, strict=False)
            cfg = resprune(model.cuda(), pct)
            os.system('pip install cifar2png')
            os.system('cifar2png cifar100 cifar100')

    if model_type == 'resnet50':
        if dataset == 'cifar10':
            model = resnet50_official(seed=0, num_classes=10)
            model.load_state_dict(weight_before_prune, strict=False)
            cfg = resprune(model.cuda(), pct)
        elif dataset == 'cifar100':
            model = resnet50_officla(seed=0, num_classes=100)
            model.load_state_dict(weight_before_prune, strict=False)
            cfg = resprune(model.cuda(), pct)
            os.system('pip install cifar2png')
            os.system('cifar2png cifar100 cifar100')

    elif model_type == 'vgg':
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
else:
    cfg=None

transform_list = [transforms.ToTensor()]
transform_chain = transforms.Compose(transform_list)
print('DATASET',dataset)
if model_type == 'resnet50':
    if dataset == 'cifar10':
        model = resnet50_official(num_classes=10, cfg=cfg, seed=0)
        model.load_state_dict(fix_robustness_ckpt(torch.load(final_weights)), strict=False)
        ds = CIFAR('cifar')
        item = datasets.CIFAR10(root='cifar', train=False, transform=transform_chain, download=True)
    elif dataset == 'cifar100':
        model = resnet50_official(num_classes=100, cfg=cfg, seed=0)
        model.load_state_dict(fix_robustness_ckpt(torch.load(final_weights)), strict=False)
        ds = CIFAR100('cifar100')
        item = datasets.CIFAR100(root='cifar100', train=False, transform=transform_chain, download=True)
elif model_type == 'resnet':
    if dataset == 'cifar10':
        model = resnet18(num_classes=10, cfg=cfg, seed=0)
        model.load_state_dict(fix_robustness_ckpt(torch.load(final_weights)), strict=False)
        ds = CIFAR('cifar')
        item = datasets.CIFAR10(root='cifar', train=False, transform=transform_chain, download=True)
    elif dataset == 'cifar100':
        model = resnet18(num_classes=100, cfg=cfg, seed=0)
        model.load_state_dict(fix_robustness_ckpt(torch.load(final_weights)), strict=False)
        ds = CIFAR100('cifar100')
        item = datasets.CIFAR100(root='cifar100', train=False, transform=transform_chain, download=True)
elif model_type == 'vgg':
    if dataset == 'cifar10':
        model = vgg(16, dataset='cifar10', cfg=cfg, seed=0)
        model.load_state_dict(fix_robustness_ckpt(torch.load(final_weights)), strict=False)
        ds = CIFAR('cifar')
        item = datasets.CIFAR10(root='cifar', train=False, transform=transform_chain, download=True)
    elif dataset == 'cifar100':
        model = vgg(16, dataset='cifar100', cfg=cfg, seed=0)
        model.load_state_dict(fix_robustness_ckpt(torch.load(final_weights)), strict=False)
        ds = CIFAR100('cifar100')
        item = datasets.CIFAR100(root='cifar100', train=False, transform=transform_chain, download=True)

## AA EVAL ##

test_loader = data.DataLoader(item, batch_size=128, shuffle=False, num_workers=0)
from autoattack import AutoAttack
log = 'store/'+log_folder+'/AA_eval-new.txt'
model = model.cuda()
adversary = AutoAttack(model, norm='Linf', eps=8/255, log_path=log,version='standard')
adversary.attacks_to_run = ['apgd-t']
l = [x for (x, y) in test_loader]
x_test = torch.cat(l, 0).cuda()
l = [y for (x, y) in test_loader]
y_test = torch.cat(l, 0).cuda()
adv_complete = adversary.run_standard_evaluation(x_test, y_test,bs=128)
save_dir = 'store/'+log_folder
print(adv_complete)
torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(
save_dir, 'aa', 'standard', adv_complete.shape[0], 8/255))

## PGD20 EVAL ##
out_store = cox.store.Store('store', exp_id=log_folder)

eval_pgd20_args = Parameters({
    'constraint': 'inf',
    'eps': 8/255,
    'attack_lr': 2/255,
    'epochs': 10,
    'attack_steps': 20,
    'save_ckpt_iters': -1,
    'adv_eval': 1,
    'use_best': 1,
    'random_restarts': 0,
})

m, _ = make_and_restore_model(arch=model, dataset=ds)
train_loader, val_loader = ds.make_loaders(batch_size=128, workers=8)
pgd20_out = train.eval_model(eval_pgd20_args, m, val_loader, store=out_store)
print('[Log w/ pgd20-eval]',pgd20_out)
with open('store/'+log_folder+'/pgd20_eval-new.txt', 'a') as f:
    f.write('[PGD-20 VAL] '+
        str(pgd20_out['nat_prec1'].item())+' '+
        str(pgd20_out['adv_prec1'].item())+' '+
        str(pgd20_out['nat_loss'])+' '+
        str(pgd20_out['adv_loss'])+' '+
        str(pgd20_out['train_prec1'])+' '+
        str(pgd20_out['train_loss'])+'\n')

# ## PGD7 EVAL ##

eval_pgd7_args = Parameters({
    'constraint': 'inf',
    'eps': 8/255,
    'attack_lr': 2/255,
    'epochs': 10,
    'attack_steps': 7,
    'save_ckpt_iters': -1,
    'adv_eval': 1,
    'use_best': 1,
    'random_restarts': 0,
})

m, _ = make_and_restore_model(arch=model, dataset=ds)
train_loader, val_loader = ds.make_loaders(batch_size=256, workers=8)
pgd7_out = train.eval_model(eval_pgd7_args, m, val_loader, store=out_store)
print('[Log w/ pgd7-eval]',pgd7_out)
with open('store/'+log_folder+'/pgd7_eval-new.txt', 'a') as f:
    f.write('[PGD-7 VAL] '+
        str(pgd7_out['nat_prec1'].item())+' '+
        str(pgd7_out['adv_prec1'].item())+' '+
        str(pgd7_out['nat_loss'])+' '+
        str(pgd7_out['adv_loss'])+' '+
        str(pgd7_out['train_prec1'])+' '+
        str(pgd7_out['train_loss'])+'\n')
