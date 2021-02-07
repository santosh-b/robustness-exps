from utils import *
import torch
import robustness

import torch.nn.functional as F
import inspect
import torch.nn as nn

unpruned_eb = sys.argv[1]
pct = sys.argv[2]
log_folder = sys.argv[3]
dataset = sys.argv[4]
alpha = sys.argv[5]
T = sys.argv[6]

alpha = float(alpha)
T = float(T)
pct = float(pct)

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
    # teacher is the unpruned EB Ticket
    teacher = vgg(16, seed=0)
    teacher.load_state_dict(weight_before_prune, strict=False)
    teacher.cuda()

    initial_weights, mask = get_pruned_init(model, cfg, pct, 'cifar10')

    ds = CIFAR('cifar')
    m, _ = make_and_restore_model(arch=initial_weights, dataset=ds)
    train_loader, val_loader = ds.make_loaders(batch_size=128, workers=8)
    out_store = cox.store.Store('store', exp_id=log_folder)
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
    # teacher is the unpruned EB Ticket
    teacher = vgg(16, dataset='cifar100', seed=0)
    teacher.load_state_dict(weight_before_prune, strict=False)
    teacher.cuda()

    initial_weights, mask = get_pruned_init(model, cfg, pct, 'cifar100')
    os.system('pip install cifar2png')
    os.system('cifar2png cifar100 cifar100')
    ds = CIFAR100('cifar100')
    m, _ = make_and_restore_model(arch=initial_weights, dataset=ds)
    train_loader, val_loader = ds.make_loaders(batch_size=128, workers=8)
    out_store = cox.store.Store('store', exp_id=log_folder)


## ---------------------------------- Implementing KD Loss ---------------------------- ##

# Define the KD Loss
def loss_fn_kd(outputs, labels, teacher_outputs):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    global alpha, T
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss

# The robustness library requires us to define a 'standard' loss and 'adversarial' loss.
# I just define adversarial loss to be the same as standard loss except with reduction='none' cross_entropy
def adv_loss_fn_kd(outputs, labels, teacher_outputs):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    global alpha, T
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels, reduction='none') * (1. - alpha)

    return KD_loss

def custom_train_loss(output, target):
    global teacher
    # The library doesn't give us the inputs by default in standard train loss,
    # so we have to look up the call stack and see it
    inputs = inspect.currentframe().f_back.f_locals['inp']
    inputs = inputs.cuda()
    with torch.no_grad():
        tch_logit = teacher(inputs)
    inputs = inputs.cpu()

    # KL div loss (tch_logit, outputs) + CE loss (target, output)
    loss = loss_fn_kd(output, target, tch_logit)
    return loss

def custom_adv_loss(model, input, target):
    output = model(input)
    global teacher
    input = input.cuda()
    with torch.no_grad():
        tch_logit = teacher(input)
    input = input.cpu()

    # KL div loss (tch_logit, outputs) + CE loss (target, output)
    loss = adv_loss_fn_kd(output, target, tch_logit)
    return loss, output

## -------------------------------------- Finished implementing KD loss --------------------------- ##

train_args = Parameters({
    'out_dir': 'train_out',
    'adv_train': 1,
    'adv_eval': 1,
    'constraint': 'inf',
    'eps': 8/255,
    'attack_lr': 2/255,
    'attack_steps': 7,
    'epochs': 110,
    'save_ckpt_iters': 110,
    'weight_decay':5e-4,
    'data_aug': 0,
    'log_iters': 1,
    'lr': 1,
    'mixed_precision':0,
    'custom_lr_multiplier': '[(0,.1),(100,.01),(105,.001)]'
})

# Fill whatever parameters are missing from the defaults
train_args = defaults.check_and_fill_args(train_args,
                        defaults.TRAINING_ARGS, CIFAR)
train_args = defaults.check_and_fill_args(train_args,
                        defaults.PGD_ARGS, CIFAR)

def log_retrain(model, log_info):
    print('[Log]',log_info)
    with open('store/'+log_folder+'/log.txt', 'a') as f:
        f.write(str(log_info['epoch'])+' '+
                str(log_info['nat_prec1'].item())+' '+
                str(log_info['adv_prec1'].item())+' '+
                str(log_info['nat_loss'])+' '+
                str(log_info['adv_loss'])+' '+
                str(log_info['train_prec1'].item())+' '+
                str(log_info['train_loss'])+'\n')
train_args.epoch_hook = log_retrain
# Setting the custom losses as our KD losses
train_args.custom_train_loss = custom_train_loss
train_args.custom_adv_loss = custom_adv_loss

train.train_model(train_args, m, (train_loader, val_loader), store=out_store)
