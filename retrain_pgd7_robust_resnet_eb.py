from utils import *
import torch
import robustness

unpruned_eb = sys.argv[1]
pct = sys.argv[2]
log_folder = sys.argv[3]
dataset = sys.argv[4]


print(pct)
pct = float(pct)

print(unpruned_eb)
weight_before_prune = fix_robustness_ckpt(torch.load(unpruned_eb))

if dataset == 'cifar10':
    model = resnet18(seed=0, num_classes=10)
    model.load_state_dict(weight_before_prune, strict=False)
    cfg = resprune(model.cuda(), pct)
    initial_weights, mask = get_resnet_pruned_init(model, cfg, pct, 'cifar10')
    ds = CIFAR('cifar')
    m, _ = make_and_restore_model(arch=initial_weights, dataset=ds)
    train_loader, val_loader = ds.make_loaders(batch_size=128, workers=8)
    out_store = cox.store.Store('store', exp_id=log_folder)

elif dataset == 'cifar100':
    model = resnet18(seed=0, num_classes=100)
    model.load_state_dict(weight_before_prune, strict=False)
    cfg = resprune(model.cuda(), pct)
    initial_weights, mask = get_resnet_pruned_init(model, cfg, pct, 'cifar100')
    os.system('pip install cifar2png')
    os.system('cifar2png cifar100 cifar100')
    ds = CIFAR100('cifar100')
    m, _ = make_and_restore_model(arch=initial_weights, dataset=ds)
    train_loader, val_loader = ds.make_loaders(batch_size=128, workers=8)
    out_store = cox.store.Store('store', exp_id=log_folder)

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
    'lr': 1,
    'weight_decay':5e-4,
    'data_aug': 0,
    'log_iters': 1,
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

train.train_model(train_args, m, (train_loader, val_loader), store=out_store)
