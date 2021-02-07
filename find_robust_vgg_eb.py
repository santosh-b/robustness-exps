from utils import *
import sys
import os
# uses standard SGD training to find a robust early bird ticket

log_folder = sys.argv[1] 
dataset = sys.argv[2]

if dataset == 'cifar10':
    model = vgg(16, seed=0)
    ds = CIFAR('cifar')
    m, _ = make_and_restore_model(arch=model, dataset=ds)
    train_loader, val_loader = ds.make_loaders(batch_size=128, workers=8)
    out_store = cox.store.Store('store', exp_id=log_folder)
elif dataset == 'cifar100':
    model = vgg(16, dataset='cifar100', seed=0)
    os.system('pip install cifar2png')
    os.system('cifar2png cifar100 cifar100')
    ds = CIFAR100('cifar100')
    m, _ = make_and_restore_model(arch=model, dataset=ds)
    train_loader, val_loader = ds.make_loaders(batch_size=128, workers=8)
    out_store = cox.store.Store('store', exp_id=log_folder)

train_args = Parameters({
    'out_dir': 'train_out',
    'adv_train': 0,
    'adv_eval': 0,
    'constraint': 'inf',
    'eps': 8/255,
    'attack_lr': 2/255,
    'attack_steps': 7,
    'epochs': 110,
    'lr': 1,
    'save_ckpt_iters': 110,
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

eb30 = EarlyBird(0.3)
eb30_found = False
eb50 = EarlyBird(0.5)
eb50_found = False
eb70 = EarlyBird(0.7)
eb70_found = False

def log_ebt(model, log_info):
    global eb30_found, eb50_found, eb70_found
    global eb30, eb50, eb70
    with open('store/'+log_folder+'/log.txt', 'a') as f:
        f.write(str(log_info['epoch'])+' '+
                str(log_info['nat_prec1'].item())+' '+
                str(log_info['adv_prec1'])+' '+
                str(log_info['nat_loss'])+' '+
                str(log_info['adv_loss'])+' '+
                str(log_info['train_prec1'].item())+' '+
                str(log_info['time'])+' '+
                str(log_info['train_loss'])+'\n')
    if (not eb30_found) and eb30.early_bird_emerge(model):
      print('[Early Bird] Found an EB30 Ticket @',log_info['epoch'])
      eb30_found = True
      torch.save(model.state_dict(), 'store/'+log_folder+'/eb30.pt')
      with open('store/'+log_folder+'/find_eb.txt','a') as f:
          f.write(f'Found EB30 Ticket @ {log_info["epoch"]} \n')
    if (not eb50_found) and eb50.early_bird_emerge(model):
      print('[Early Bird] Found an EB50 Ticket @',log_info['epoch'])
      eb50_found = True
      torch.save(model.state_dict(), 'store/'+log_folder+'/eb50.pt')
      with open('store/'+log_folder+'/find_eb.txt','a') as f:
          f.write(f'Found EB50 Ticket @ {log_info["epoch"]} \n')
    if (not eb70_found) and eb70.early_bird_emerge(model):
      print('[Early Bird] Found an EB70 Ticket @',log_info['epoch'])
      eb70_found = True
      torch.save(model.state_dict(), 'store/'+log_folder+'/eb70.pt')
      with open('store/'+log_folder+'/find_eb.txt','a') as f:
          f.write(f'Found EB70 Ticket @ {log_info["epoch"]} \n')
train_args.epoch_hook = log_ebt

train.train_model(train_args, m, (train_loader, val_loader), store=out_store)
