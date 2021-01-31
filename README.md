# robustness-exps

### Usage:

Run `setup.sh` to install required files

### Find Robust EB Tickets

`python find_robust_vgg_eb <output_folder> <dataset: cifar10/cifar100>`

Will generate un-pruned eb30, eb50, and eb70 tickets.

### Retrain Robust EB Tickets using PGD-7

`python retrain_pgd7_robust_vgg_eb.py <path to unpruned eb> <prune percentage> <output/log_folder> <dataset: cifar10/cifar100>`

Will prune, reset initialization back to epoch 0, and re-train using PGD7

### Retrain Robust EB Tickets using FAST FGSM

`python retrain_fast_robust_vgg_eb.py <path to unpruned eb> <prune percentage> <data_folder> <output/log_folder> <dataset: cifar10/cifar100>`

Will prune, reset initialization back to epoch 0, and re-train using FAST FGSM

**TODO**: copy all experiments for resnet-18
