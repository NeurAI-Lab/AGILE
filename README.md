### AGILE - Mitigating Interference in Incremental Learning through Attention-Guided Rehearsal

Extended on original repo:  [Dark Experience for General Continual Learning: a Strong, Simple Baseline](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)

#### How to run?
+ python main.py  --seed 10  --dataset seq-cifar10  --model agile --buffer_size 200   --load_best_args \
 --tensorboard --notes 'AGILE baseline'
        
#### Setup

+ Use `./utils/main.py` to run experiments.
+ Use argument `--load_best_args` to use the best hyperparameters from the paper.
+ New models can be added to the `models/` folder.
+ New datasets can be added to the `datasets/` folder.

#### Models

+ Attention-Guided Incremental Learning (AGILE)

#### Datasets

**Class-Il / Task-IL settings**

+ Sequential CIFAR-10
+ Sequential CIFAR-100
+ Sequential Tiny ImageNet
