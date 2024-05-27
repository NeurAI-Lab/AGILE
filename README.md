# AGILE - Mitigating Interference in Incremental Learning through Attention-Guided Rehearsal

The official repository for [CoLLAs'24 paper](https://openreview.net/pdf?id=kDLv7fvm9Y). We extended the original repo [DER++](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html) with our method.

<img width="878" alt="Screenshot 2024-05-27 at 13 30 25" src="https://github.com/NeurAI-Lab/AGILE/assets/27284368/46109324-6ed1-48ef-a87b-a7d37d0bec2c">


### How to run?
+ python main.py  --seed 10  --dataset seq-cifar10  --model agile --buffer_size 200   --load_best_args \
 --tensorboard --notes 'AGILE baseline'
        
### Setup

+ Use `./utils/main.py` to run experiments.
+ Use argument `--load_best_args` to use the best hyperparameters from the paper.
+ New models can be added to the `models/` folder.
+ New datasets can be added to the `datasets/` folder.

### Models

+ Attention-Guided Incremental Learning (AGILE)

### Datasets

   **Class-Il / Task-IL settings**

     + Sequential CIFAR-10
     + Sequential CIFAR-100
     + Sequential Tiny ImageNet
