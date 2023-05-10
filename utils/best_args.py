best_args = {
    'seq-cifar10': {'agile': {200: {'lr': 0.07,
                                    'minibatch_size': 32,
                                    'batch_size': 32,
                                    'n_epochs': 50,
                                    'ema_alpha': 0.999,
                                    'ema_update_freq': 0.2,
                                    'reg_weight': 0.15,
                                    'pairwise_weight': 0.1,
                                    'task_loss_weight': 1.0,
                                    'code_dim': 256,
                                    'attention': 'ae_sigmoid',
                                    },
                              500: {'lr': 0.05,
                                    'minibatch_size': 32,
                                    'batch_size': 32,
                                    'n_epochs': 50,
                                    'ema_alpha': 0.999,
                                    'ema_update_freq': 0.2,
                                    'reg_weight': 0.10,
                                    'pairwise_weight': 0.1,
                                    'task_loss_weight': 1.0,
                                    'code_dim': 256,
                                    'attention': 'ae_sigmoid',
                                    }},
                    },
    'seq-cifar100': {'agile': {200: {'lr': 0.03,
                                     'minibatch_size': 32,
                                     'batch_size': 32,
                                     'n_epochs': 50,
                                     'ema_alpha': 0.999,
                                     'ema_update_freq': 0.05,
                                     'reg_weight': 0.10,
                                     'pairwise_weight': 0.1,
                                     'task_loss_weight': 1.0,
                                     'code_dim': 256,
                                     'attention': 'ae_sigmoid',
                                     },
                               500: {'lr': 0.07,
                                     'minibatch_size': 32,
                                     'batch_size': 32,
                                     'n_epochs': 50,
                                     'ema_alpha': 0.999,
                                     'ema_update_freq': 0.08,
                                     'reg_weight': 0.15,
                                     'pairwise_weight': 0.1,
                                     'task_loss_weight': 1.0,
                                     'code_dim': 256,
                                     'attention': 'ae_sigmoid',
                                     }},
                     },
    'seq-tinyimg': {'agile': {200: {'lr': 0.05,
                                     'minibatch_size': 32,
                                     'batch_size': 32,
                                     'n_epochs': 50,
                                     'ema_alpha': 0.999,
                                     'ema_update_freq': 0.05,
                                     'reg_weight': 0.10,
                                     'pairwise_weight': 0.1,
                                     'task_loss_weight': 1.0,
                                     'code_dim': 256,
                                     'attention': 'ae_sigmoid',
                                     },
                               500: {'lr': 0.03,
                                     'minibatch_size': 32,
                                     'batch_size': 32,
                                     'n_epochs': 50,
                                     'ema_alpha': 0.999,
                                     'ema_update_freq': 0.05,
                                     'reg_weight': 0.10,
                                     'pairwise_weight': 0.5,
                                     'task_loss_weight': 1.0,
                                     'code_dim': 256,
                                     'attention': 'ae_sigmoid',
                                     }},
    }}
