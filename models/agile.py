from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
from copy import deepcopy
from torch.nn import BCEWithLogitsLoss


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    return parser


class AGILE(ContinualModel):
    NAME = 'agile'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(AGILE, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task_id = 0
        self.ema_model = deepcopy(self.net).to(self.device)
        # set regularization weight
        self.reg_weight = self.args.reg_weight
        # set parameters for ema model
        self.ema_update_freq = self.args.ema_update_freq
        self.ema_alpha = self.args.ema_alpha
        self.consistency_loss = torch.nn.MSELoss(reduction='none')
        self.current_task = 0
        self.global_step = 0
        self.bce_loss = BCEWithLogitsLoss(reduction='mean')

    def update_ema_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.ema_alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def end_task(self, dataset):
        self.task_id += 1
        for i in range(self.task_id):
            self.net.ae_params[i].requires_grad = False
            self.net.classifiers[i].weight.requires_grad = False

    def buffer_through_ae(self):
        loss = 0
        buf_inputs, buf_labels, task_labels = self.buffer.get_data(
            self.args.minibatch_size, transform=self.transform)
        buf_feats = self.net.features(buf_inputs)
        buf_feats_ema = self.ema_model.features(buf_inputs)
        buf_output = []
        buf_outout_ema = []

        for i in range(self.task_id + 1):
            out_enc = self.net.shared_ae.encoder(buf_feats)
            out_enc = out_enc * self.net.ae_params[i].squeeze()
            out_dec = self.net.shared_ae.decoder(out_enc)
            out_cls = self.net.classifiers[i](out_dec * buf_feats).squeeze()

            out_enc_e = self.ema_model.shared_ae.encoder(buf_feats_ema)
            out_dec_e = self.ema_model.shared_ae.decoder(out_enc_e * self.ema_model.ae_params[i])
            out_cls_e = self.ema_model.classifiers[i](out_dec_e * buf_feats_ema).squeeze()

            buf_output.append(out_cls)
            buf_outout_ema.append(out_cls_e)

        buf_output = torch.cat(buf_output, dim=1).to(self.device)
        buf_outout_ema = torch.cat(buf_outout_ema, dim=1).to(self.device)

        return buf_output, buf_outout_ema, buf_labels, loss

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()

        loss = 0
        feats = self.net.features(inputs)
        outputs = []
        task_labels = torch.ones(labels.shape[0], device=self.device) * self.task_id

        out_decoder = []
        for i in range(self.task_id + 1):
            out_enc = self.net.shared_ae.encoder(feats)
            out_enc = out_enc * self.net.ae_params[i]
            out_dec = self.net.shared_ae.decoder(out_enc)
            out_decoder.append(out_dec)
            if i == self.task_id:
                task_output = self.net.task_classifier(out_enc)
                loss += self.args.task_loss_weight * self.loss(task_output.squeeze(), task_labels.long())
            out_cls = self.net.classifiers[i](out_dec * feats).squeeze()
            outputs.append(out_cls)

        # Pair wise loss
        for i in range(self.task_id + 1):
            softmax = torch.nn.Softmax(dim=-1)
            pairwise_dist = torch.pairwise_distance(softmax(out_decoder[i].detach()),
                                                    softmax(out_decoder[self.task_id]), p=1).mean()
            loss -= self.args.pairwise_weight * pairwise_dist

        outputs = torch.cat(outputs, dim=1).to(self.device)
        loss += self.loss(outputs, labels)

        loss_1 = torch.tensor(0)
        if not self.buffer.is_empty():
            buf_outputs_1, buf_logits_1, _, loss_pairwise_1 = self.buffer_through_ae()
            loss_1 = self.reg_weight * F.mse_loss(buf_outputs_1, buf_logits_1.detach())
            loss += loss_1 + loss_pairwise_1

            # CE for buffered images
            buf_outputs_2, _, buf_labels_2, loss_pairwise_2 = self.buffer_through_ae()
            loss += self.loss(buf_outputs_2, buf_labels_2)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             task_labels=task_labels)

        # Update the ema model
        self.global_step += 1
        if torch.rand(1) < self.ema_update_freq:
            self.update_ema_model_variables()

        return loss.item(), loss_1.item()
