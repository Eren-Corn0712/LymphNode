import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from toolkit.utils import LOGGER
from typing import Dict


class DINOLoss(nn.Module):
    def __init__(
            self,
            out_dim,
            ncrops,
            warmup_teacher_temp,
            teacher_temp,
            warmup_teacher_temp_epochs,
            nepochs,
            student_temp=0.1,
            center_momentum=0.9
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch, targets_mixup):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = torch.zeros(1, device=teacher_out[0].device)
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                if targets_mixup:
                    loss = -torch.sum(targets_mixup[v] * torch.mm(q, F.log_softmax(student_out[v], dim=-1).t()), dim=-1)
                else:
                    loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        return (
            total_loss.sum(),
            {
                "global_loss": total_loss.item()
            }
        )

    @torch.inference_mode()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if torch.distributed.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        else:
            batch_center = torch.mean(batch_center, dim=0, keepdim=True)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DDINOLoss(nn.Module):
    def __init__(
            self,
            out_dim,
            ncrops,
            warmup_teacher_temp,
            teacher_temp,
            warmup_teacher_temp_epochs,
            nepochs,
            student_temp=0.1,
            center_momentum=0.9
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center_grid", torch.zeros(1, out_dim))

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(
            self,
            s_cls_out,
            s_region_out,
            s_fea,
            s_npatch,
            t_cls_out,
            t_region_out,
            t_fea,
            t_npatch,
            epoch
    ):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        t_cls = F.softmax((t_cls_out - self.center) / temp, dim=-1)
        t_cls = t_cls.detach().chunk(2)

        t_region = F.softmax((t_region_out - self.center_grid) / temp, dim=-1)
        t_region = t_region.detach().chunk(2)
        t_fea = t_fea.chunk(2)

        num_patches = t_npatch[0]  # num of patches in the first view
        batch_size = t_region[0].shape[0] // num_patches  # batch size,

        # student sharpening
        s_cls = s_cls_out / self.student_temp
        s_cls = s_cls.chunk(self.ncrops)

        s_region = s_region_out / self.student_temp
        s_split_size = [s_npatch[0]] * 2 + [s_npatch[1]] * (self.ncrops - 2)

        s_split_size_bs = [i * batch_size for i in s_split_size]

        s_region = torch.split(s_region, s_split_size_bs, dim=0)
        s_fea = torch.split(s_fea, s_split_size_bs, dim=0)

        total_loss = torch.zeros(2, device=s_cls[0].device)
        n_loss_terms = 0
        for iq, q in enumerate(t_cls):
            for v in range(len(s_cls)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue

                # view level prediction loss
                loss = 0.5 * torch.sum(-q * F.log_softmax(s_cls[v], dim=-1), dim=-1)
                total_loss[0] += loss.mean()

                # region level prediction loss
                s_region_cur = s_region[v].view(batch_size, s_split_size[v], -1)  # B x T_s x K
                s_fea_cur = s_fea[v].view(batch_size, s_split_size[v], -1)  # B x T_s x P

                t_region_cur = t_region[iq].view(batch_size, num_patches, -1)  # B x T_t x K
                t_fea_cur = t_fea[iq].view(batch_size, num_patches, -1)  # B x T_t x P

                # similarity matrix between two sets of region features
                region_sim_matrix = torch.matmul(F.normalize(s_fea_cur, p=2, dim=-1),
                                                 F.normalize(t_fea_cur, p=2, dim=-1).permute(0, 2, 1))  # B x T_s x T_t

                # B x T_s; collect the argmax index in teacher for a given student feature
                region_sim_ind = region_sim_matrix.max(dim=2)[1]

                # B x T_s x K (index matrix: B, T_s, 1)
                t_indexed_region = torch.gather(
                    t_region_cur, 1, region_sim_ind.unsqueeze(2).expand(-1, -1, t_region_cur.size(2)))

                # B x T_s x K --> B
                loss_grid = torch.sum(- t_indexed_region * F.log_softmax(s_region_cur, dim=-1), dim=[-1]).mean(-1)

                total_loss[1] += (0.5 * loss_grid).mean()
                n_loss_terms += 1

        # global local
        total_loss[0] /= n_loss_terms
        total_loss[1] /= n_loss_terms

        self.update_center(t_cls_out, t_region_out)

        return (
            total_loss.sum(),
            {
                "global_loss": total_loss[0].item(),
                "local_loss": total_loss[1].item()
            }
        )

    @torch.inference_mode()
    def update_center(self, teacher_output, teacher_grid_output):
        """
        Update center used for teacher output.
        """

        # view level center update
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        else:
            batch_center = torch.mean(batch_center, dim=0, keepdim=True)

        # region level center update
        batch_grid_center = torch.sum(teacher_grid_output, dim=0, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(batch_grid_center)
            batch_grid_center = batch_grid_center / (len(teacher_grid_output) * dist.get_world_size())
        else:
            batch_grid_center = torch.mean(batch_grid_center, dim=0, keepdim=True)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        self.center_grid = self.center_grid * self.center_momentum + batch_grid_center * (1 - self.center_momentum)


class MixDDINOLoss(nn.Module):
    def __init__(
            self,
            out_dim,
            ncrops,
            warmup_teacher_temp,
            teacher_temp,
            warmup_teacher_temp_epochs,
            nepochs,
            student_temp=0.1,
            center_momentum=0.9
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center_grid", torch.zeros(1, out_dim))

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(
            self,
            s_mix_region_out,
            s_fea,
            s_npatch,
            t_mix_region_out,
            t_fea,
            t_npatch,
            epoch
    ):
        # teacher centering and sharpening
        teacher_temp = self.teacher_temp_schedule[epoch]
        t_region = F.softmax((t_mix_region_out - self.center_grid) / teacher_temp, dim=-1)
        t_region = t_region.detach().chunk(2)
        t_fea = t_fea.chunk(2)

        num_patches = t_npatch[0]  # num of patches in the first view
        batch_size = t_region[0].shape[0] // num_patches  # batch size

        # student sharpening
        s_region = s_mix_region_out / self.student_temp
        s_split_size = [s_npatch[0]] * 2 + [s_npatch[1]] * (self.ncrops - 2)  # crop size 49 or 9

        s_split_size_bs = [i * batch_size for i in s_split_size]

        s_region = torch.split(s_region, s_split_size_bs, dim=0)
        s_fea = torch.split(s_fea, s_split_size_bs, dim=0)

        total_loss = torch.zeros(1, device=s_fea[0].device)
        n_loss_terms = 0
        for iq, q in enumerate(t_region):
            for iv, v in enumerate(s_region):
                if iq == iv:
                    continue
                t_region_cur = q.view(batch_size, num_patches, -1)  # B x T_t x K
                t_fea_cur = t_fea[iq].view(batch_size, num_patches, -1)  # B x T_t x P

                s_region_cur = v.view(batch_size, s_split_size[iv], -1)  # B x T_s x K
                s_fea_cur = s_fea[iv].view(batch_size, s_split_size[iv], -1)  # B x T_s x P

                # similarity matrix between two sets of region features
                region_sim_matrix = torch.matmul(F.normalize(s_fea_cur, p=2, dim=-1),
                                                 F.normalize(t_fea_cur, p=2, dim=-1).permute(0, 2, 1))  # B x T_s x T_t

                # B x T_s; collect the argmax index in teacher for a given student feature
                region_sim_ind = region_sim_matrix.max(dim=2)[1]

                # B x T_s x K (index matrix: B, T_s, 1)
                t_indexed_region = torch.gather(
                    t_region_cur, 1, region_sim_ind.unsqueeze(2).expand(-1, -1, t_region_cur.size(2)))

                # B x T_s x K --> B
                loss_grid = torch.sum(- t_indexed_region * F.log_softmax(s_region_cur, dim=-1), dim=[-1]).mean(-1)

                total_loss[0] += loss_grid.mean()
                n_loss_terms += 1

        total_loss[0] /= n_loss_terms
        self.update_center(teacher_grid_output=t_mix_region_out)
        return (
            total_loss.sum(),
            {
                "mix_loss": total_loss.item()
            }
        )

    @torch.inference_mode()
    def update_center(self, teacher_grid_output):
        """
        Update center used for teacher output.
        """

        # region level center update
        batch_grid_center = torch.sum(teacher_grid_output, dim=0, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(batch_grid_center)
            batch_grid_center = batch_grid_center / (len(teacher_grid_output) * dist.get_world_size())
        else:
            batch_grid_center = torch.mean(batch_grid_center, dim=0, keepdim=True)

        self.center_grid = self.center_grid * self.center_momentum + batch_grid_center * (1 - self.center_momentum)


class CARELoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.mse_loss = nn.MSELoss()

    def forward(self, s_attn_out, t_attn_out, epoch):
        teacher_temp = self.teacher_temp_schedule[epoch]
        t_attn_out = t_attn_out.detach().chunk(2)
        s_attn_out = s_attn_out.chunk(2)

        total_loss = torch.zeros(1, device=s_attn_out[0].device)
        n_loss_terms = 0
        for t_idx, t_attn in enumerate(t_attn_out):
            for s_idx, s_attn in enumerate(s_attn_out):
                if t_idx == s_idx:
                    continue
                loss = self.mse_loss(t_attn, s_attn)
                total_loss[0] += loss
                n_loss_terms += 1

        total_loss[0] = total_loss[0] / n_loss_terms

        return (
            total_loss.sum(),
            {"attn_loss": total_loss[0].item()}
        )


def build_loss(args, device) -> Dict:
    criterion = {}

    if args.use_head_prediction and args.use_dense_prediction is not True:
        # Both view and region level tasks are considered
        criterion["ddino_loss"] = DDINOLoss(
            args.out_dim,
            args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs,
        ).to(device)

    if args.use_dense_prediction and args.use_head_prediction is not True:
        # Only view level task is considered
        criterion["dino_loss"] = DINOLoss(
            args.out_dim,
            args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs,
        ).to(device)

    if args.use_mix_prediction:
        criterion["mix_ddino_loss"] = MixDDINOLoss(
            args.out_dim,
            args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs,
        ).to(device)

    s = " ".join(v.__class__.__name__ for k, v in criterion.items())
    LOGGER.info(f"Criterion : {s}")
    return criterion
