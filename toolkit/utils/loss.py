import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from toolkit.utils import LOGGER
from typing import Dict
from scipy.optimize import linear_sum_assignment


@torch.inference_mode()
def hungarian(similarity):
    batch_size = similarity.shape[0]  # B
    # convert it to cost_matrix

    cost_matrix = similarity - similarity.amax((-1, -2)).shape
    cost_matrix = cost_matrix.cpu()

    batch_row_ind, batch_col_ind = [], []
    for i in range(batch_size):
        row_ind, col_ind = linear_sum_assignment(cost_matrix[i])
        batch_row_ind.append(torch.from_numpy(row_ind)[None, ...])
        batch_col_ind.append(torch.from_numpy(col_ind)[None, ...])

    batch_row_ind = torch.cat(batch_row_ind, 0).to(similarity.device)
    batch_col_ind = torch.cat(batch_col_ind, 0).to(similarity.device)
    return batch_row_ind, batch_col_ind


class BaseDINOLoss(nn.Module):
    def __init__(
            self, out_dim,
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


class DINOLoss(BaseDINOLoss):
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
        super().__init__(
            out_dim,
            ncrops,
            warmup_teacher_temp,
            teacher_temp,
            warmup_teacher_temp_epochs,
            nepochs,
            student_temp,
            center_momentum)

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


class TransformerLoss(nn.Module):
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
        self.register_buffer("center_grid", torch.zeros(1, 49, out_dim))

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self,
                s_trans_output,
                s_fea,
                s_npatch,
                t_trans_output,
                t_fea,
                t_npatch,
                epoch,
                ):
        teacher_temp = self.teacher_temp_schedule[epoch]

        t_region = F.softmax((t_trans_output[0] - self.center_grid) / teacher_temp, dim=-1)
        t_region = t_region.detach().chunk(2)
        t_fea = t_fea.chunk(2)

        num_patches = t_npatch[0]  # num of patches in the first view
        batch_size = t_region[0].shape[0]  # batch size

        # preprocess the student output
        s_global_region = s_trans_output[0] / self.student_temp
        s_global_region = s_global_region.chunk(2)
        s_local_region = s_trans_output[1] / self.student_temp
        s_local_region = s_local_region.chunk(int(self.ncrops - 2))

        s_split_size = [s_npatch[0]] * 2 + [s_npatch[1]] * (self.ncrops - 2)  # crop size 49 or 9
        s_split_size_bs = [i * batch_size for i in s_split_size]
        s_fea = torch.split(s_fea, s_split_size_bs, dim=0)

        total_loss = torch.zeros(1, device=t_trans_output[0].device)

        n_loss_terms = 0
        for iq, q in enumerate(t_region):
            for ik, v in enumerate(s_global_region + s_local_region):
                if iq == ik:
                    continue

                t_fea_cur = t_fea[iq].view(batch_size, num_patches, -1)  # B x T_t x P
                s_fea_cur = s_fea[ik].view(batch_size, s_split_size[ik], -1)

                # similarity matrix between two sets of region features
                region_sim_matrix = torch.matmul(F.normalize(s_fea_cur, p=2, dim=-1),
                                                 F.normalize(t_fea_cur, p=2, dim=-1).permute(0, 2, 1))
                # B x T_s x T_t
                s_index, t_index = hungarian(region_sim_matrix)

                t_indexed_region = torch.gather(q, 1, t_index.unsqueeze(2).expand(-1, -1, q.size(2)))
                s_indexed_region = torch.gather(v, 1, s_index.unsqueeze(2).expand(-1, -1, v.size(2)))

                loss_grid = torch.sum(
                    - t_indexed_region * F.log_softmax(s_indexed_region, dim=-1), dim=-1).mean(-1)

                total_loss += loss_grid.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms

        self.update_center(t_trans_output[0])

        return (
            total_loss.sum(),
            {
                "trans_loss": total_loss.item()
            }
        )

    @torch.no_grad()
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


class MultiDDINOLoss(nn.Module):
    def __init__(
            self,
            out_dim,
            ncrops,
            warmup_teacher_temp,
            teacher_temp,
            warmup_teacher_temp_epochs,
            nepochs,
            student_temp=0.1,
            center_momentum=0.9,
            mode="similarity"
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        for i in range(4):
            self.register_buffer(f"center_grid_{i}", torch.zeros(1, out_dim))

        self.mode = mode
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(
            self,
            s_multi_level_region_out,
            s_multi_level_fea,
            s_multi_level_npatch,
            t_multi_level_region_out,
            t_multi_level_fea,
            t_multi_level_npatch,
            epoch
    ):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        nl = len(s_multi_level_region_out)

        teacher_temp = self.teacher_temp_schedule[epoch]
        layer_total_loss = torch.zeros(nl).to(s_multi_level_fea[0].device)

        for i in range(nl):
            center_grid = getattr(self, f"center_grid_{i}")
            t_region_out = t_multi_level_region_out[i]
            t_fea = t_multi_level_fea[i]
            t_npatch = t_multi_level_npatch[i]

            t_region = F.softmax((t_region_out - center_grid) / teacher_temp, dim=-1)
            t_region = t_region.detach().chunk(2)
            t_fea = t_fea.chunk(2)

            num_patches = t_npatch[0]  # num of patches in the first view
            batch_size = t_region[0].shape[0] // num_patches  # batch size,

            s_region_out = s_multi_level_region_out[i]
            s_npatch = s_multi_level_npatch[i]
            s_fea = s_multi_level_fea[i]

            s_region = s_region_out / self.student_temp
            s_split_size = [s_npatch[0]] * 2 + [s_npatch[1]] * (self.ncrops - 2)

            s_split_size_bs = [i * batch_size for i in s_split_size]

            s_region = torch.split(s_region, s_split_size_bs, dim=0)
            s_fea = torch.split(s_fea, s_split_size_bs, dim=0)

            total_loss = torch.zeros(1, device=t_fea[0].device)
            n_loss_terms = 0
            for iq in range(len(t_region)):
                for v in range(len(s_region)):
                    if v == iq:
                        # we skip cases where student and teacher operate on the same view
                        continue
                    # region level prediction loss
                    s_region_cur = s_region[v].view(batch_size, s_split_size[v], -1)  # B x T_s x K
                    s_fea_cur = s_fea[v].view(batch_size, s_split_size[v], -1)  # B x T_s x P

                    t_region_cur = t_region[iq].view(batch_size, num_patches, -1)  # B x T_t x K
                    t_fea_cur = t_fea[iq].view(batch_size, num_patches, -1)  # B x T_t x P

                    # similarity matrix between two sets of region features
                    region_sim_matrix = torch.matmul(F.normalize(s_fea_cur, p=2, dim=-1),
                                                     F.normalize(t_fea_cur, p=2, dim=-1).permute(0, 2,
                                                                                                 1))  # B x T_s x T_t


                    if self.mode == "similarity":
                        # B x T_s; collect the argmax index in teacher for a given student feature
                        region_sim_ind = region_sim_matrix.max(dim=2)[1]

                        # B x T_s x K (index matrix: B, T_s, 1)
                        t_indexed_region = torch.gather(
                            t_region_cur, 1, region_sim_ind.unsqueeze(2).expand(-1, -1, t_region_cur.size(2)))

                        # B x T_s x K --> B
                        loss_grid = torch.sum(- t_indexed_region * F.log_softmax(s_region_cur, dim=-1), dim=[-1]).mean(-1)
                        total_loss[0] += loss_grid.mean()

                    else:
                        s_index, t_index = hungarian(region_sim_matrix)

                        t_indexed_region = torch.gather(q, 1, t_index.unsqueeze(2).expand(-1, -1, q.size(2)))
                        s_indexed_region = torch.gather(v, 1, s_index.unsqueeze(2).expand(-1, -1, v.size(2)))

                        loss_grid = torch.sum(
                            - t_indexed_region * F.log_softmax(s_indexed_region, dim=-1), dim=-1).mean(-1)
                        total_loss[0] += loss_grid.mean()

                    n_loss_terms += 1

            total_loss /= n_loss_terms
            layer_total_loss[i] = total_loss

            center_grid = self.update_center(center=center_grid, teacher_grid_output=t_region_out)
            setattr(self, f"center_grid_{i}", center_grid)

        layer_total_loss = layer_total_loss / nl

        return (
            layer_total_loss.sum(),
            {
                f"{i + 1}_dino_loss": l for i, l in enumerate(layer_total_loss)
            }
        )

    @torch.inference_mode()
    def update_center(self, center, teacher_grid_output):
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

        # ema update
        center = center * self.center_momentum + batch_grid_center * (1 - self.center_momentum)
        return center


def build_loss(args, device) -> Dict:
    criterion = {}

    if args.use_dense_prediction is True and args.use_head_prediction is False:
        # Both view and region level tasks are considered
        criterion["ddino_loss"] = DDINOLoss(
            args.out_dim,
            args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs,
        ).to(device)

    if args.use_dense_prediction is False and args.use_head_prediction is True:
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

    if args.use_trans_prediction:
        criterion["trans_loss"] = TransformerLoss(
            args.out_dim,
            args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs,
        ).to(device)
    if args.use_multi_level:
        criterion["multi_level_loss"] = MultiDDINOLoss(
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
