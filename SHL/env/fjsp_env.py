import sys
import gym
import torch
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from env.load_data import load_fjs, nums_detec



@dataclass
class EnvState:
    opes_appertain_batch: torch.Tensor = None
    ope_pre_adj_batch: torch.Tensor = None
    ope_sub_adj_batch: torch.Tensor = None
    end_ope_biases_batch: torch.Tensor = None
    nums_opes_batch: torch.Tensor = None
    batch_idxes: torch.Tensor = None
    feat_opes_batch: torch.Tensor = None
    feat_mas_batch: torch.Tensor = None
    proc_times_batch: torch.Tensor = None
    ope_ma_adj_batch: torch.Tensor = None
    time_batch:  torch.Tensor = None
    adap_feat_opes_batch: torch.Tensor = None
    adap_feat_mas_batch: torch.Tensor = None
    ope_ope_adj_batch: torch.Tensor = None
    mask_job_procing_batch: torch.Tensor = None
    mask_job_finish_batch: torch.Tensor = None
    mask_ma_procing_batch: torch.Tensor = None
    ope_step_batch: torch.Tensor = None


    def update(self, batch_idxes, feat_opes_batch, feat_mas_batch, proc_times_batch, ope_ma_adj_batch,
               adap_feat_opes_batch, adap_feat_mas_batch, ope_ope_adj_batch,
               mask_job_procing_batch, mask_job_finish_batch, mask_ma_procing_batch, ope_step_batch, time):

        self.batch_idxes = batch_idxes
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.proc_times_batch = proc_times_batch
        self.ope_ma_adj_batch = ope_ma_adj_batch
        self.adap_feat_opes_batch = adap_feat_opes_batch
        self.adap_feat_mas_batch = adap_feat_mas_batch
        self.ope_ope_adj_batch = ope_ope_adj_batch

        self.mask_job_procing_batch = mask_job_procing_batch
        self.mask_job_finish_batch = mask_job_finish_batch
        self.mask_ma_procing_batch = mask_ma_procing_batch
        self.ope_step_batch = ope_step_batch
        self.time_batch = time

def convert_feat_job_2_ope(feat_job_batch, opes_appertain_batch):
    return feat_job_batch.gather(1, opes_appertain_batch)

class FJSPEnv(gym.Env):
    def __init__(self, case, env_paras, data_source='case'):

        self.batch_size = env_paras["batch_size"]
        self.num_jobs = env_paras["num_jobs"]
        self.num_mas = env_paras["num_mas"]
        self.paras = env_paras
        self.device = env_paras["device"]
        num_data = 9
        tensors = [[] for _ in range(num_data)]
        self.num_opes = 0
        lines = []
        if data_source=='case':
            for i in range(self.batch_size):
                lines.append(case.get_case(i)[0])
                num_jobs, num_mas, num_opes = nums_detec(lines[i])
                self.num_opes = max(self.num_opes, num_opes)
        else:
            for i in range(self.batch_size):
                with open(case[i]) as file_object:
                    line = file_object.readlines()
                    lines.append(line)
                num_jobs, num_mas, num_opes = nums_detec(lines[i])
                self.num_opes = max(self.num_opes, num_opes)
        for i in range(self.batch_size):
            load_data = load_fjs(lines[i], num_mas, self.num_opes)
            for j in range(num_data):
                tensors[j].append(load_data[j])

        self.proc_times_batch = torch.stack(tensors[0], dim=0)
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()
        self.cal_cumul_adj_batch = torch.stack(tensors[7], dim=0).float()
        self.ope_ope_adj_batch = torch.stack(tensors[8], dim=0).long()
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)
        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0)
        self.opes_appertain_batch = torch.stack(tensors[4], dim=0).long()
        self.num_ope_biases_batch = torch.stack(tensors[5], dim=0).long()
        self.nums_ope_batch = torch.stack(tensors[6], dim=0).long()
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)
        self.batch_idxes = torch.arange(self.batch_size)
        self.time = torch.zeros(self.batch_size)
        self.N = torch.zeros(self.batch_size).int()
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        # Generate raw feature vectors
        feat_opes_batch = torch.zeros(size=(self.batch_size, self.paras["ope_feat_dim"], self.num_opes))
        feat_mas_batch = torch.zeros(size=(self.batch_size, self.paras["ma_feat_dim"], num_mas))

        feat_opes_batch[:, 1, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)
        feat_opes_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9)
        feat_opes_batch[:, 3, :] = convert_feat_job_2_ope(self.nums_ope_batch, self.opes_appertain_batch)
        feat_opes_batch[:, 5, :] = torch.bmm(feat_opes_batch[:, 2, :].unsqueeze(1),
                                             self.cal_cumul_adj_batch).squeeze()
        end_time_batch = (feat_opes_batch[:, 5, :] +
                          feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch)
        feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch)
        feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1)
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.adap_feat_opes_batch = feat_opes_batch[:,[0,1, 2, 5],:]
        self.adap_feat_mas_batch =  feat_mas_batch[:,[0, 2],:]
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, num_mas), dtype=torch.bool, fill_value=False)
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        self.schedules_batch[:, :, 2] = feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :]
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]  # shape: (batch_size)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)  # shape: (batch_size) #各批次的完成情况

        self.state = EnvState(batch_idxes=self.batch_idxes,
                              feat_opes_batch=self.feat_opes_batch, feat_mas_batch=self.feat_mas_batch,
                              adap_feat_opes_batch=self.adap_feat_opes_batch, adap_feat_mas_batch=self.adap_feat_mas_batch,
                              ope_ope_adj_batch=self.ope_ope_adj_batch,
                              proc_times_batch=self.proc_times_batch, ope_ma_adj_batch=self.ope_ma_adj_batch,
                              ope_pre_adj_batch=self.ope_pre_adj_batch, ope_sub_adj_batch=self.ope_sub_adj_batch,
                              mask_job_procing_batch=self.mask_job_procing_batch,
                              mask_job_finish_batch=self.mask_job_finish_batch,
                              mask_ma_procing_batch=self.mask_ma_procing_batch,
                              opes_appertain_batch=self.opes_appertain_batch,
                              ope_step_batch=self.ope_step_batch,
                              end_ope_biases_batch=self.end_ope_biases_batch,
                              time_batch=self.time, nums_opes_batch=self.nums_opes)
        self.old_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self.old_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)
        self.old_cal_cumul_adj_batch = copy.deepcopy(self.cal_cumul_adj_batch)
        self.old_feat_opes_batch = copy.deepcopy(self.feat_opes_batch)
        self.old_feat_mas_batch = copy.deepcopy(self.feat_mas_batch)
        self.old_state = copy.deepcopy(self.state)
        self.old_adap_feat_opes_batch = copy.deepcopy(self.adap_feat_opes_batch)
        self.old_adap_feat_mas_batch = copy.deepcopy(self.adap_feat_mas_batch)
        self.old_ope_ope_adj_batch = copy.deepcopy(self.ope_ope_adj_batch)

    def step(self, actions):
        opes = actions[0, :]
        mas = actions[1, :]
        jobs = actions[2, :]
        self.N += 1
        remain_ope_ma_adj = torch.zeros(size=(self.batch_size, self.num_mas), dtype=torch.int64)
        remain_ope_ma_adj[self.batch_idxes, mas] = 1
        self.ope_ma_adj_batch[self.batch_idxes, opes] = remain_ope_ma_adj[self.batch_idxes, :]
        self.proc_times_batch *= self.ope_ma_adj_batch
        proc_times = self.proc_times_batch[self.batch_idxes, opes, mas]
        self.feat_opes_batch[self.batch_idxes, :3, opes] = torch.stack(
            (torch.ones(self.batch_idxes.size(0), dtype=torch.float),
             torch.ones(self.batch_idxes.size(0), dtype=torch.float),
             proc_times), dim=1)
        last_opes = torch.where(opes - 1 < self.num_ope_biases_batch[self.batch_idxes, jobs],self.num_opes - 1, opes - 1)
        self.cal_cumul_adj_batch[self.batch_idxes, last_opes, :] = 0
        start_ope = self.num_ope_biases_batch[self.batch_idxes, jobs]
        end_ope = self.end_ope_biases_batch[self.batch_idxes, jobs]
        for i in range(self.batch_idxes.size(0)):
            self.feat_opes_batch[self.batch_idxes[i], 3, start_ope[i]:end_ope[i] + 1] -= 1
        self.feat_opes_batch[self.batch_idxes, 5, opes] = self.time[self.batch_idxes]
        is_scheduled = self.feat_opes_batch[self.batch_idxes, 0, :]
        mean_proc_time = self.feat_opes_batch[self.batch_idxes, 2, :]
        start_times = self.feat_opes_batch[self.batch_idxes, 5,:] * is_scheduled
        un_scheduled = 1 - is_scheduled
        estimate_times = torch.bmm((start_times + mean_proc_time).unsqueeze(1),
                                   self.cal_cumul_adj_batch[self.batch_idxes, :, :]).squeeze() \
                         * un_scheduled
        self.feat_opes_batch[self.batch_idxes, 5, :] = start_times + estimate_times
        end_time_batch = ((self.feat_opes_batch[self.batch_idxes, 5, :] +
                           self.feat_opes_batch[self.batch_idxes, 2, :]).gather(1,self.end_ope_biases_batch[ self.batch_idxes,:]))
        self.feat_opes_batch[self.batch_idxes, 4, :] = convert_feat_job_2_ope(end_time_batch,self.opes_appertain_batch[self.batch_idxes, :])


        self.schedules_batch[self.batch_idxes, opes, :2] = torch.stack((torch.ones(self.batch_idxes.size(0)), mas), dim=1)
        self.schedules_batch[self.batch_idxes, :, 2] = self.feat_opes_batch[self.batch_idxes, 5,:]
        self.schedules_batch[self.batch_idxes, :, 3] = self.feat_opes_batch[self.batch_idxes, 5,:] + \
                                                       self.feat_opes_batch[self.batch_idxes, 2,:]
        self.machines_batch[self.batch_idxes, mas, 0] = torch.zeros(self.batch_idxes.size(0))
        self.machines_batch[self.batch_idxes, mas, 1] = self.time[self.batch_idxes] + proc_times
        self.machines_batch[self.batch_idxes, mas, 2] += proc_times
        self.machines_batch[self.batch_idxes, mas, 3] = jobs.float()
        self.feat_mas_batch[self.batch_idxes, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch[self.batch_idxes, :, :], dim=1).float()
        self.feat_mas_batch[self.batch_idxes, 1, mas] = self.time[self.batch_idxes] + proc_times
        utiliz = self.machines_batch[self.batch_idxes, :, 2]
        cur_time = self.time[self.batch_idxes, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[self.batch_idxes, None] + 1e-9)
        self.feat_mas_batch[self.batch_idxes, 2, :] = utiliz
        self.ope_step_batch[self.batch_idxes, jobs] += 1
        self.mask_job_procing_batch[
            self.batch_idxes, jobs] = True
        self.mask_ma_procing_batch[
            self.batch_idxes, mas] = True
        self.mask_job_finish_batch = torch.where(
            self.ope_step_batch == self.end_ope_biases_batch + 1,
            True,
            self.mask_job_finish_batch)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.done = self.done_batch.all()

        max = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.reward_batch = self.makespan_batch - max
        self.makespan_batch = max
        flag_trans_2_next_time = self.if_no_eligible()
        while ~((~((flag_trans_2_next_time == 0) & (~self.done_batch))).all()):
            self.next_time(flag_trans_2_next_time)
            flag_trans_2_next_time = self.if_no_eligible()
        mask_finish = (self.N + 1) <= self.nums_opes
        if ~(mask_finish.all()):
            self.batch_idxes = torch.arange(self.batch_size)[mask_finish]
        info = []
        self.adap_feat_opes_batch = self.feat_opes_batch[:, [0,1, 2, 5],:]
        self.adap_feat_mas_batch = self.feat_mas_batch[:, [0, 2],:]
        self.state.update(self.batch_idxes, self.feat_opes_batch, self.feat_mas_batch,self.proc_times_batch,self.ope_ma_adj_batch,
                            self.adap_feat_opes_batch, self.adap_feat_mas_batch, self.ope_ope_adj_batch,
                            self.mask_job_procing_batch,self.mask_job_finish_batch, self.mask_ma_procing_batch,
                            self.ope_step_batch, self.time)
        return self.state, self.reward_batch, self.done_batch, info
    def if_no_eligible(self):
        ope_step_batch = torch.where(self.ope_step_batch > self.end_ope_biases_batch,
                                     self.end_ope_biases_batch, self.ope_step_batch)
        op_proc_time = self.proc_times_batch.gather(1, ope_step_batch.unsqueeze(-1).expand(-1, -1,
                                                                                        self.proc_times_batch.size(2)))
        ma_eligible = ~self.mask_ma_procing_batch.unsqueeze(1).expand_as(op_proc_time)
        job_eligible = ~(self.mask_job_procing_batch + self.mask_job_finish_batch)[:, :, None].expand_as(
            op_proc_time)
        flag_trans_2_next_time = torch.sum(torch.where(ma_eligible & job_eligible, op_proc_time.double(), 0.0).transpose(1, 2),
                                           dim=[1, 2])
        return flag_trans_2_next_time

    def next_time(self, flag_trans_2_next_time):
        flag_need_trans = (flag_trans_2_next_time==0) & (~self.done_batch)
        a = self.machines_batch[:, :, 1]
        b = torch.where(a > self.time[:, None], a, torch.max(self.feat_opes_batch[:, 4, :]) + 1.0)
        c = torch.min(b, dim=1)[0]
        d = torch.where((a == c[:, None]) & (self.machines_batch[:, :, 0] == 0) & flag_need_trans[:, None], True, False)
        e = torch.where(flag_need_trans, c, self.time)
        self.time = e
        aa = self.machines_batch.transpose(1, 2)
        aa[d, 0] = 1
        self.machines_batch = aa.transpose(1, 2)
        utiliz = self.machines_batch[:, :, 2]
        cur_time = self.time[:, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[:, None] + 1e-5)
        self.feat_mas_batch[:, 2, :] = utiliz
        jobs = torch.where(d, self.machines_batch[:, :, 3].double(), -1.0).float()
        jobs_index = np.argwhere(jobs.cpu() >= 0).to(self.device)
        job_idxes = jobs[jobs_index[0], jobs_index[1]].long()
        batch_idxes = jobs_index[0]
        self.mask_job_procing_batch[batch_idxes, job_idxes] = False
        self.mask_ma_procing_batch[d] = False
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)

    def reset(self):
        self.proc_times_batch = copy.deepcopy(self.old_proc_times_batch)
        self.ope_ma_adj_batch = copy.deepcopy(self.old_ope_ma_adj_batch)
        self.cal_cumul_adj_batch = copy.deepcopy(self.old_cal_cumul_adj_batch)
        self.feat_opes_batch = copy.deepcopy(self.old_feat_opes_batch)
        self.feat_mas_batch = copy.deepcopy(self.old_feat_mas_batch)
        self.state = copy.deepcopy(self.old_state)
        self.adap_feat_opes_batch= copy.deepcopy(self.old_adap_feat_opes_batch)
        self.adap_feat_mas_batch = copy.deepcopy(self.old_adap_feat_mas_batch)
        self.ope_ope_adj_batch = copy.deepcopy(self.old_ope_ope_adj_batch)
        self.batch_idxes = torch.arange(self.batch_size)
        self.time = torch.zeros(self.batch_size)
        self.N = torch.zeros(self.batch_size)
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool, fill_value=False)
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        self.schedules_batch[:, :, 2] = self.feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = self.feat_opes_batch[:, 5, :] + self.feat_opes_batch[:, 2, :]
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))
        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        return self.state

    def render(self, mode='human'):
        num_jobs = self.num_jobs
        num_mas = self.num_mas
        min_value, min_index = torch.min(self.makespan_batch, dim=0)
        color = []
        color += ['#' + ''.join([random.choice("56789ABCDEF") for _ in range(6)]) for _ in range(num_jobs)]
        for batch_id in range(min_index, min_index + 1):
            schedules = self.schedules_batch[batch_id].to('cpu')

            plt.rcParams["font.family"] = ["Times New Roman"]
            plt.rcParams["lines.linewidth"] = 0.5
            fig = plt.figure(figsize=(10, 6))
            axes = fig.add_axes([0.06, 0.1, 0.9, 0.85])
            axes.cla()
            axes.set_xlabel('Time')
            axes.set_ylabel('Machine')
            y_ticks_loc = list(range(num_mas, 0, -1))
            axes.set_yticks(y_ticks_loc)
            for i in range(int(self.nums_opes[batch_id])):
                id_ope = i
                idx_job, idx_ope = self.get_idx(id_ope, batch_id)
                id_machine = schedules[id_ope][1] + 1
                axes.barh(id_machine,
                          schedules[id_ope][3] - schedules[id_ope][2],
                          left=schedules[id_ope][2],
                          color=color[idx_job],
                          height=0.4,   edgecolor='black',  linewidth=0.3)
                axes.text(schedules[id_ope][2] + (schedules[id_ope][3] - schedules[id_ope][2]) / 8, id_machine - 0.05,
                          '{0}'.format(idx_job + 1), size=6,rotation=0)
            makespan = torch.max(schedules[:, 3]).item()
            axes.axvline(x=makespan, c='k', ls='--', lw=0.5)
            axes.text(makespan, 0.7, '{0}'.format(makespan), color='#FF0000', size=10)
            filename = '{0}/{1}.png'.format('.//data', min_value)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        return

    def get_idx(self, id_ope, batch_id):
        idx_job = max([idx for (idx, val) in enumerate(self.num_ope_biases_batch[batch_id]) if id_ope >= val])
        idx_ope = id_ope - self.num_ope_biases_batch[batch_id][idx_job]
        return idx_job, idx_ope

    def validate_gantt(self):
        ma_gantt_batch = [[[] for _ in range(self.num_mas)] for __ in range(self.batch_size)]
        for batch_id, schedules in enumerate(self.schedules_batch):
            for i in range(int(self.nums_opes[batch_id])):
                step = schedules[i]
                ma_gantt_batch[batch_id][int(step[1])].append([i, step[2].item(), step[3].item()])
        proc_time_batch = self.proc_times_batch
        flag_proc_time = 0
        flag_ma_overlap = 0
        flag = 0
        for k in range(self.batch_size):
            ma_gantt = ma_gantt_batch[k]
            proc_time = proc_time_batch[k]
            for i in range(self.num_mas):
                ma_gantt[i].sort(key=lambda s: s[1])
                for j in range(len(ma_gantt[i])):
                    if (len(ma_gantt[i]) <= 1) or (j == len(ma_gantt[i])-1):
                        break
                    if ma_gantt[i][j][2]>ma_gantt[i][j+1][1]:
                        flag_ma_overlap += 1
                    if ma_gantt[i][j][2]-ma_gantt[i][j][1] != proc_time[ma_gantt[i][j][0]][i]:
                        flag_proc_time += 1
                    flag += 1
        flag_ope_overlap = 0
        for k in range(self.batch_size):
            schedule = self.schedules_batch[k]
            nums_ope = self.nums_ope_batch[k]
            num_ope_biases = self.num_ope_biases_batch[k]
            for i in range(self.num_jobs):
                if int(nums_ope[i]) <= 1:
                    continue
                for j in range(int(nums_ope[i]) - 1):
                    step = schedule[num_ope_biases[i]+j]
                    step_next = schedule[num_ope_biases[i]+j+1]
                    if step[3] > step_next[2]:
                        flag_ope_overlap += 1
        flag_unscheduled = 0
        for batch_id, schedules in enumerate(self.schedules_batch):
            count = 0
            for i in range(schedules.size(0)):
                if schedules[i][0]==1:
                    count += 1
            add = 0 if (count == self.nums_opes[batch_id]) else 1
            flag_unscheduled += add
        if flag_ma_overlap + flag_ope_overlap + flag_proc_time + flag_unscheduled != 0:
            return False, self.schedules_batch
        else:
            return True, self.schedules_batch
    def close(self):
        pass
