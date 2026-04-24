import copy
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from mlp import MLPCritic, MLPActor
from hnn_ope import HGTConv_ope
from hgnn import GAT
import random
class Memory:
    def __init__(self):
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.action_indexes = []
        
        self.ope_ma_adj = []
        self.ope_ope_adj = []
        self.batch_idxes = []
        self.raw_opes = []
        self.raw_mas = []
        self.proc_time = []
        self.jobs_gather = []
        self.eligible = []
        self.scale = []
        
    def clear_memory(self):
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.action_indexes[:]
        del self.ope_ma_adj[:]
        del self.ope_ope_adj[:]
        del self.batch_idxes[:]
        del self.raw_opes[:]
        del self.raw_mas[:]
        del self.proc_time[:]
        del self.jobs_gather[:]
        del self.eligible[:]
        del self.scale[:]

class HGNNScheduler(nn.Module):
    def __init__(self, model_paras):
        super(HGNNScheduler, self).__init__()
        self.device = model_paras["device"]
        self.in_dim = model_paras["in_dim"]
        self.out_dim = model_paras["out_dim"]
        self.actor_dim = model_paras["actor_in_dim"]
        self.critic_dim = model_paras["critic_in_dim"]
        self.n_latent_actor = model_paras["n_latent_actor"]
        self.n_latent_critic = model_paras["n_latent_critic"]
        self.n_hidden_actor = model_paras["n_hidden_actor"]
        self.n_hidden_critic = model_paras["n_hidden_critic"]
        self.action_dim = model_paras["action_dim"]
        self.get_machines = nn.ModuleList()
        self.get_machines.append(GAT((4, 2), self.out_dim, num_head=1))
        self.get_machines.append(GAT((self.out_dim, self.out_dim), self.out_dim,num_head=1))
        self.get_operations = nn.ModuleList()
        self.get_operations.append(HGTConv_ope(in_dim_list=[self.out_dim, 4], out_dim=self.out_dim, n_heads=2))
        self.get_operations.append(HGTConv_ope(in_dim_list=[self.out_dim, self.out_dim], out_dim=self.out_dim, n_heads=2))
        self.actor = MLPActor(self.n_hidden_actor, self.actor_dim, self.n_latent_actor, self.action_dim).to(self.device)
        self.critic = MLPCritic(self.n_hidden_critic, self.critic_dim, self.n_latent_critic, 1).to(self.device)
        self.cond_1 = nn.Sequential(
            nn.Linear(4, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 8, bias=False)
        )
        self.proj_N_h = nn.Sequential(
            nn.Linear(8, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 8, bias=False)
        )
    def forward(self):
        raise NotImplementedError

    def feature_normalize(self, data):
        return (data - torch.mean(data)) / ((data.std() + 1e-5))

    def get_normalized(self, raw_opes, raw_mas, proc_time, batch_idxes, nums_opes, flag_sample=False, flag_train=False):
        batch_size = batch_idxes.size(0)
        if not flag_sample and not flag_train:
            mean_opes = []
            std_opes = []
            for i in range(batch_size):
                mean_opes.append(torch.mean(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))
                std_opes.append(torch.std(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))

                proc_idxes =(proc_time[i] != 0)
                proc_values = torch.masked_select(proc_time[i], proc_idxes)
                proc_norm = self.feature_normalize(proc_values)
                proc_time[i] = torch.masked_scatter(proc_time[i], proc_idxes, proc_norm)
            mean_opes = torch.stack(mean_opes, dim=0)
            std_opes = torch.stack(std_opes, dim=0)
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)
            proc_time_norm = proc_time
        else:
            mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)
            std_opes = torch.std(raw_opes, dim=-2, keepdim=True)
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)
            proc_time_norm = self.feature_normalize(proc_time)
        return ((raw_opes - mean_opes) / (std_opes + 1e-5), (raw_mas - mean_mas) / (std_mas + 1e-5),
                proc_time_norm)

    def get_action_prob(self, state, memories, flag_sample=False, flag_train=False):

        batch_idxes = state.batch_idxes
        raw_opes = state.adap_feat_opes_batch.transpose(1, 2)[batch_idxes]
        raw_mas = state.adap_feat_mas_batch.transpose(1, 2)[batch_idxes]
        proc_time = state.proc_times_batch[batch_idxes]
        nums_opes = state.nums_opes_batch[batch_idxes]
        features = self.get_normalized(raw_opes, raw_mas, proc_time, batch_idxes, nums_opes, flag_sample, flag_train)
        norm_opes = (copy.deepcopy(features[0]))
        norm_mas = (copy.deepcopy(features[1]))
        norm_proc = (copy.deepcopy(features[2]))
        yi=torch.sum(state.feat_opes_batch[batch_idxes, 0, :], dim=1)
        y = torch.sum(torch.count_nonzero(state.ope_ma_adj_batch[batch_idxes, :, :], dim=2),dim=1)-yi
        q = nums_opes - yi
        a = torch.sum((~state.mask_job_finish_batch[batch_idxes, :]).to(torch.long) , dim=1)
        b = torch.sum((~state.mask_ma_procing_batch[batch_idxes, :]).to(torch.long), dim=1)
        num_job0 = torch.tensor((state.mask_job_finish_batch).size(-1))
        num_mac0 = torch.tensor(proc_time.size(-1))

        scale_input = torch.cat([y.unsqueeze(-1) / ((num_job0*num_mac0*(num_mac0+1)/2).int()),
                                q.unsqueeze(-1) / ((num_job0*num_mac0*1.2).int()),
                                 a.unsqueeze(-1) / num_job0,
                                 b.unsqueeze(-1) / num_mac0],dim=-1)
        for i in range(2):
            h_mas = self.get_machines[i](state.ope_ma_adj_batch,
                                         state.batch_idxes,
                                         features)
            features = (features[0], h_mas, features[2])

            h_opes = self.get_operations[i](ope_ma_adj_batch=state.ope_ma_adj_batch,
                                            ope_ope_adj_batch=state.ope_ope_adj_batch,
                                            batch_idxes=state.batch_idxes,
                                            feat=features)

            features = (h_opes, features[1], features[2])
        h_mas = (features[1])
        h_opes = (features[0])
        h_mas_pooled = h_mas.mean(dim=-2)
        if not flag_sample and not flag_train:
            h_opes_pooled = []
            for i in range(len(batch_idxes)):
                h_opes_pooled.append(torch.mean(h_opes[i, :nums_opes[i], :], dim=-2))
            h_opes_pooled = torch.stack(h_opes_pooled)
        else:
            h_opes_pooled = h_opes.mean(dim=-2)
        scale_emb     =  self.cond_1(scale_input)
        h_opes_pooled =  h_opes_pooled + self.proj_N_h(h_opes_pooled + scale_emb)
        h_mas_pooled  =  h_mas_pooled  + self.proj_N_h(h_mas_pooled + scale_emb)
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,state.end_ope_biases_batch, state.ope_step_batch)
        jobs_gather = ope_step_batch[..., :, None].expand(-1, -1, h_opes.size(-1))[batch_idxes]
        h_jobs = h_opes.gather(1, jobs_gather)
        eligible_proc = state.ope_ma_adj_batch[batch_idxes].gather(1,
                          ope_step_batch[..., :, None].expand(-1, -1, state.ope_ma_adj_batch.size(-1))[batch_idxes])
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, state.proc_times_batch.size(-1), -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)
        ma_eligible = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(h_jobs_padding[..., 0])
        job_eligible = ~(state.mask_job_procing_batch[batch_idxes] +
                         state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(h_jobs_padding[..., 0])
        eligible = job_eligible & ma_eligible & (eligible_proc == 1)
        if (~(eligible)).all():
            print("No eligible O-M pair!")
            return
        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
                              dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)
        mask = eligible.transpose(1, 2).flatten(1)
        scores = self.actor(h_actions).flatten(1)
        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)
        if flag_train == True:
            memories.ope_ma_adj.append(copy.deepcopy(state.ope_ma_adj_batch))
            memories.ope_ope_adj.append(copy.deepcopy(state.ope_ope_adj_batch))
            memories.batch_idxes.append(copy.deepcopy(state.batch_idxes))
            memories.raw_opes.append(copy.deepcopy(norm_opes))
            memories.raw_mas.append(copy.deepcopy(norm_mas))
            memories.proc_time.append(copy.deepcopy(norm_proc))
            memories.jobs_gather.append(copy.deepcopy(jobs_gather))
            memories.eligible.append(copy.deepcopy(eligible))
            memories.scale.append(copy.deepcopy(scale_input))

        return action_probs, ope_step_batch, h_pooled

    def act(self, state, memories, dones, flag_sample=True, flag_train=True):
        action_probs, ope_step_batch, h_pooled = self.get_action_prob(state, memories, flag_sample, flag_train=flag_train)
        a = 1 if flag_train else 0.3
        if random.random() < a:
            dist = Categorical(action_probs)
            action_indexes = dist.sample()
        else:
            action_indexes = action_probs.argmax(dim=1)
        mas = (action_indexes / state.mask_job_finish_batch.size(1)).long()
        jobs = (action_indexes % state.mask_job_finish_batch.size(1)).long()
        opes = ope_step_batch[state.batch_idxes, jobs]
        if flag_train == True:
            #memories.states.append(copy.deepcopy(state))
            memories.logprobs.append(dist.log_prob(action_indexes))
            memories.action_indexes.append(action_indexes)
        return torch.stack((opes, mas, jobs), dim=1).t()

    def evaluate(self, ope_ma_adj,ope_ope_adj, raw_opes, raw_mas, proc_time,jobs_gather, eligible, action_envs, scale_input,flag_sample=False):
        batch_idxes = torch.arange(0, ope_ma_adj.size(-3)).long()
        features = (raw_opes, raw_mas, proc_time)
        for i in range(2):
            h_mas = self.get_machines[i](ope_ma_adj, batch_idxes,features)
            features = (features[0], h_mas, features[2])
            h_opes = self.get_operations[i](ope_ma_adj_batch=ope_ma_adj,
                                            ope_ope_adj_batch=ope_ope_adj,
                                            batch_idxes=batch_idxes,
                                            feat=features)
            features = (h_opes, features[1], features[2])

        h_mas =features[1]
        h_opes = features[0]
        h_mas_pooled = h_mas.mean(dim=-2)
        h_opes_pooled = h_opes.mean(dim=-2)
        scale_emb = self.cond_1(scale_input)
        h_opes_pooled = h_opes_pooled + self.proj_N_h(h_opes_pooled + scale_emb)
        h_mas_pooled = h_mas_pooled + self.proj_N_h(h_mas_pooled + scale_emb)
        h_jobs = h_opes.gather(1, jobs_gather)
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, proc_time.size(-1), -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
                              dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)
        scores = self.actor(h_actions).flatten(1)
        mask = eligible.transpose(1, 2).flatten(1)
        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)
        state_values = self.critic(h_pooled)
        dist = Categorical(action_probs.squeeze())
        action_logprobs = dist.log_prob(action_envs)
        dist_entropys = dist.entropy()
        return action_logprobs, state_values.squeeze().double(), dist_entropys

class PPO:
    def __init__(self, model_paras, train_paras, num_envs=None):
        self.lr = train_paras["lr"]
        self.betas = train_paras["betas"]
        self.gamma = train_paras["gamma"]
        self.eps_clip = train_paras["eps_clip"]
        self.K_epochs = train_paras["K_epochs"]
        self.A_coeff = train_paras["A_coeff"]
        self.vf_coeff = train_paras["vf_coeff"]
        self.entropy_coeff = train_paras["entropy_coeff"]
        self.num_envs = num_envs
        self.device = model_paras["device"]

        self.policy = HGNNScheduler(model_paras).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.MseLoss = nn.MSELoss()

    def update(self, memory, env_paras, train_paras):
        device = env_paras["device"]
        minibatch_size = train_paras["minibatch_size"]
        old_ope_ma_adj = torch.stack(memory.ope_ma_adj, dim=0).transpose(0, 1).flatten(0,1)
        old_ope_ope_adj = torch.stack(memory.ope_ope_adj, dim=0).transpose(0, 1).flatten(0,  1)
        old_raw_opes = torch.stack(memory.raw_opes, dim=0).transpose(0, 1).flatten(0, 1)
        old_raw_mas = torch.stack(memory.raw_mas, dim=0).transpose(0, 1).flatten(0, 1)
        old_proc_time = torch.stack(memory.proc_time, dim=0).transpose(0, 1).flatten(0, 1)
        old_jobs_gather = torch.stack(memory.jobs_gather, dim=0).transpose(0, 1).flatten(0,1)
        old_eligible = torch.stack(memory.eligible, dim=0).transpose(0, 1).flatten(0, 1)
        memory_rewards = torch.stack(memory.rewards, dim=0).transpose(0, 1)
        memory_is_terminals = torch.stack(memory.is_terminals, dim=0).transpose(0, 1)
        old_logprobs = torch.stack(memory.logprobs, dim=0).transpose(0, 1).flatten(0, 1)
        old_action_envs = torch.stack(memory.action_indexes, dim=0).transpose(0, 1).flatten( 0,1)
        old_scale = torch.stack(memory.scale, dim=0).transpose(0, 1).flatten(0,1)
        rewards_envs = []
        discounted_rewards = 0
        for i in range(self.num_envs):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memory_rewards[i]),
                                           reversed(memory_is_terminals[i])):
                if is_terminal:
                    discounted_rewards += discounted_reward
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            discounted_rewards += discounted_reward
            rewards = torch.tensor(rewards, dtype=torch.float64).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards_envs.append(rewards)
        rewards_envs = torch.cat(rewards_envs)
        loss_epochs = 0
        full_batch_size = old_ope_ma_adj.size(0)
        num_complete_minibatches = math.floor(full_batch_size / minibatch_size)
        for _ in range(self.K_epochs):
            for i in range(num_complete_minibatches + 1):
                if i < num_complete_minibatches:
                    start_idx = i * minibatch_size
                    end_idx = (i + 1) * minibatch_size
                else:
                    start_idx = i * minibatch_size
                    end_idx = full_batch_size
                logprobs, state_values, dist_entropy = \
                    self.policy.evaluate(old_ope_ma_adj[start_idx: end_idx, :, :],
                                         old_ope_ope_adj[start_idx: end_idx, :, :],
                                         old_raw_opes[start_idx: end_idx, :, :],
                                         old_raw_mas[start_idx: end_idx, :, :],
                                         old_proc_time[start_idx: end_idx, :, :],
                                         old_jobs_gather[start_idx: end_idx, :, :],
                                         old_eligible[start_idx: end_idx, :, :],
                                         old_action_envs[start_idx: end_idx],
                                         old_scale[start_idx: end_idx]
                                         )
                ratios = torch.exp(logprobs - old_logprobs[i * minibatch_size:(i + 1) * minibatch_size].detach())
                advantages = rewards_envs[i * minibatch_size:(i + 1) * minibatch_size] - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip,1 + self.eps_clip) * advantages
                loss = - self.A_coeff * torch.min(surr1, surr2) \
                       + self.vf_coeff * self.MseLoss(state_values, rewards_envs[i * minibatch_size:( i + 1) * minibatch_size]) \
                       - self.entropy_coeff * dist_entropy
                loss_epochs += loss.mean().detach()

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

        return loss_epochs.item() / self.K_epochs, discounted_rewards.item() / (self.num_envs * train_paras["update_timestep"])
