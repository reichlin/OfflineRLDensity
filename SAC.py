import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random


class Network_Q(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Network_Q, self).__init__()

        self.Q = nn.Sequential(
            nn.Linear(state_dim+action_dim, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 1, bias=True)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        return self.Q(x).view(-1)


class Network_policy(nn.Module):

    def __init__(self, state_dim, action_dim, device):
        super(Network_policy, self).__init__()

        self.body = nn.Sequential(
            nn.Linear(state_dim, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
        )

        self.policy_mean = nn.Sequential(
            nn.Linear(256, action_dim, bias=True)
        )

        self.policy_var = nn.Sequential(
            nn.Linear(256, action_dim, bias=True),
            nn.Sigmoid(),
        )

        self.N = torch.distributions.normal.Normal(torch.zeros(action_dim).to(device), torch.ones(action_dim).to(device))

    def get_action(self, s, greedy=False):
        if greedy:
            return torch.tanh(self.policy_mean(self.body(s))), None
        h = self.body(s)
        mean = self.policy_mean(h)
        var = torch.clip(self.policy_var(h), 1e-10, 2.)
        xi = self.N.sample()
        u = mean + var * xi.detach()
        a = torch.tanh(u)
        # neg_log_pi = torch.sum(F.gaussian_nll_loss(mean, a, var, reduction='none'), -1) + torch.sum(torch.log(1 - torch.tanh(a)**2), -1)
        log_pi = torch.sum(-torch.log(torch.clamp(var, 1e-6)), -1) - 0.5 * torch.sum(((u - mean) / var)**2, -1) - torch.sum(torch.log(1 - torch.tanh(u)**2), -1)
        return a, log_pi


class Memory(object):

    def __init__(self, size):
        self.size = size
        self.memory = []
        self.position = 0

    def push_dataset(self, st, a, r, terminal, st1):

        self.memory = [{'st': st[i], 'a': a[i], 'r': r[i], 'terminal': terminal[i], 'st1': st1[i]} for i in range(st.shape[0])]
        self.position = len(self.memory) % self.size

    def push(self, st, a, r, terminal, st1):
        if len(self.memory) < self.size:
            self.memory.append(None)

        element = {'st': st, 'a': a, 'r': r, 'terminal': terminal, 'st1': st1}

        self.memory[int(self.position)] = element
        self.position = (self.position + 1) % self.size

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


'''
    MAIN CLASS REPRESENTING THE RL AGENT
'''
class Agent(nn.Module):

    def __init__(self, dataset_size, data_density, density_uncertainty, beta, offset, temperature, device, data_as_tensor_already=True):
        super(Agent, self).__init__()

        ''' HYPER PARAMETERS '''
        state_dim = 4
        action_dim = 2
        memory_size = dataset_size
        lr = 3e-4
        self.alpha = 0.2
        self.gamma = 0.99
        self.soft_tau = 0.995
        self.batch_size = 100
        self.device = device

        self.data_density = data_density
        self.beta = beta
        self.offset = offset
        self.temperature = temperature

        self.density_uncertainty = density_uncertainty

        self.data_as_tensor_already = data_as_tensor_already

        ''' INITIALIZE NETWORKS '''

        self.policy = Network_policy(state_dim, action_dim, device)
        self.Q_1 = Network_Q(state_dim, action_dim)
        self.Q_2 = Network_Q(state_dim, action_dim)

        self.Q_target_1 = Network_Q(state_dim, action_dim)
        self.Q_target_1.Q.load_state_dict(self.Q_1.Q.state_dict())
        self.Q_target_1.eval()
        self.Q_target_2 = Network_Q(state_dim, action_dim)
        self.Q_target_2.Q.load_state_dict(self.Q_2.Q.state_dict())
        self.Q_target_2.eval()

        self.D = Memory(memory_size)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_Q = optim.Adam(list(self.Q_1.parameters())+list(self.Q_2.parameters()), lr=lr)

    def get_action(self, s, test=False):
        a, log_pi = self.policy.get_action(s, test)
        return a, log_pi

    def get_tensor(self, x):
        return torch.from_numpy(x).float().to(self.device)

    def push_memory(self, s, a, r, t, s1):
        self.D.push(s, a, r, t, s1)

    def get_unc(self, s, a):
        with torch.no_grad():
            a_uncertainties = self.data_density.get_uncertainty(s, a)
            a_unc = torch.sigmoid((a_uncertainties - self.offset) / self.temperature) - 1
            return a_unc

    def update_agent(self):

        if len(self.D.memory) < 1000:
            return None, None, None, None, None, None

        self.policy.train()
        self.Q_1.train()
        self.Q_2.train()

        data = self.D.sample(self.batch_size)

        if self.data_as_tensor_already:
            st = torch.cat([x['st'].view(1, -1) for x in data], 0)
            at = torch.cat([x['a'].view(1, -1) for x in data], 0)
            r = torch.cat([x['r'].float().view(1) for x in data], 0)
            terminal = torch.cat([x['terminal'].float().view(1) for x in data], 0)
            st1 = torch.cat([x['st1'].view(1, -1) for x in data], 0)
        else:
            st = torch.cat([self.get_tensor(x['st']).view(1, -1) for x in data], 0)
            at = torch.cat([self.get_tensor(x['a']).view(1, -1) for x in data], 0)
            r = torch.cat([torch.tensor(x['r'], dtype=torch.float32).view(1) for x in data], 0).to(self.device)
            terminal = torch.cat([torch.tensor(x['terminal'] * 1.0, dtype=torch.float32).view(1) for x in data], 0).to(self.device)
            st1 = torch.cat([self.get_tensor(x['st1']).view(1, -1) for x in data], 0)

        at1, log_pi_t1 = self.policy.get_action(st1)

        ''' THIS IS WHERE UNCERTAINTY IS COMPUTED AND USED TO SCALE Q ESTIMATE '''
        if self.density_uncertainty:
            at1_unc = self.get_unc(st1, at1)
        else:
            at1_unc = 0

        Q1t1 = self.Q_target_1(st1, at1)
        Q2t1 = self.Q_target_2(st1, at1)
        Qt1 = torch.minimum(Q1t1, Q2t1).view(-1) + self.beta * at1_unc
        y = (r + terminal * self.gamma * (Qt1 - self.alpha * log_pi_t1)).detach()

        Q1 = self.Q_1(st, at)
        Q2 = self.Q_2(st, at)
        loss_Q = torch.mean((Q1 - y)**2) + torch.mean((Q2 - y)**2)

        self.optimizer_Q.zero_grad()
        loss_Q.backward()
        self.optimizer_Q.step()

        at_new, log_pi_t_new = self.policy.get_action(st)

        ''' THIS IS WHERE UNCERTAINTY IS COMPUTED AND USED TO SCALE Q ESTIMATE '''
        if self.data_density is not None:
            at_unc = self.get_unc(st, at_new)
        else:
            at_unc = 0

        Q1_new = self.Q_1(st, at_new)
        Q2_new = self.Q_2(st, at_new)
        Q_new = torch.minimum(Q1_new, Q2_new).view(-1) + self.beta * at_unc
        loss_policy = - torch.mean(Q_new - self.alpha * log_pi_t_new)

        self.optimizer_policy.zero_grad()
        loss_policy.backward()
        self.optimizer_policy.step()

        self.update_target()

        unc_at1 = self.data_density.get_uncertainty(st1, at1).detach().cpu().numpy()
        unc_at = self.data_density.get_uncertainty(st, at_new).detach().cpu().numpy()

        if torch.any(torch.isnan(loss_policy)):
            return -1, 0, 0, 0, 0, 0

        return loss_Q.detach().cpu().item(), loss_policy.detach().cpu().item(), torch.mean(Q_new).detach().cpu().item(), torch.mean(log_pi_t_new).detach().cpu().item(), unc_at1, unc_at

    def update_target(self):

        for target_param, param in zip(self.Q_target_1.parameters(), self.Q_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)
        for target_param, param in zip(self.Q_target_2.parameters(), self.Q_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

        self.Q_target_1.eval()
        self.Q_target_2.eval()
