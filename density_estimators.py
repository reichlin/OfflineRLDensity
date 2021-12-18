import numpy as np
import torch
from tqdm import tqdm

class KDE():

    def __init__(self, states, actions):
        super(KDE, self).__init__()

        self.states = states
        self.actions = actions

        self.N = self.states.shape[0]
        self.d = [self.states.shape[1], self.actions.shape[1]]
        self.h = [0.1, 0.3] #[0.05, 0.3]

        # self.h = nn.Parameter(torch.tensor([1.0, 0.3], requires_grad=True))
        # self.optimizer = optim.Adam([self.h], lr=0.001)

        # logdir = "logs_kde/3"
        # self.writer = SummaryWriter(log_dir=logdir)

        return

    def K(self, x, x_data, i):

        diff = torch.cdist(x, x_data, p=2)
        delta = - 0.5 * (diff / self.h[i]) ** 2
        return torch.exp(delta)

    def get_uncertainty(self, s, a):

        ps = self.K(s, self.states, 0)
        pa = self.K(a, self.actions, 1)

        # p = torch.sum(pa * ps, -1) / (self.h[1] * torch.sum(ps, -1))
        p = torch.sum(pa * ps, -1) / (self.h[1] * torch.sum(ps, -1))

        if torch.any(torch.isnan(p)):
            p = torch.nan_to_num(p, nan=1e-6)

        return p

    # def fit(self, s, ap, an):
    #
    #     N = s.shape[0]
    #     batch_size = 200
    #
    #     avg_pp = []
    #     avg_pn = []
    #     for batch in tqdm(range(int(N / batch_size))):
    #         pp = kde.get_uncertainty(s[batch * batch_size:(batch + 1) * batch_size], ap[batch * batch_size:(batch + 1) * batch_size])
    #         pn = kde.get_uncertainty(s[batch * batch_size:(batch + 1) * batch_size], an[batch * batch_size:(batch + 1) * batch_size])
    #
    #         loss = torch.sum(pn) - torch.sum(pp)
    #
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()

if __name__ == "__main__":
    import gym
    import d4rl
    import torch.optim as optim
    import torch.nn as nn
    from torch.utils.tensorboard import SummaryWriter
    import time
    import matplotlib.pyplot as plt

    device = torch.device('cuda:0')  # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    env = gym.make('maze2d-umaze-v1')

    dataset = d4rl.qlearning_dataset(env)
    dataset_size = dataset['observations'].shape[0]

    states = torch.tensor(dataset['observations']).float().to(device)
    actions = torch.tensor(dataset['actions']).float().to(device)

    N = states.shape[0]
    test_size = 10000
    train_size = 100000

    states_train = states[:train_size]#[:(N-test_size)]
    states_test = states[(N - test_size):]
    actions_train = actions[:train_size]#[:(N-test_size)]
    actions_test = actions[(N - test_size):]
    actions_outliers = torch.rand(test_size, 2).to(device) * 2 - 1

    kde = KDE(states_train, actions_train)


    # get distribution of density estimation
    N = states_test.shape[0]
    batch_size = 200

    avg_pp = []
    avg_pn = []
    for batch in tqdm(range(int(actions_test.shape[0] / batch_size))):
        pp = kde.get_uncertainty(states_test[batch * batch_size:(batch + 1) * batch_size], actions_test[batch * batch_size:(batch + 1) * batch_size])
        pn = kde.get_uncertainty(states_test[batch * batch_size:(batch + 1) * batch_size], actions_outliers[batch * batch_size:(batch + 1) * batch_size])

        avg_pp.extend(pp.detach().cpu().numpy())  # += torch.sum(pp).detach().item()
        avg_pn.extend(pn.detach().cpu().numpy())  # += torch.sum(pn).detach().item()

    avg_pp = [x.item() for x in avg_pp]
    avg_pn = [x.item() for x in avg_pn]

    print(np.mean(np.array(avg_pp)), np.mean(np.array(avg_pn)))

    print()

    # get picture

    # for h1 in [0.05]:
    #     for h2 in [0.3]:
    #         print("h = [" + str(h1) + ", " + str(h2) + "]", end=" ")
    #         kde.h = [h1, h2]
    #         kde.fit(states_test, actions_test, actions_outliers)

    # print(kde.get_uncertainty(x[0:1], y[0:1]))
    #
    # st = states_test[0:1]
    #
    # discretization = 100
    # img = np.zeros((discretization, discretization))
    #
    # for ix, ax in tqdm(enumerate(np.linspace(-1., 1., discretization))):
    #     for iy, ay in enumerate(np.linspace(-1., 1., discretization)):
    #         at = torch.tensor([[ax, ay]]).float()
    #
    #         img[ix, iy] = kde.get_uncertainty(st, at).item()
    #
    # plt.imshow(img)
    # plt.show()

    print()