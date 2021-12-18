from SAC import Agent
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
from density_estimators import KDE
import argparse
import os

import matplotlib.pyplot as plt


'''
    Test the policy on the environment and return the average reward
'''
def test_policy(env, agent, MAX_STEPS):
    st = env.reset()
    done = False
    tot_r = 0
    steps = 0
    while (not done) and steps < MAX_STEPS:
        a, _ = agent.get_action(agent.get_tensor(st), test=True)
        st1, r, done, boh = env.step(a.detach().cpu().numpy())
        st = st1
        tot_r += r
        steps += 1
    return tot_r


'''
    One step of training for the agent by sampling data from the collected buffer, write data on logs and save model once in while
'''
def train(agent, env, epoch, writer, simulate, GAMES, MAX_STEPS, run_name, save_models):

    loss_Q, loss_policy, avg_Q, log_pi_t_new, unc_at1, unc_at = agent.update_agent()

    if loss_Q == -1:
        return 1

    if loss_Q is not None:

        if epoch % 100 == 99 and simulate:
            tot_r = 0
            for game in range(GAMES):
                tot_r += test_policy(env, agent, MAX_STEPS)
            writer.add_scalar("Reward", tot_r / GAMES, epoch)

        writer.add_scalar("Loss_Q", loss_Q, epoch)
        writer.add_scalar("Loss_Policy", loss_policy, epoch)
        writer.add_scalar("Avg_Q", avg_Q, epoch)
        writer.add_scalar("Log_Pi", log_pi_t_new, epoch)

        # if epoch % 20 == 0:
        #     plt.hist(unc_at1, 20)
        #     plt.show()
        #     plt.hist(unc_at, 20)
        #     plt.show()

        if unc_at1 is not None and epoch % 20 == 0:
            writer.add_histogram("Unc_at1", unc_at1, epoch)
            writer.add_histogram("Unc_at", unc_at, epoch)

    if epoch % 10000 == 9999 and save_models:
        torch.save(agent.state_dict(), "./models/" + run_name + "/" + str(epoch / 10000) + ".pt")

    return 0

'''
    fully train a model on a previously collected dataset, this function is used when running on a device that can't use the simulator
'''
def train_on_dataset(params, dataset_name, env_name, MAX_STEPS, EPOCHS, GAMES, kde_data_size, simulate, save_models, device):

    density_uncertainty = (params['kde'] == 1)
    beta = params['beta']
    offset = params['offset']
    temperature = params['temperature']

    run_name = "Offline=" + str(True) + "_KDE=" + str(density_uncertainty) + "_beta=" + str(beta) + "_offset=" + str(offset) + "_temperature=" + str(temperature)
    logdir = "logs/" + run_name
    writer = SummaryWriter(log_dir=logdir)

    if not os.path.exists("./models/" + run_name):
        os.mkdir("./models/" + run_name)


    dataset = np.load(dataset_name)
    dataset_size = dataset['observations'].shape[0]

    if simulate:

        import gym
        import d4rl

        env = gym.make(env_name)
        # dataset = d4rl.qlearning_dataset(env)


    # if density_uncertainty:
    #     data_density = KDE(torch.from_numpy(dataset['observations'][:kde_data_size]).float().to(device), torch.from_numpy(dataset['actions'][:kde_data_size]).float().to(device))
    # else:
    #     data_density = None
    data_density = KDE(torch.from_numpy(dataset['observations'][:kde_data_size]).float().to(device), torch.from_numpy(dataset['actions'][:kde_data_size]).float().to(device))

    agent = Agent(dataset_size, data_density, density_uncertainty, beta, offset, temperature, device, data_as_tensor_already=True).to(device)



    print("Load dataset in Buffer:")

    obs = torch.from_numpy(dataset['observations']).float().to(device)
    a = torch.from_numpy(dataset['actions']).float().to(device)
    r = torch.from_numpy(dataset['rewards']).float().to(device)
    t = torch.from_numpy(1 - dataset['rewards']).float().to(device)
    next_obs = torch.from_numpy(dataset['next_observations']).float().to(device)

    agent.D.push_dataset(obs, a, r, t, next_obs)

    time.sleep(0.1)

    print("Start learning:")
    for epoch in tqdm(range(EPOCHS)):

        error = train(agent, None, epoch, writer, simulate, GAMES, MAX_STEPS, run_name, save_models)
        if error == 1:
            return

    writer.close()

    return


'''
    fully train a model on the environment either online or offline
'''
def train_on_env(params, env_name, MAX_STEPS, EPOCHS, GAMES, kde_data_size, simulate, save_models, device):

    offline = (params['offline'] == 1)
    density_uncertainty = (params['kde'] == 1)
    beta = params['beta']
    offset = params['offset']
    temperature = params['temperature']


    run_name = "Offline=" + str(offline) + "_KDE=" + str(density_uncertainty) + "_beta=" + str(beta) + "_offset=" + str(offset) + "_temperature=" + str(temperature)
    logdir = "logs/" + run_name
    writer = SummaryWriter(log_dir=logdir)

    if not os.path.exists("./models/" + run_name):
        os.mkdir("./models/" + run_name)

    import gym
    import d4rl

    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    dataset_size = dataset['observations'].shape[0]

    # if density_uncertainty:
    #     data_density = KDE(torch.from_numpy(dataset['observations'][:kde_data_size]).float().to(device), torch.from_numpy(dataset['actions'][:kde_data_size]).float().to(device))
    # else:
    #     data_density = None
    data_density = KDE(torch.from_numpy(dataset['observations'][:kde_data_size]).float().to(device), torch.from_numpy(dataset['actions'][:kde_data_size]).float().to(device))

    if offline:
        agent = Agent(dataset_size, data_density, density_uncertainty, beta, offset, temperature, device, data_as_tensor_already=True).to(device)

        print("Load dataset in Buffer:")

        obs = torch.from_numpy(dataset['observations']).float().to(device)
        a = torch.from_numpy(dataset['actions']).float().to(device)
        r = torch.from_numpy(dataset['rewards']).float().to(device)
        t = torch.from_numpy(1 - dataset['terminals']).float().to(device)
        next_obs = torch.from_numpy(dataset['next_observations']).float().to(device)

        agent.D.push_dataset(obs, a, r, t, next_obs)

        time.sleep(0.1)
    else:
        agent = Agent(dataset_size, data_density, False, beta, offset, temperature, device, data_as_tensor_already=False).to(device)

    print("Start learning:")
    for epoch in tqdm(range(EPOCHS)):

        if not offline:
            st = env.reset()
            done = False
            steps = 0
            while (not done) and steps < MAX_STEPS:

                # if epoch < 100:
                #     a = env.action_space.sample()
                # else:
                a, _ = agent.get_action(agent.get_tensor(st))
                a = a.detach().cpu().numpy()

                st1, r, done, _ = env.step(a)
                # if r > 0:
                #     done = True
                # else:
                #     done = False
                agent.push_memory(st, a, r, not done, st1)
                st = st1
                steps += 1

        error = train(agent, env, epoch, writer, simulate, GAMES, MAX_STEPS, run_name, save_models)
        if error == 1:
            return

    writer.close()

    return


'''
    test saved models on the environment
'''
def test_saved_models(env_name, GAMES, MAX_STEPS, device):
    import gym
    import d4rl
    from os import listdir
    import matplotlib.pyplot as plt

    env = gym.make(env_name)
    for model_type_path in listdir("./models/"):
        writer = SummaryWriter(log_dir="./logs_test/" + model_type_path)
        models = listdir("./models/" + model_type_path)

        sorted_models = []
        N = len(models)
        for i in range(len(models)):
            min_model_name = N
            min_model = None
            for j, model in enumerate(models):
                if min_model_name > int(model[:-8]):
                    min_model_name = int(model[:-8])
                    min_model = model
            sorted_models.append(min_model)
            models.remove(min_model)

        for i, model_path in enumerate(sorted_models[90:]):

            agent = Agent(None, None, None, None, None, device, data_as_tensor_already=False).to(device)
            agent.load_state_dict(torch.load('./models/' + model_type_path + "/" + model_path))

            tot_r = 0
            for game in tqdm(range(GAMES)):
                tot_r += test_policy(env, agent, MAX_STEPS)

            writer.add_scalar("Reward", tot_r / GAMES, i + 90)

        writer.close()

    return


if __name__ == "__main__":

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', default=1, type=int, help='Offline RL')
    parser.add_argument('--kde', default=1, type=int, help='Use density')
    parser.add_argument('--beta', default=1., type=float, help='Scale constant')
    parser.add_argument('--offset', default=0.5, type=float, help='Zero of sigmoid')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature of sigmoid')
    args = parser.parse_args()
    params = args.__dict__

    device = torch.device('cpu')  # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    env_name = 'maze2d-umaze-v1'  # 'maze2d-large-v1'
    dataset_name = "./d4rl_dataset_.npz"  # "./d4rl_dataset_large.npz"
    MAX_STEPS = 300  # 800  # 300
    EPOCHS = 1000000
    GAMES = 100
    kde_data_size = 100000
    simulate = True
    save_models = False

    ''' RUN EITHER ONE OF THE THREE BY UNCOMMENTING IT '''
    # train_on_dataset(params, dataset_name, env_name, MAX_STEPS, EPOCHS, GAMES, kde_data_size, simulate, save_models, device)  # train on saved npz files
    #
    # train_on_env(params, env_name, MAX_STEPS, EPOCHS, GAMES, kde_data_size, simulate, save_models, device)  # train on d4rl library data
    #
    # test_saved_models(env_name, GAMES, MAX_STEPS, device)  # test saved models

    EPOCHS = 10000

    # params['offline'] = 0
    # params['kde'] = 0
    # train_on_env(params, env_name, MAX_STEPS, EPOCHS, GAMES, kde_data_size, simulate, save_models, device)
    # params['offline'] = 1
    # params['kde'] = 0
    # train_on_env(params, env_name, MAX_STEPS, EPOCHS, GAMES, kde_data_size, simulate, save_models, device)
    params['offline'] = 1
    params['kde'] = 1
    params['beta'] = 0.5
    train_on_env(params, env_name, MAX_STEPS, EPOCHS, GAMES, kde_data_size, simulate, save_models, device)









