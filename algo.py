import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class RandomAlg:
    def __init__(self, B):
        self.B = B

    def take_action(self, context):
        action = np.arange(len(context))
        np.random.shuffle(action)

        return action[:min(self.B, len(context))]

    def update(self, context, action, reward, R):
        pass

class Model(nn.Module):

    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()
        self.affine1 = nn.Linear(input_size, hidden_size)
        self.affine2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        return self.affine2(x)



class NeuralMAB(RandomAlg):
    def __init__(self, d, B, hidden_size=128, lr=1e-4):
        self.d = d
        self.B = B
        self.f_hat = Model(d, hidden_size, 1)
        self.g_hat = Model(B, hidden_size, 1)
        self.f_hat.to(device)
        self.g_hat.to(device)
        self.optimizer_f = optim.Adam(self.f_hat.parameters(), lr=lr)
        self.optimizer_g = optim.Adam(self.g_hat.parameters(), lr=lr)

    def take_action(self, context):
        action = set()
        r_hat = torch.zeros((self.B, ))
        for b in range(min(self.B, len(context))):
            index = -1
            max_g = -float('inf')
            max_f = None
            #print(action, r_hat, len(context))
            for i, c in enumerate(context):
                if i not in action:
                    with torch.no_grad():
                        r_c = self.f_hat(torch.tensor(c, dtype=torch.float, device=device).view(1,-1)).item()
                        g_in = r_hat.clone()
                        g_in[len(action)] = r_c
                        g_in = g_in.to(device).view(1,-1)
                        g_out = self.g_hat(g_in).item()
                        if g_out > max_g:
                            max_g = g_out
                            index = i
                            max_f = r_c
            r_hat[len(action)] = max_f
            action.add(index)

        return list(action)

    def update(self, context, action, reward, R):
        loss = F.mse_loss(self.f_hat(torch.tensor(context[action], dtype=torch.float, device=device)), torch.tensor(reward, dtype=torch.float, device=device).view(-1,1))
        g_in = torch.zeros(self.B, dtype=torch.float, device=device)
        g_in[:len(reward)] = torch.tensor(reward, device=device)
        loss += F.mse_loss(self.g_hat(g_in.view(1,-1)), torch.tensor(R, dtype=torch.float, device=device).view(-1,1))
        self.optimizer_f.zero_grad()
        self.optimizer_g.zero_grad()
        loss.backward()
        self.optimizer_f.step()
        self.optimizer_g.step()

class NeuralMABConstraint(RandomAlg):
    def __init__(self, d, B, hidden_size=128, lr=1e-4, eta=0.1, eps=0.01):
        self.d = d
        self.B = B
        self.f_hat = Model(d, hidden_size, 1)
        self.g_hat = Model(B, hidden_size, 1)
        self.f_hat.to(device)
        self.g_hat.to(device)
        self.optimizer_f = optim.Adam(self.f_hat.parameters(), lr=lr)
        self.optimizer_g = optim.Adam(self.g_hat.parameters(), lr=lr)
        self.eta = eta
        self.eps = eps
        self.T = 1
        self.Q = np.zeros(1)


    def take_action(self, context, weight):
        action = set()
        r_hat = torch.zeros((self.B, ))

        cost = self.eta_t * np.matmul(self.Q[None], weight[:, :, None]).flatten()
        for b in range(min(self.B, len(context))):
            index = -1
            max_g = -float('inf')
            max_f = None
            #print(action, r_hat, len(context))
            for i, c in enumerate(context):
                if i not in action:
                    with torch.no_grad():
                        r_c = self.f_hat(torch.tensor(c, dtype=torch.float, device=device).view(1,-1)).item() - cost[i]
                        g_in = r_hat.clone()
                        g_in[len(action)] = r_c
                        g_in = g_in.to(device).view(1,-1)
                        g_out = self.g_hat(g_in).item()
                        if g_out > max_g:
                            max_g = g_out
                            index = i
                            max_f = r_c
            r_hat[len(action)] = max_f
            action.add(index)

        return list(action)

    def update(self, context, action, reward, R, weight):
        loss = F.mse_loss(self.f_hat(torch.tensor(context[action], dtype=torch.float, device=device)), torch.tensor(reward, dtype=torch.float, device=device).view(-1,1))
        g_in = torch.zeros(self.B, dtype=torch.float, device=device)
        g_in[:len(reward)] = torch.tensor(reward, device=device)
        loss += F.mse_loss(self.g_hat(g_in.view(1,-1)), torch.tensor(R, dtype=torch.float, device=device).view(-1,1))
        self.optimizer_f.zero_grad()
        self.optimizer_g.zero_grad()
        loss.backward()
        self.optimizer_f.step()
        self.optimizer_g.step()

        print(weight[action].flatten().sum())
        self.Q = max(np.zeros(1), self.Q + weight[action].flatten().sum() + self.eps_t)
        self.T += 1

    @property
    def eps_t(self):
        return self.eps * np.sqrt(1/self.T)

    @property
    def eta_t(self):
        return self.eta * np.sqrt(1/self.T)

class CCMAB(RandomAlg):

    def __init__(self, d, B, epochs):

        self.d = d
        self.B = B
        self.epochs = epochs
        self.ht = np.ceil(self.epochs**(1/(3+d)))
        self.cube_length = 1 / self.ht
        self.epochs = epochs
        self.total_reward_arr = np.zeros(epochs)
        self.regret_arr = np.zeros(epochs)
        self.hypercube_played_counter_dict = {}
        self.avg_reward_dict = {}


    def get_hypercube_of_context(self, context):
        return tuple((context / self.cube_length).astype(int))

    def take_action(self, context, t):
        arrived_cube_arms_dict = {}
        available_arms = list(range(len(context)))

        arrived_cube_set = set()
        for i, c in enumerate(context):
            hypercube = self.get_hypercube_of_context(c)
            if hypercube not in arrived_cube_set:
                arrived_cube_arms_dict[hypercube] = list()
            arrived_cube_arms_dict[hypercube].append(i)
            arrived_cube_set.add(hypercube)


        underexplored_arm_set = set()
        for cube in arrived_cube_set:
            if self.hypercube_played_counter_dict.get(cube, 0) <= t ** (2 / (3 + self.d)) * np.log(t):
                underexplored_arm_set.update(arrived_cube_arms_dict[cube])

        if len(underexplored_arm_set) >= self.B:
            slate = random.sample(underexplored_arm_set, self.B)

        else:
            slate = []
            slate.extend(underexplored_arm_set)
            not_chosen_arms = list(set(available_arms) - underexplored_arm_set)
            i = 0
            conf_list = np.empty(len(not_chosen_arms))
            for arm in not_chosen_arms:
                conf_list[i] = self.avg_reward_dict.get(self.get_hypercube_of_context(context[arm]), 0)
                i += 1
            arm_indices = np.argsort(conf_list)[len(slate)-self.B:]
            for index in arm_indices:
                selected_arm = not_chosen_arms[index]
                slate.append(selected_arm)

        return slate

    def update(self, context, action, reward, R):

        context = context[action]
        for i, c in enumerate(context):
            cube_with_context = self.get_hypercube_of_context(c)
            new_counter = self.hypercube_played_counter_dict[cube_with_context] = self.hypercube_played_counter_dict.get(
                cube_with_context, 0) + 1
            self.avg_reward_dict[cube_with_context] = (self.avg_reward_dict.get(cube_with_context, 0) * (
                    new_counter - 1) + reward[i]) / new_counter




