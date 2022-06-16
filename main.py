import numpy as np

from algo import *
from data import *
import matplotlib.pyplot as plt
import tqdm

class RewardTracker:

    def __init__(self, keys, epochs=1000, name="neural_mab"):
        self.tracker = {key: np.zeros((epochs,)) for key in keys}
        self.name = name
        self.point = 0
        self.epochs = epochs
        self.times = 1

    def add(self, **kwargs):
        for key in kwargs.keys():
            self.tracker[key][self.point] += 1/self.times * (kwargs[key] - self.tracker[key][self.point])

        self.point += 1

    def next(self):
        self.times += 1
        self.point = 0



def plot(trackers):
    keys = trackers[0].tracker.keys()
    for key in keys:
        for i, tracker in enumerate(trackers):
            plt.plot(trackers[i].tracker[key])
            print(key, trackers[i].tracker[key])
        plt.legend([trackers[i].name for i in range(len(trackers))])
        plt.title(key)
        plt.grid()
        plt.xlabel('T')
        plt.show()


if __name__=="__main__":

    B = 5
    pmean = 50
    sigma_f = 0.05
    trials = 1
    num_rounds = 1000

    tmp = np.array([[-1.0, 10.0]])
    theta_c = - tmp / np.linalg.norm(tmp, axis=1)
    rho = 0.5
    fc = lambda x: np.matmul(theta_c, x[:, :, None]) + rho

    env = ConstraintEnv(fc=fc, pmean=pmean, B=B, sigma_f=sigma_f)
    reward_tracker_neural = RewardTracker(('regret', 'violation'), epochs=num_rounds, name="neural_mab")
    reward_tracker_rand = RewardTracker(('regret', 'violation'), epochs=num_rounds, name="rand")
    reward_tracker_cc = RewardTracker(('regret', 'violation'), epochs=num_rounds, name="cc_mab")
    reward_tracker_neural_con = RewardTracker(('regret', 'violation'), epochs=num_rounds, name="n3c_mab(*)")


    for i in range(trials):
        regret_neural = 0
        regret_rand = 0
        regret_cc = 0
        regret_neural_con = 0

        weights_neural = 0
        weights_rand = 0
        weights_cc = 0
        weights_neural_con = 0

        env.reset()
        neural_mab = NeuralMAB(2, B, lr=3e-4)
        rand = RandomAlg(B)
        cc = CCMAB(2, B, epochs=num_rounds)
        neural_mab_con = NeuralMABConstraint(2, B, lr=3e-4, eta=0.05, eps=0.01)

        for t in tqdm.tqdm(range(num_rounds)):


            context, weight = env.observe()


            action_neural = neural_mab.take_action(context)
            reward_neural, R_neural, oracle_neural = env.step(action_neural)
            neural_mab.update(context, action_neural, reward_neural, R_neural)
            regret_neural += (oracle_neural - R_neural).item()
            weights_neural += weight[action_neural].sum().item()
            reward_tracker_neural.add(regret=regret_neural, violation=max(weights_neural, 0))

            

            action_cc = cc.take_action(context, t+1)
            reward_cc, R_cc, oracle_cc = env.step(action_cc)
            cc.update(context, action_cc, reward_cc, R_cc)
            regret_cc += (oracle_cc - R_cc).item()
            weights_cc += weight[action_cc].sum().item()
            reward_tracker_cc.add(regret=regret_cc, violation=max(weights_cc, 0))


            action_rand = rand.take_action(context)
            reward_rand, R_rand, oracle_rand = env.step(action_rand)
            rand.update(context, action_rand, reward_rand, R_rand)
            regret_rand += (oracle_rand - R_rand).item()
            weights_rand += weight[action_rand].sum().item()
            reward_tracker_rand.add(regret=regret_rand, violation=max(weights_rand, 0))

            action_neural_con = neural_mab_con.take_action(context, weight)
            reward_neural_con, R_neural_con, oracle_neural_con = env.step(action_neural_con)
            neural_mab_con.update(context, action_neural_con, reward_neural_con, R_neural_con, weight)
            regret_neural_con += (oracle_neural_con - R_neural_con).item()
            weights_neural_con += weight[action_neural_con].sum().item()
            reward_tracker_neural_con.add(regret=regret_neural_con, violation=max(weights_neural_con, 0))

            env.update()

        reward_tracker_neural.next()
        reward_tracker_rand.next()
        reward_tracker_cc.next()
        reward_tracker_neural_con.next()


    plot([reward_tracker_neural, reward_tracker_rand, reward_tracker_cc, reward_tracker_neural_con])
    #plot([reward_tracker_rand, reward_tracker_neural_con])