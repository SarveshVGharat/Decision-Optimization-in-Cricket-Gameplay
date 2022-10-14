from pickle import FALSE
from xmlrpc.client import boolean
import numpy as np
import random,argparse,sys
import os
import pulp
parser = argparse.ArgumentParser()

class Planner():
    def __init__(self, mdp, algorithm, policy):
        random.seed(0)
        if (policy == None):
            if algorithm == "lp":
                self.lp(mdp, policy)
            elif algorithm == "hpi":
                self.hpi(mdp, policy)
            elif algorithm == 'vi':
                self.VI(mdp, policy)
        else:
            self.value_calculator(mdp, policy)


    def VI(self, mdp, policy):
        with open(mdp, 'r') as f:
            lines = f.readlines()
            states = int(lines[0].strip().split()[1])
            actions = int(lines[1].strip().split()[1])
            end_state = int(lines[2].strip().split()[1])
            transition = []
            reward = []
            for i in range(actions):
                transition.append(np.zeros((states, states)))
                reward.append(np.zeros((states, states)))
            i = 3
            while(str(lines[i].strip().split()[0]) == 'transition'):
                actionab = int(lines[i].strip().split()[2])
                state_a = int(lines[i].strip().split()[1])
                state_b = int(lines[i].strip().split()[3])
                rewardab = float(lines[i].strip().split()[4])
                transition_prob = float(lines[i].strip().split()[5])
                transition[actionab][state_a][state_b] = float(transition_prob)
                reward[actionab][state_a][state_b] = float(rewardab)
                i += 1
            type = str(lines[i].strip().split()[1])
            discount = float(lines[i+1].strip().split()[1])

            V_old = np.zeros(states)
            action = np.zeros(states)
            temp = np.zeros((states, actions))
            V_new = 10000*(V_old + 1)
            thresh = 1e-8
            for s in range(states):
                while(np.abs(V_new[s]-V_old[s]) > thresh):
                    for s1 in range(states):
                        for a in range(actions):
                            sum = 0
                            for s2 in range(states):
                                sum += transition[a][s1][s2]*(reward[a][s1][s2]+discount*V_old[s2])
                            temp[s1][a] = sum
                        action[s1] = np.argmax(temp[s1])
                        V_old[s1] = V_new[s1]
                        V_new[s1] = np.max(temp[s1])
            for s in range(states):
                print("%.8f" % V_new[s],'\t', int(action[s]))

    def hpi(self, mdp, policy):
        with open(mdp, 'r') as f:
            lines = f.readlines()
            states = int(lines[0].strip().split()[1])
            actions = int(lines[1].strip().split()[1])
            end_state = int(lines[2].strip().split()[1])
            transition = []
            reward = []
            for i in range(states):
                transition.append(np.zeros((actions, states)))
                reward.append(np.zeros((actions, states)))
            transition = np.array(transition)
            reward = np.array(reward)
            i = 3
            while(str(lines[i].strip().split()[0]) == 'transition'):
                actionab = int(lines[i].strip().split()[2])
                state_a = int(lines[i].strip().split()[1])
                state_b = int(lines[i].strip().split()[3])
                rewardab = float(lines[i].strip().split()[4])
                transition_prob = float(lines[i].strip().split()[5])
                transition[state_a][actionab][state_b] = float(transition_prob)
                reward[state_a][actionab][state_b] = float(rewardab)
                i += 1
            type = str(lines[i].strip().split()[1])
            discount = float(lines[i+1].strip().split()[1])
            policy_actions = np.zeros(states)
            policy_actions = policy_actions.astype(int)
            
            update = 0
            while True:
                Transition_policy = transition[np.arange(states),policy_actions]
                Reward_policy = reward[np.arange(states),policy_actions]
                Value_policy = np.squeeze(np.linalg.inv(np.eye(states) - discount*Transition_policy) @ np.sum(Transition_policy * Reward_policy, axis = -1, keepdims = True))
                Action_value = np.zeros((states,actions))
                for s in range(states):
                    for a in range(actions):
                        if policy_actions[s] != a:
                            #update = 0
                            for s1 in range(states):
                                Action_value[s][a] += transition[s][a][s1] * (reward[s][a][s1] + discount*Value_policy[s1])
                            #Action_value[s][a] = update
                        else:
                            Action_value[s][a] = Value_policy[s]
                update = update + 1
                IA = {}
                IS = []
                for s in range(states):
                    for a in range(actions):
                        if Action_value[s][a] > Value_policy[s]:
                            if s not in IA.keys():
                                IA[s] = [a]
                            else:
                                IA[s].append(a)
                    if s in IA.keys():
                        IS.append(s)
                        policy_actions[s] = random.choice(IA[s])
                if len(IS) == 0 or update == 100:
                    break
            for s in range(states):
                print("%.8f" % Value_policy[s],'\t', int(policy_actions[s]))



    def lp(self, mdp, policy):
        with open(mdp, 'r') as f:
            lines = f.readlines()
            states = int(lines[0].strip().split()[1])
            actions = int(lines[1].strip().split()[1])
            end_state = int(lines[2].strip().split()[1])
            transition = []
            reward = []
            for i in range(states):
                transition.append(np.zeros((actions, states)))
                reward.append(np.zeros((actions, states)))
            transition = np.array(transition)
            reward = np.array(reward)
            i = 3
            while(str(lines[i].strip().split()[0]) == 'transition'):
                actionab = int(lines[i].strip().split()[2])
                state_a = int(lines[i].strip().split()[1])
                state_b = int(lines[i].strip().split()[3])
                rewardab = float(lines[i].strip().split()[4])
                transition_prob = float(lines[i].strip().split()[5])
                transition[state_a][actionab][state_b] = float(transition_prob)
                reward[state_a][actionab][state_b] = float(rewardab)
                i += 1
            type = str(lines[i].strip().split()[1])
            discount = float(lines[i+1].strip().split()[1])

            probability = pulp.LpProblem("LP", pulp.LpMinimize)
            value = np.array(list(pulp.LpVariable.dicts("value", [i for i in range(states)]).values()))
            probability += pulp.lpSum(value)
            for s1 in range(states):
                for a in range(actions):
                    probability += value[s1] >= pulp.lpSum(transition[s1][a] * (reward[s1][a] + discount*value))
            probability.solve(pulp.apis.PULP_CBC_CMD(msg = 0))
            value = np.array(list(map(pulp.value, value)))
            policy = np.argmax(np.sum(transition*(reward + discount*value), axis = -1), axis = -1) 
            for s in range(states):
                print("%.8f" % value[s],'\t', int(policy[s]))

    def value_calculator(self, mdp, policy):
        if (isinstance(policy, str)):
            if (os.path.isfile(args.policy)):
                with open(policy, 'r') as f:
                    lines = f.readlines()
                    policy_actions = np.zeros(len(lines))
                    for i in range(len(lines)):
                        policy_actions[i] = int(lines[i])
                    policy_actions = policy_actions.astype(int)
               
        with open(mdp, 'r') as f:
            lines = f.readlines()
            states = int(lines[0].strip().split()[1])
            actions = int(lines[1].strip().split()[1])
            end_state = int(lines[2].strip().split()[1])
            transition = []
            reward = []
            for i in range(states):
                transition.append(np.zeros((actions, states)))
                reward.append(np.zeros((actions, states)))
            transition = np.array(transition)
            reward = np.array(reward)
            i = 3
            while(str(lines[i].strip().split()[0]) == 'transition'):
                actionab = int(lines[i].strip().split()[2])
                state_a = int(lines[i].strip().split()[1])
                state_b = int(lines[i].strip().split()[3])
                rewardab = float(lines[i].strip().split()[4])
                transition_prob = float(lines[i].strip().split()[5])
                transition[state_a][actionab][state_b] = float(transition_prob)
                reward[state_a][actionab][state_b] = float(rewardab)
                i += 1
            type = str(lines[i].strip().split()[1])
            discount = float(lines[i+1].strip().split()[1])

            Transition_policy = transition[np.arange(states),policy_actions]
            Reward_policy = reward[np.arange(states),policy_actions]
            Value_policy = np.squeeze(np.linalg.inv(np.eye(states) - discount*Transition_policy) @ np.sum(Transition_policy * Reward_policy, axis = -1, keepdims = True))
            for s in range(states):
                print("%.8f" % Value_policy[s],'\t', int(policy_actions[s]))

if __name__ == "__main__":
    parser.add_argument("--mdp", type=str, required=True)
    parser.add_argument("--algorithm", type=str, default="hpi")
    parser.add_argument("--policy", type=str, required=False)
    args = parser.parse_args()

    algo = Planner(args.mdp, args.algorithm, args.policy)


