import argparse

class decoder():
    def __init__(self, s_path, v_path):
        self.decode(s_path, v_path)
    
    def decode(self, s_path, v_path):
        states = []
        value_policy = []
        action_list = [0, 1, 2, 4, 6, 7]

        state_data = open(s_path).read().strip().split("\n")
        for row in state_data:
            states.append(row)
        numStates = len(states)

        value_policy_data= open(v_path).read().strip().split("\n")
        for row in value_policy_data:
            value_policy.append(row)
        for i in range(numStates):
            vp = value_policy[i].split()
            print(states[i], action_list[int(vp[1])], vp[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--value-policy", type = str)
    parser.add_argument("--states", type = str)
    args = parser.parse_args()
    s_path = args.states
    v_path = args.value_policy

    decoder(s_path, v_path)