import argparse

class encoder():
    def __init__(self,s_path, p_path, q):
        self.encode(s_path, p_path, q)

    def encode(self, s_path, p_path, q):
        with open(s_path, 'r') as f:
            lines = f.readlines()
            state_conc = []
            state_orig = []
            for i in range(len(lines)):
                state_orig.append(lines[i].strip())
                state_conc.append("A"+lines[i].strip())
            for i in range(len(lines)):
                state_conc.append("B"+lines[i].strip())
            state_conc.append("Loose")
            state_conc.append("Win")
            state_orig.append("Loose")
            state_orig.append("Win")
        states = len(state_conc)

        print("numStates", states)
        print("numActions", 6)
        print("end", -1)

        with open(p_path, 'r') as f:
            lines = f.readlines()
            prob = []
            for i in range(len(lines)):
                if lines[i].split()[0]!="action":
                    prob.append(lines[i].strip().split()[1:])
            for i in range(len(prob)):
                prob[i] = prob[i][1:] + prob[i][:1]
        
        actions = [0, 1, 2, 4, 6, 7]
        A_outcome = [0, 1, 2, 3, 4, 6, -1]
        B_outcome = [0, 1, -1]

        for s in range(states-2):
            state = state_conc[s]
            player = state[0]
            balls_left = int(state[1:3])
            runs_left = int(state[3:])
            if player == "A":
                for action in range(len(actions)-1):
                    win_prob = 0
                    loose_prob = 0

                    for outcomes in range(len(A_outcome)):
                        outcome = A_outcome[outcomes]
                        if outcome >= runs_left:
                            next_state = "Win" 
                        elif outcome == -1 or balls_left == 1:
                            next_state = "Loose"
                        else:
                            bb = balls_left - 1
                            rr = runs_left - outcome
                            bbrr = str(bb).zfill(2) + str(rr).zfill(2)
                            if (balls_left % 6) != 1:
                                if outcome == 1 or outcome == 3:
                                    next_state = "B" + bbrr
                                else:
                                    next_state = "A" + bbrr
                            else:
                                if outcome == 1 or outcome == 3:
                                    next_state = "A" + bbrr
                                else:
                                    next_state = "B" + bbrr
                        new_state = state_conc.index(next_state)
                        if next_state == "Win":
                            reward = 1
                        else:
                            reward = 0
                        if next_state == "Win":
                            win_prob += float(prob[action][outcomes])
                        elif next_state == "Loose":
                            loose_prob += float(prob[action][outcomes])
                        else:
                            probabibity = float(prob[action][outcomes])
                            if probabibity != 0:
                                print("transition", s, action, new_state, reward, probabibity)
                    if win_prob != 0:
                        print("transition", s, action, state_conc.index("Win"), 1, win_prob)
                    if loose_prob != 0:
                        print("transition", s, action, state_conc.index("Loose"), 0, loose_prob)

            if player == "B":
                for outcomes in range(len(B_outcome)):
                    outcome = B_outcome[outcomes]
                    if outcome >= runs_left:
                        next_state = "Win"
                    elif balls_left == 1 or outcome == -1:
                        next_state = "Loose"
                    else:
                        bb = balls_left - 1
                        rr = runs_left - outcome
                        bbrr = str(bb).zfill(2) + str(rr).zfill(2)
                        if (balls_left % 6) == 1:
                            if outcome == 0:
                                next_state = "A" + bbrr
                            else:
                                next_state = "B" + bbrr
                        else:
                            if outcome == 0:
                                next_state = "B" + bbrr
                            else:
                                next_state = "A" + bbrr
                    new_state = state_conc.index(next_state)
                    if next_state == "Win":
                        reward = 1
                    else:
                        reward = 0
                    if outcome == -1:
                        probabibity = q
                    else:
                        probabibity = (1-q)/2
                    if probabibity != 0:
                        action = 5
                        print("transition", s, action, new_state, reward, probabibity)
        
        print("mdptype", "episodic")
        print("discount", 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type = str)
    parser.add_argument("--parameters", type = str)
    parser.add_argument("--q", type = float)
    args = parser.parse_args()
    s_path = args.states
    p_path = args.parameters
    q = args.q

    encoder(s_path, p_path, q)