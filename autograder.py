#! /usr/bin/python
from email import policy
import random,argparse,sys,subprocess,os
parser = argparse.ArgumentParser()
import numpy as np
random.seed(0)

input_file_ls = ['data/mdp/continuing-mdp-10-5.txt','data/mdp/continuing-mdp-2-2.txt','data/mdp/continuing-mdp-50-20.txt','data/mdp/episodic-mdp-10-5.txt','data/mdp/episodic-mdp-2-2.txt','data/mdp/episodic-mdp-50-20.txt']
flag_ok = 0

class VerifyOutputPlanner:
    def __init__(self,algorithm,print_error):
        algorithm_ls = list()
        if algorithm=='all':
            algorithm_ls+=['hpi','vi','lp', 'default']
        else:
            algorithm_ls.append(algorithm)
            
        for algo in algorithm_ls:
            print('verify output',algo)
            counter = 1    
        
            for in_file in input_file_ls:
                print("\n\n","-"*100)
                if algo == 'default':
                    cmd_planner = "python","planner.py","--mdp",in_file
                else:
                    cmd_planner = "python","planner.py","--mdp",in_file,"--algorithm",algo
                print('test case',str(counter),algo,":\t"," ".join(cmd_planner))
                counter+=1
                cmd_output = subprocess.check_output(cmd_planner,universal_newlines=True)
                self.verifyOutput(cmd_output,in_file,print_error)
        policy_eval_files = ['data/mdp/continuing-mdp-10-5.txt', 'data/mdp/episodic-mdp-10-5.txt']
        for in_file in policy_eval_files:
            cmd_planner = "python","planner.py","--mdp",in_file, "--policy", in_file.replace("continuing","rand-continuing").replace("episodic","rand-episodic")
            print('test case',str(counter),'policy evaluation',":\t"," ".join(cmd_planner))
            counter+=1
            cmd_output = subprocess.check_output(cmd_planner,universal_newlines=True)
            self.verifyOutput(cmd_output,in_file,print_error, pol_eval = True)
        
                    
            
            

    def verifyOutput(self,cmd_output,in_file,pe, pol_eval = False):

        sol_file = in_file.replace("continuing","sol-continuing").replace("episodic","sol-episodic")
        if (pol_eval):
            sol_file = in_file.replace("continuing","sol-rand-continuing").replace("episodic","sol-rand-episodic")
        base = np.loadtxt(sol_file,delimiter=" ",dtype=float)
        output = cmd_output.split("\n")
        nstates = base.shape[0]
        
        est = [i.split() for i in output if i!='']
        
        
        mistakeFlag = False
        #Check1: Checking the number of lines printed
        if not len(est)==nstates:
            mistakeFlag = True
            print("\n","*"*10,"Mistake:Exact number of line in standard output should be",nstates,"but have",len(est),"*"*10)
            
        #Check2: Each line should have only two values
        for i in range(len(est)):
            if not len(est[i])==2:
                mistakeFlag = True
                print("\n","*"*10,"Mistake: On each line you should print only value,policy for a state","*"*10)
                break
        
        if not mistakeFlag:
            print("ALL CHECKS PASSED!")
        else:
            print("You haven't printed output in correct format.")
            
        pe_ls = ['no','NO','No','nO']
        if pe not in pe_ls:
            if not mistakeFlag:
                print("Calculating error of your value function...")
            else:
                print("\nExiting without calculating error of your value function")
                return
            #calculating the error
            for i in range(len(est)):
                est_V = float(est[i][0]);base_V = float(base[i][0])
                print("%10.6f"%est_V,"%10.6f"%base_V,"%10.6f"%abs(est_V-base_V),end="\t")
                if abs(est_V-base_V) <= (10**-4):
                    print("OK")
                else:
                    flag_ok = 1
                    print("\tNot OK")
            

def run(states, p1_parameter, q):
    cmd_encoder = "python","encoder.py","--parameters", p1_parameter, "--q", q, "--states",states
    print("\n","Generating the MDP encoding using encoder.py")
    f = open('verify_attt_mdp','w')
    subprocess.call(cmd_encoder,stdout=f)
    f.close()

    cmd_planner = "python","planner.py","--mdp","verify_attt_mdp"
    print("\n","Generating the value policy file using planner.py using default algorithm")
    f = open('verify_attt_planner','w')
    subprocess.call(cmd_planner,stdout=f)
    f.close()

    cmd_decoder = "python","decoder.py","--value-policy","verify_attt_planner","--states",states 
    print("\n","Generating the decoded policy file using decoder.py")
    cmd_output = subprocess.check_output(cmd_decoder,universal_newlines=True)

    os.remove('verify_attt_mdp')
    os.remove('verify_attt_planner')
    return cmd_output

def verifyOutput(states, output, in_file, q):
    output = output.split('\n')
    output.remove('')
    with open(states,'r') as file:
        lines = file.readlines()
    states = [line.strip() for line in lines]
    if len(output) != len(states):
        print("\n","*"*10,f"Mistake: Expected {len(states)} policy lines, got {len(output)}")
        sys.exit()
    
    policy_states=[]
    for idx,out in enumerate(output):
        terms = out.split(' ')
        if (terms[1] not in ['0','1','2','4','6']):
            print("\n", terms[1], " is not a valid action")
        if len(terms) !=3:
            print("\n","*"*10,f"Mistake: In line {idx+1}, expected 2 terms , got {len(terms)}. {out}")
            sys.exit()
        policy_states.append(terms[0])
        try:
            p = list(map(float,terms[1:]))
        except:
            print("\n","*"*10,f"Mistake: In line {idx+1}, Number format excpetion. {out}")
            sys.exit()
    
    states_intersection = set(states).intersection(set(policy_states))
    if len(states_intersection) != len(states):
        print('States missing', set(states).difference(states_intersection))
        print("\n","*"*10,f"Mistake: States in policy file and input states file do not match")
        sys.exit()

    if (q == "0.25"):
        print("Verifying policy and win probabilities")
        sol_file = in_file.replace("sample","sol")
        base = np.loadtxt(sol_file,delimiter=" ",dtype=float)
        
        for i in range(len(output)):
            terms = output[i].split(' ')
            est_V = float(terms[2])
            base_V = float(base[i][2])
            est_A = int(terms[1])
            base_A = int(base[i][1])
            if(base_A != est_A):
                print(terms[0], end=' ')
                print("Action does not match, but it may be correct if the same value function is obtained for another action")
            
            if abs(est_V-base_V) > (10**-4):
                print(terms[0], end=' ')
                print("%10.6f"%est_V,"%10.6f"%base_V,"%10.6f"%abs(est_V-base_V),end="\t")
                print("\t Value function not OK")
                return
    else:
        print("Not verified policy and win probabilities. Use default --q to verify.")
        return

    
    print("All OK")




if __name__ == "__main__":
    parser.add_argument('--task', type = int)
    parser.add_argument("--algorithm",type=str,default="default")
    parser.add_argument("--pe",type=str,default="yes")
    parser.add_argument("--states",type=str,help="File with valid states of the player", default="NA")
    parser.add_argument("--parameters",type=str,help="File with valid environment parameters", default="NA")
    parser.add_argument("--q",type=str,help="Weakness of player B", default="0.25")
    args = parser.parse_args()

    #print(args)
    #sys.exit(0)
    if(args.task == 1):
        algo = VerifyOutputPlanner(args.algorithm,args.pe)
        if(flag_ok):
            print("THERE IS A MISTAKE in Task 1")
    else:
        in_file_ls = ['data/cricket/sample-p1.txt', 'data/cricket/sample-p2.txt']
        states = 'data/cricket/cricket_state_list.txt'
        if (args.states != 'NA'):
            states = args.states
        if (args.parameters != 'NA'):
            in_file_ls = [args.parameters]
        for in_file in in_file_ls:
            print ("Running for ", states, ' with policy', in_file)
            output = run(states,in_file,args.q)
            verifyOutput(states, output, in_file, args.q)

