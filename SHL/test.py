import copy
import json
import os
import random
import time as time
import gym
import pandas as pd
import torch
import numpy as np
#import pynvml
import PPO_model
from env.load_data import nums_detec
#from line_profiler import LineProfiler
import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # PyTorch initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    #print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    # Load config and init objects
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    test_paras = load_dict["test_paras"]
    env_paras["device"] = device
    model_paras["device"] = device
    env_test_paras = copy.deepcopy(env_paras)
    num_ins = test_paras["num_ins"]
    if test_paras["sample"]:
        env_test_paras["batch_size"] = test_paras["num_sample"]
    else:
        env_test_paras["batch_size"] = 1
    model_paras["actor_in_dim"] = model_paras["out_dim"] * 2 + model_paras["out_dim"] * 2 #actor
    model_paras["critic_in_dim"] = model_paras["out_dim"] + model_paras["out_dim"]#critic

    data_path = "./data_test/{0}/".format(test_paras["data_path"])
    test_files = os.listdir(data_path)
    test_files.sort(key=lambda x: x[:-4])
    test_files = test_files[: num_ins]
    mod_files = os.listdir('./model/')[:]

    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_paras, train_paras)
    rules = []
    envs = []  # Store multiple environments

    # Detect and add models to "rules"
    for root, ds, fs in os.walk('./model/'):
        for f in fs:
            if f.endswith('.pt'):
                 rules.append(f)

    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './save/test_{0}'.format(str_time)
    os.makedirs(save_path)
    writer = pd.ExcelWriter('{0}/makespan{1}.xlsx'.format(save_path, str_time))
    writer_time = pd.ExcelWriter('{0}/time_{1}.xlsx'.format(save_path, str_time))
    file_name = [test_files[i] for i in range(num_ins)]
    data_file = pd.DataFrame(file_name, columns=["file_name"])
    data_file.to_excel(writer, sheet_name='Sheet1', index=False,startrow= 0)
    writer.save()
    writer.close()
    data_file.to_excel(writer_time, sheet_name='Sheet1', index=False)
    writer_time.save()
    writer_time.close()

    # Rule-by-rule (model-by-model) testing
    start = time.time()
    for i_rules in range(0,len(rules)):
        rule = rules[i_rules]
        print('\nrule:', rule)
        # Load trained model
        if device.type == 'cuda':
            model_CKPT = torch.load('./model/' + mod_files[i_rules])
        else:
            model_CKPT = torch.load('./model/' + mod_files[i_rules], map_location='cpu')
        model.policy_old.load_state_dict(model_CKPT)

        # Schedule instance by instance
        step_time_last = time.time()
        makespans = []
        times = []
        for i_ins in range(num_ins):
            test_file = data_path + test_files[i_ins]
            print(test_file)
            with open(test_file) as file_object:
                line = file_object.readlines()
                ins_num_jobs, ins_num_mas, _ = nums_detec(line)
            env_test_paras["num_jobs"] = ins_num_jobs #
            env_test_paras["num_mas"] = ins_num_mas

            # Environment object already exists
            if len(envs) == num_ins:
                env = envs[i_ins]
            # Create environment object
            else:
                if test_paras["sample"]:
                    env = gym.make('fjsp-v0', case=[test_file] * test_paras["num_sample"],
                                   env_paras=env_test_paras, data_source='file')
                    env.reset()
                else:
                    env = gym.make('fjsp-v0', case=[test_file], env_paras=env_test_paras, data_source='file')
                    env.reset()
                envs.append(copy.deepcopy(env))
                #print("Create env[{0}]".format(i_ins))
            if test_paras["sample"]:
                makespan, t, _ = schedule(env, model, memories, flag_sample=test_paras["sample"])
                print("makespan: ", torch.min(makespan))
                makespans.append(torch.min(makespan))
                times.append(t)
            # DRL-G
            else:
                makespan, t,_ = schedule(env, model, memories)
                makespans.append(makespan)
                times.append(t)
            #print("finish env {0}".format(i_ins))
        #print("rule_spend_time: ", time.time() - step_time_last)

        # Save makespan and time data to files
        data = pd.DataFrame(torch.tensor(makespans).t().tolist(), columns=[rule])
        data.to_excel(writer, sheet_name='Sheet1', index=False,startrow=0, startcol=i_rules + 1)
        writer.save()
        writer.close()
        data = pd.DataFrame(torch.tensor(times).t().tolist(), columns=[rule])
        data.to_excel(writer_time, sheet_name='Sheet1', index=False,startrow=0,startcol=i_rules + 1)
        writer_time.save()
        writer_time.close()
        ''''''
        for env in envs:
            env.reset()
    #print("total_spend_time: ", time.time() - start)

def schedule(env, model, memories, flag_sample=False):
    # Get state and completion signal
    state = env.state
    dones = env.done_batch
    done = False  # Unfinished at the beginning
    start_time = time.time()
    while ~done:
        with torch.no_grad():
            actions = model.policy_old.act(state, memories, dones, flag_sample=flag_sample, flag_train=False)
        state, rewards, dones,_ = env.step(actions)  # environment transit
        done = dones.all()
    spend_time = time.time() - start_time  # The time taken to solve this environment (instance)
    #print("spend_time: ", spend_time)
    #env.render()
    return copy.deepcopy(env.makespan_batch), spend_time,[]


if __name__ == '__main__':
    main()