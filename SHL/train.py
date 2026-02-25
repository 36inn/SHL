import copy
import json
import os
import random
import time
from collections import deque
import gym
import pandas as pd
import torch
import numpy as np

import PPO_model
from env.case_generator import CaseGenerator
from validate import validate, get_validate_env

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # PyTorch initialization
    # gpu_tracker = MemTracker()  # Used to monitor memory (of gpu)
    setup_seed(3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)
    torch.set_num_threads(2)

    # Load config and init objects
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    valid_paras=  load_dict["valid_paras"]
    env_paras["device"] = device
    model_paras["device"] = device
    valid_paras["device"] = device
    model_paras["actor_in_dim"] = model_paras["out_dim"] * 2 + model_paras["out_dim"] * 2
    model_paras["critic_in_dim"] = model_paras["out_dim"] + model_paras["out_dim"]

    num_jobs = env_paras["num_jobs"]
    num_mas = env_paras["num_mas"]
    opes_per_job_min = int(num_mas * 0.8)
    opes_per_job_max = int(num_mas * 1.2)

    memories = PPO_model.Memory() #
    model = PPO_model.PPO(model_paras, train_paras, num_envs=env_paras["batch_size"])
    env_valid = get_validate_env(valid_paras)  # Create an environment for validation
    valid_paras1=copy.deepcopy(valid_paras)
    valid_paras1["num_jobs"]=15
    valid_paras1["num_mas"] = 8
    env_valid1 = get_validate_env(valid_paras1)
    valid_paras2 = copy.deepcopy(valid_paras)
    valid_paras2["num_jobs"] = 20
    valid_paras2["num_mas"] = 10
    env_valid2 = get_validate_env(valid_paras2)
    maxlen = 1  # Save the best model
    best_models = deque()
    makespan_best = float('inf')

    # Use visdom to visualize the training process
    '''
    is_viz = train_paras["viz"]
    if is_viz:
        viz = Visdom(env=train_paras["viz_name"])
    '''
    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './save/train_{0}'.format(str_time)
    os.makedirs(save_path)
    # Training curve storage path (average of validation set)
    writer_ave = pd.ExcelWriter('{0}/training_ave_{1}.xlsx'.format(save_path, str_time))
    valid_results = []
    data_file = pd.DataFrame(np.arange(train_paras["save_timestep"], train_paras["save_timestep"]+train_paras["max_iterations"], train_paras["save_timestep"]), columns=["iterations"])
    data_file.to_excel(writer_ave, sheet_name='Sheet1', index=False)
    writer_ave.save()
    writer_ave.close()


    # Start training iteration
    start_time = time.time()
    env = None
    for i in range(1, train_paras["max_iterations"]+1):

        if (i - 1) % train_paras["parallel_iter"] == 0:
            if i<=200:
                pass
            elif i<=400:
                num_jobs=15
                num_mas=8
                env_paras["num_jobs"] = num_jobs
                env_paras["num_mas"] = num_mas
            else:
                num_jobs = 20
                num_mas = 10
                env_paras["num_jobs"] = num_jobs
                env_paras["num_mas"] = num_mas
            nums_ope = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
            case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope)
            env = gym.make('fjsp-v0', case=case, env_paras=env_paras)
            env.reset()
            print('num_job: ', num_jobs, '\tnum_mas: ', num_mas, '\tnum_opes: ', sum(nums_ope))


        # Get state and completion signal
        state = env.state
        done = False
        dones = env.done_batch
        last_time = time.time()

        # Schedule in parallel
        while ~done:
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, dones)
            state, rewards, dones,_ = env.step(actions)
            done = dones.all()
            memories.rewards.append(rewards)
            memories.is_terminals.append(dones)
            # gpu_tracker.track()  # Used to monitor memory (of gpu)
        print("spend_time: ", time.time()-last_time)


        env.reset()

        # if iter mod x = 0 then update the policy
        if i % train_paras["update_timestep"] == 0:
            loss, reward = model.update(memories, env_paras, train_paras)
            print("reward: ", '%.3f' % reward, "; loss: ", '%.3f' % loss)
            memories.clear_memory()

        # if iter mod x = 0 then validate the policy
        if i % train_paras["save_timestep"] == 0 :
            print('\nStart validating')
            # Record the average results and the results on each instance
            vali_result, vali_result_100 = validate(valid_paras, env_valid, model.policy_old)
            vali_result1, _ = validate(valid_paras1, env_valid1, model.policy_old)
            vali_result2, _ = validate(valid_paras2, env_valid2, model.policy_old)
            vali_result=(vali_result+vali_result1+vali_result2)/3
            print('valid_results:',i,vali_result)
            valid_results.append(vali_result.item())


            # Save the best model
            if vali_result < makespan_best:
                makespan_best = vali_result
                if len(best_models) == maxlen:
                    delete_file = best_models.popleft()
                    os.remove(delete_file)
                save_file = '{0}/save_best_{1}_{2}_{3}.pt'.format(save_path, num_jobs, num_mas, i)
                best_models.append(save_file)
                torch.save(model.policy.state_dict(), save_file)

    # Save the data of training curve to files
    data = pd.DataFrame(np.array(valid_results).transpose(), columns=["res"])
    data.to_excel(writer_ave, sheet_name='Sheet1', index=False, startcol=1)
    writer_ave.save()
    writer_ave.close()

    print("total_time: ", time.time() - start_time)

if __name__ == '__main__':
    main()