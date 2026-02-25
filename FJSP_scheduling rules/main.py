from FJSP_Env import FJSP
import time
from torch.utils.data import DataLoader
import os
from DataRead import getdata
import numpy as np

def runDRs(rule, file):
    batch_size = 1
    Data = getdata(file)
    n_j = Data['n']
    n_m = Data['m']
    num_operation = []
    num_operations = []
    for i in Data['J']:
        num_operation.append(Data['OJ'][i][-1])
    num_operation_max = np.array(num_operation).max()

    time_window = np.zeros(shape=(Data['n'], num_operation_max, Data['m']))

    data_set = []
    for i in range(Data['n']):

        for j in Data['OJ'][i + 1]:
            mchForJob = Data['operations_machines'][(i + 1, j)]
            for k in mchForJob:
                time_window[i][j - 1][k - 1] = Data['operations_times'][(i + 1, j, k)]

    for i in range(batch_size):
        num_operations.append(num_operation)
        data_set.append(time_window)
    num_operation = np.array(num_operations)

    train_dataset = np.array(data_set)
    # ------------------------------------------------------------------
    data_loader = DataLoader(train_dataset, batch_size=batch_size)
    result = []
    for batch_idx, data_set in enumerate(data_loader):
        data_set = data_set.numpy()
        batch_size = data_set.shape[0]

        env = FJSP(n_j=n_j, n_m=n_m, EachJob_num_operation=num_operation)


        # random rollout to test
        adj, _, omega, mask, mch_mask, _, mch_time, _ = env.reset(data_set, rule)

        # job = omega #
        rewards = []
        # flags = [] #

        d = 0
        while True:
            action = []
            mch_a = []
            for i in range(batch_size):
                a = np.random.choice(omega[i][np.where(mask[i] == 0)])
                row = np.where(a <= env.last_col[i])[0][0]
                col = a - env.first_col[i][row]
                m = np.random.choice(np.where(mch_mask[i][row][col] == 0)[0])
                action.append(a)
                mch_a.append(m)
            d += 1

            adj, _, reward, done, omega, mask, job, mch_mask, mch_time, _ = env.step(action, mch_a)
            rewards.append(reward)

            if env.done():
                break

        result.append(env.mchsEndTimes.max(-1).max(-1))
    return np.array(result).mean()


# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    path = './1_Brandimarte'
    path_list = os.listdir(path)
    path_list.sort(key=lambda x: int(x[13:-4]))  #filename
    filename = []
    num=10 #
    for fikena in path_list[0:num]:
        filename.append(os.path.join(path, fikena))

    DRs = ['FIFO_SPT', 'FIFO_EET', 'MOPNR_SPT', 'MOPNR_EET', 'LWKR_SPT', 'LWKR_EET', 'MWKR_SPT', 'MWKR_EET']
    i=1
    result = [] #makespan
    t=[]         #time

    for file in filename:
        starttime=time.time()
        a = runDRs(DRs[i], file)
        endtime=time.time()
        result.append(a)
        t.append(endtime-starttime)
        print(file)
    print(result)
    print(t)
    
