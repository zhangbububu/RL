import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler,MinMaxScaler
class Crowdsourcing:
    

    def __init__(self,):
        '''

        worker

            state : 0000 (有没有被推荐任务)

            reward:  
                    [任务是不是第一次被拿] * [project的奖金]  *  [woker的兴趣 与 proj的相关性] 

            action: 第proj——id个proj 分给  第work-id个worker

        proj
            state: 0000 (是不是第一次推荐给别人). proj的状态, 0/1。 worker只能被分配一个proj，但是一个proj能分给多个worker
                    因此，我们设定proj的奖金只给第一个被分配的worker。

            reward:
                    [woker的兴趣 与 proj的相关性] * worker_quality

            action: 第proj——id个proj 分给  第work-id个worker

        策略 pi：
            pi(0000,000) -----》 (proj to worker)[action]
        例如：
            pi(0000,000) -----》 (proj0 -> woker1) [action]
            如何两者的状态，输出action

        Q函数：
            Q((p_state, w_state), action) = reward

        流程：
            s*:状态
            a*：action
            r*: reward

            # (0000,000) s0
            # (0100,100) s1 a1 r1
            # (0110,110) s2 a2 r2
            # (0111,111) s3 a3 r3
            # (1111,111) s4 a4 r4
            # END

            得到：（s1,a1,r1,s2,a2,r2,s3,a3,r3............)


        Data
            worker的特征：
                 兴趣, qualit   
            proj的特征:
                 domain(与worker中兴趣的意义相同), award. 
            具体看 project-info.csv and  worker.info.csv

        --------------------------------------------------------------------------

        交互环境
            step() 
            reset()
        '''
        super(Crowdsourcing, self,).__init__()
        self.worker_num = 500
        self.proj_num = 500
        self.reset() 

    def reset(self,):
        '''
        重置环境

        return :
            woker的初始状态
            proj的初始状态
        '''

        data_worker = pd.read_csv('/public_data/zba/RL/worker_info.csv')
        data_worker = data_worker.values[:,2:]
        index = np.random.randint(0,len(data_worker)-self.worker_num)
        self.worker_info = data_worker[index:index+self.worker_num] # worker的特征,具体查看woker-info文件，shape为（worker_num, 2）
        self.worker_domain = self.worker_info[:,0] # worker的兴趣
        self.worker_quality = self.worker_info[:,1] # worker的质量

        self.worker_info = torch.tensor(self.worker_info, dtype=torch.float32) 
        self.worker_info[:,0] /= 10
        data_proj = pd.read_csv('/public_data/zba/RL/projects_info.csv')
        data_proj = data_proj.values[:,2:]
        index = np.random.randint(0,len(data_proj)-self.proj_num)
        self.proj_info = data_proj[index:index+self.proj_num] #proj的特征,具体看proj—info文件, shape为（worker_num, 2）
        self.proj_award = self.proj_info[:,0]  # proj的奖金
        self.proj_domain = self.proj_info[:,1] # proj的类型，与 worker的兴趣 意义相同

        self.proj_info = torch.tensor(self.proj_info, dtype=torch.float32) # proj的特征
        self.proj_info[:,0] /= 1000
        self.proj_info[:,1] /= 10

        self.state_worker = torch.zeros(self.worker_num) # worker的状态, 0/1值，代表目前该woker有没有被分配proj

        self.state_proj = torch.zeros(self.proj_num)     #  proj的状态, 0/1。 worker只能被分配一个proj，但是一个proj
                                                         #  能分给多个worker，因此，我们设定proj的奖金只给第一个被分配的worker。



        return self.state_worker.clone(), self.state_proj.clone()


    def parse(self, action):
        '''
        将action映射为 worker的编号 和 proj的编号
            action的值域为【worker_num x proj_num】
            则action//proj_num为worker的编号
            action%proj_num 为proj的编号
        '''
        woker = action//self.proj_num
        proj = action % self.proj_num

        return woker, proj

    def calc_reward(self,worker_id, proj_id):
        '''
        计算奖励值：
        对于某一个动作
            worker获得奖励值为：     [任务是不是第一次被拿] * [project的奖金]  *  [woker的兴趣 与 proj的相关性] 

            proj获得奖励值为：   [woker的兴趣 与 proj的相关性] * worker_quality
        则总奖励为两只之和

        设定 [woker的兴趣 与 proj的相关性] 这一项为：
            当两者的domain相同时，系数为5，否则系数为1.

        '''
        r_worker  = self.state_proj[proj_id] * \
                     (5 if self.worker_quality[worker_id]==1 else self.proj_domain[proj_id]) *\
                     self.proj_award[proj_id]

        r_proj = (5 if self.worker_quality[worker_id]==1 else self.proj_domain[proj_id]) * self.worker_quality[worker_id]
        
        return r_worker + r_proj



    def step(self,action) :
        '''
        给该环境输入动作 action，更新环境状态：

        并返回：
            self.state_worker.clone() worker的下一步状态
            self.state_proj.clone()   proj的下一步状态
            reward 本次action获得的奖励
            done 是否结束（到所有的worker都被分到proj时结束）

        '''

        worker_id, proj_id = self.parse(action) 
        #worker——id 和 proj-id代表了本次的action，将编号为proj-id的proj 分配给了 编号为worker-id的worker、
        reward = self.calc_reward(worker_id, proj_id)
        self.state_worker[worker_id] = 1
        self.state_proj[proj_id] = 1

        if sum(self.state_worker) == len(self.state_worker):
            done = True
        else :
            done = False
        # print('check end')

        return self.state_worker.clone(), self.state_proj.clone(), reward, done, None
        



env = Crowdsourcing()
env.reset()
