from ast import List
import torch
import tqdm



from env import Crowdsourcing

class PolicyNet(torch.nn.Module):
    def __init__(self, worker_f_dim, proj_f_dim,hidden_dim):
        super(PolicyNet, self).__init__()

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(worker_f_dim+1, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),

        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(proj_f_dim+1, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self,  worker_state, proj_state,  worker_f, proj_f):

        worker_num = worker_state.shape[0]
        proj_num =proj_state.shape[0]

        # proj_state: 00110101 (proj_num, )
        # proj_feature:    (proj_num, feature_dim)

        worker_f = torch.concat([worker_f, worker_state.reshape(-1,1)],dim=-1) 
        proj_f = torch.concat([proj_f, proj_state.reshape(-1,1)],dim=-1) 
        worker_f = worker_f.cuda()
        proj_f = proj_f.cuda()

        worker_f = self.fc1(worker_f)
        proj_f = self.fc2(proj_f)

        h = worker_f @ proj_f.permute(1,0)
        # worker_state[2:3] = 1
        h[worker_state==1] = -1e9
        # print('[1] h.shape = ',h.shape)

        h = torch.nn.functional.softmax(h.reshape(-1), dim=-1).reshape(worker_num, proj_num)
        # print('[2] h.shape = ',h.shape)

        return h



class REINFORCE:
    def __init__(self, worker_f_dim, proj_f_dim, worker_state_dim, proj_state_dim,
                 worker_feature, proj_feature, hidden_dim,gamma):
                 
        self.policy_net = PolicyNet(worker_f_dim, proj_f_dim, hidden_dim)
        self.policy_net.cuda()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate, weight_decay=0.003)  # 使用Adam优化器

        self.gamma = gamma  # 折扣因子
        self.worker_feature = worker_feature.cuda()
        self.proj_feature = proj_feature.cuda()


        # self.device = device

    def take_action(self, state_worker, state_proj):  # 根据动作概率分布随机采样
        '''
        @state_worker (worker_num,)  Tensor
        @state_proj   (proj_num, ) . Tensor
        
        '''
        state_worker = state_worker.cuda()
        state_proj = state_proj.cuda()
        probs = self.policy_net(state_worker, state_proj, self.worker_feature, self.proj_feature)

        action_dist = torch.distributions.Categorical(probs.reshape(-1))
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state_worker, state_proj = state_list[i]
            state_worker = state_worker.cuda()
            state_proj = state_proj.cuda()
            action = torch.tensor(action_list[i]).cuda()
            prob = self.policy_net(state_worker, state_proj, self.worker_feature, self.proj_feature)
            prob = prob.reshape(-1)
            t = prob[action]
            log_prob = torch.log(t)

            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降





learning_rate = 0.002
hidden_dim = 16
gamma = 0.8

env = Crowdsourcing()
env.reset()
worker_f_dim=env.worker_info.shape[1]
proj_f_dim=env.proj_info.shape[1]
worker_state_dim=env.worker_num
proj_state_dim = env.proj_num

agent = REINFORCE(worker_f_dim, proj_f_dim, worker_state_dim, proj_state_dim,
                  env.worker_info, env.proj_info,hidden_dim,gamma)


return_list = []


epochs = 1000
for i in tqdm.tqdm(range(epochs)):
    episode_return = 0
    transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': []
    }
    state_worker, state_proj = env.reset()
    done = False
    while not done:
        action = agent.take_action(state_worker, state_proj)
        next_state_worker, next_state_proj , reward, done, _ = env.step(action)
        transition_dict['states'].append((state_worker, state_proj))
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append((next_state_worker, next_state_proj))
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)
        # print(done)
        state_worker = next_state_worker
        state_proj = next_state_proj
        episode_return += reward

    return_list.append(episode_return)
    agent.update(transition_dict)
    print(episode_return)

for i in return_list:
    print(i)

