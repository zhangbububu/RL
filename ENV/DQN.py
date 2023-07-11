from torch import nn
from env import Crowdsourcing
import torch
from tqdm import tqdm


class QNetwork(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int):
        super(QNetwork,self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Trainer:
    def __init__(self, Q_network: QNetwork,
                 env: Crowdsourcing,
                 epochs: int = 1000,
                 device: str = 'cuda'):
        self.q = Q_network.to(device)
        self.env = env
        self.state_worker, self.state_proj = self.env.reset()
        self.state = torch.cat([self.state_worker, self.state_proj])
        self.gamma = 0.9
        self.criteria = nn.MSELoss()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=0.001, weight_decay=0.003)
        self.epoch = epochs
        self.device = device


    def train(self):
        for e in tqdm(range(self.epoch)):
            episode_return = 0
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            self.state_worker, self.state_proj = self.env.reset()
            self.state = torch.cat([self.state_worker, self.state_proj]).to(self.device)
            while True:
                q_val = self.q(self.state)
                indices = torch.argmax(q_val)
                # print(q_val.shape)
                # print(f'{indices=}')
                nxt_state_worker,nxt_state_proj,reward,f,_ = self.env.step(indices)
                nxt_state = torch.cat([nxt_state_worker, nxt_state_proj]).to(self.device)
                q_val_nxt = self.q(nxt_state)
                q_val_nxt_max = torch.max(q_val_nxt)
                loss = self.criteria(q_val[indices], reward + self.gamma * q_val_nxt_max)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                self.state = nxt_state
                episode_return += reward
                # print(f)
                if f:break
            print(f'Episode {e} return {episode_return}')




if __name__ == '__main__':
    env = Crowdsourcing()
    pro_num = env.proj_num
    worker_num = env.worker_num
    q = QNetwork(pro_num+worker_num, 1000, pro_num*worker_num)
    trainer = Trainer(q, env)
    trainer.train()