from torch import nn
import torch
from Policy import PolicyNet
from torch.optim import Adam
from Policy import REINFORCE
from env import Crowdsourcing
from DQN import QNetwork
class ActorCritic:
    def __init__(self, actor: nn.Module,
                 critic: nn.Module,
                 agent:REINFORCE,
                 env:Crowdsourcing,
                 epoch:int = 1000,
                 device:str = "cpu"
                 ):
        self.actor = actor
        self.critic = critic
        self.opt_actor = Adam(self.actor.parameters(), lr=0.001, weight_decay=0.003)
        self.opt_critic = Adam(self.critic.parameters(), lr=0.001, weight_decay=0.003)
        self.agent = agent
        self.env = env
        self.criteria = nn.MSELoss()
        self.epoch = epoch
        self.device = device

    def train(self):
        for e in range(self.epoch):
            state_worker, state_proj = self.env.reset()
            state_worker = state_worker.to(self.device)
            state_proj = state_proj.to(self.device)
            episode_return = 0
            while True:
                action = self.agent.take_action(state_worker, state_proj)
                nxt_state_worker, nxt_state_proj, reward, done,_ = self.env.step(action)
                prob = self.actor(state_worker, state_proj, self.agent.worker_feature, self.agent.proj_feature)
                prob = prob.reshape(-1)
                state = torch.cat([state_worker,state_proj])
                q_val = self.critic(state)
                log_prob = torch.sum(torch.log(prob))
                loss_actor = -log_prob * q_val
                self.opt_actor.zero_grad()
                loss_actor.backward()
                self.opt_actor.step()
                nxt_state = torch.cat([nxt_state_worker,nxt_state_proj]).to(self.device)
                q_val_nxt = self.critic(nxt_state)
                loss_critic = self.criteria(q_val_nxt+reward, q_val)
                self.opt_critic.zero_grad()
                loss_critic.backward()
                self.opt_critic.step()
                episode_return += reward
                if done:break
            print(f'Episode {e} return {episode_return}')

if __name__ == '__main__':
    env = Crowdsourcing()
    pro_num = env.proj_num
    worker_num = env.worker_num
    worker_f_dim = env.worker_info.shape[1]
    proj_f_dim = env.proj_info.shape[1]
    actor = PolicyNet(worker_f_dim, proj_f_dim, hidden_dim=16)
    critic = QNetwork(pro_num + worker_num, 1000, 1)
    agent = REINFORCE(actor, critic)
    trainer = ActorCritic(actor, critic, agent, env,device="mps")
    trainer.train()








