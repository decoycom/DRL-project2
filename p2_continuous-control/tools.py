import numpy as np
import pandas as pd
import torch
from time import sleep

from deep_rl import *
from agent import *

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from joblib import Parallel, delayed

import os
base_path = os.getcwd()

# === [ Brains ] ===

def brain(brain_name,agent,config,env, it=0):
    n_episodes = config.eval_episodes # 2000, 
    max_t = config.max_steps # 1500, 
    # eps_start=1.0, 
    # eps_end=0.01, 
    # eps_decay=0.995
    window = getattr(config, 'scores_window', 100)
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=window)  # last 100 scores
    # eps = eps_start                    # initialize epsilon

    pritty = getattr(config, 'pritty_fields', {})
    st = ''
    for k in config.update_fields:
        v = getattr(config, k)
        if k in pritty:
            k = pritty[k]
        if float(v).is_integer():
            st += '{}:{}\t'.format(k,v)
        else:
            st += '{}:{:.8f}\t'.format(k,v)
    print('\rStart[{}]\t{}'.format(it,st), flush = True)

    no_reg = getattr(config, 'stop_regression', True)
    max_reg = getattr(config, 'max_regression', 0.5)
    perc_reg = getattr(config, 'perc_regression', 1)
    s_margin = getattr(config, 'save_margin', 32.0)
    s_postfix = getattr(config, 'save_postfix', '')    
    win_mean = 0
    last_win = win_mean
    max_win = win_mean
    max_mean = 0
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]
        
        score = 0
        for t in range(max_t):
            action = agent.act(state) # eps
            
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
    
            agent.step(state, action, reward, next_state, done)
        
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        
        win_mean = np.mean(scores_window)
        max_mean = max(max_mean,win_mean)
        # eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode[{}] {}\t{} a-Average Score: {:.2f}'.format(it,i_episode,st,win_mean), end="")
        if i_episode % window == 0:
            if no_reg and max_win - (max_win / 100.0 * perc_reg) > win_mean :
                print('\rFinished![{}] {}\t{} a-Average Score: {:.2f}'.format(it, i_episode,st,win_mean), flush = True)
                return (scores, win_mean,False,max_win,max_mean)
            else:
                print('\rEpisode[{}] {}\t{} a-Average Score: {:.2f}'.format(it, i_episode,st,win_mean), flush = True)
            last_win = win_mean
            max_win = max(max_win,win_mean)
    
        if np.mean(scores_window)>=31.0:
            print('\nEnvironment[{}] solved in {:d} episodes!\t{} Average Score: {:.2f}'.format(it,i_episode-window,st, win_mean), flush = True)
            torch.save(agent.network.state_dict(), 'checkpoint_{}{}.pth'.format(it,s_postfix))
            return (scores, win_mean,True,max_win,max_mean)
            
    return (scores, win_mean,False,max_win,max_mean)

def brain_multy(brain_name,agent,config,env, it=0):
    number = config.num_workers
    n_episodes = config.eval_episodes # 2000, 
    max_t = config.max_steps # 1500, 
    window = getattr(config, 'scores_window', 100)
    
    scores = []                        # list containing scores from each episode
    scores_window = []  # last 100 scores
    
    for i in range(number):
#         scores.append([])
        scores_window.append(deque(maxlen=window))
    
    # eps = eps_start                    # initialize epsilon

    pritty = getattr(config, 'pritty_fields', {})
    st = ''
    for k in config.update_fields:
        v = getattr(config, k)
        if k in pritty:
            k = pritty[k]
        if float(v).is_integer():
            st += '{}:{}\t'.format(k,v)
        else:
            st += '{}:{:.8f}\t'.format(k,v)
    print('\rStart[{}]\t{}'.format(it,st), flush = True)
#     , end=""

#     next_states, rewards, terminals, info
    def step_fn(actions):
        env_info = env.step(actions)[brain_name]        # send the action to the environment
        next_states = env_info.vector_observations   # get the next state
        rewards = env_info.rewards                   # get the reward
        dones = env_info.local_done                  # see if episode has finished
        dones = np.asarray(dones, dtype=np.int32)
        return (next_states,rewards,dones,env_info)
        
#     = lambda action: torch.optim.Adam(params, lr=config.actor_lr), #, weight_decay=config.weight_decay

    no_reg = getattr(config, 'stop_regression', True)
    max_reg = getattr(config, 'max_regression', 0.5)
    perc_reg = getattr(config, 'perc_regression', 5)
    s_margin = getattr(config, 'save_margin', 32.0)
    s_postfix = getattr(config, 'save_postfix', '')
    
    win_mean = 0
    last_win = win_mean
    max_win = win_mean
    max_mean = 0
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        states = env_info.vector_observations
#         print("shape:{}".format(states.shape))
#         print(states)
#         break
        score = np.zeros(number)
        for t in range(max_t):
            
            states,rewards,dones = agent.learn(states,step_fn)         
            score = score + rewards
            if np.any(dones):
                break 
        for i in range(number):
            scores_window[i].append(score[i])       # save most recent score
        scores.append(np.mean(score))              # save most recent score
        
        win_mean = np.mean(scores_window)
        max_mean = max(max_mean,win_mean)
        # eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode[{}] {}\t{} a-Average Score: {:.2f}'.format(it, i_episode,st,win_mean), end="")
        if i_episode % window == 0:
            if no_reg and max_win - (max_win / 100.0 * perc_reg) > win_mean :
                print('\rFinished![{}] {}\t{} a-Average Score: {:.2f}'.format(it, i_episode,st,win_mean), flush = True)
                return (scores, win_mean,False,max_win,max_mean)
            else:
                print('\rEpisode[{}] {}\t{} a-Average Score: {:.2f}'.format(it, i_episode,st,win_mean), flush = True)
            last_win = win_mean
            max_win = max(max_win,win_mean)
    
        if win_mean>=s_margin:
            print('\nEnvironment[{}] solved in {:d} episodes!\t{} Average Score: {:.2f}'.format(it,i_episode-window,st, win_mean), flush = True)
            torch.save(agent.network.state_dict(), 'checkpoint_{}{}.pth'.format(it,s_postfix))
            return (scores, win_mean,True,max_win,max_mean)
            
    return (scores, win_mean,False,max_win,max_mean)

# === [ Strategies ] ===

def initConf_ddpg(state_size,action_size,brain_name,env):
    # select_device(0)

    config = Config()

    config.update_fields = [
        'fc1','fc2','fc3',
        'weight_decay_act','weight_decay',
        'actor_lr','critic_lr',
        'target_network_mix','discount',
        'gradient_clip',
        # 'act_clip',
    ]
    config.pritty_fields = {'weight_decay_act':'a_W','weight_decay':'c_W','actor_lr':'a_Lr','critic_lr':'c_Lr','target_network_mix':'tau','gradient_clip':'c_Clip','act_clip':'a_Clip','discount':'G'}
    # config.merge(kwargs)
    config.device = Config.DEVICE
    # print("config.device:{}".format(config.device))
    config.brain_name = brain_name
    config.seed = 0    

    config.fc1 = 300
    config.fc2 = 200
    config.fc3 = 100

    config.eval_episodes = 200
    config.max_steps = int(1e7)
    config.batch_size = 64
    
    config.gradient_clip = 5
    config.eval_interval = 4
    config.memory_size = int(1e6)            
    config.warm_up = config.batch_size

    # num_agents
    config.state_dim = state_size
    config.action_dim = action_size
    # 400, 300
    config.network_fn = lambda cfg: DeterministicActorCriticNet(
        cfg.state_dim, cfg.action_dim,
        actor_body=FCBody(cfg.state_dim, (cfg.fc1, cfg.fc2, cfg.fc3), gate=F.relu),
        critic_body=FCBody(cfg.state_dim + cfg.action_dim, (cfg.fc1, cfg.fc2, cfg.fc3), gate=F.relu),

        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=cfg.actor_lr, weight_decay=cfg.weight_decay_act),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=cfg.critic_lr, weight_decay=cfg.weight_decay))

    # self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)    
    config.replay_fn = lambda: UniformReplay(memory_size=config.memory_size, batch_size=config.batch_size)

    config.agent_fn = lambda conf: DDPGAg(conf)
    config.brain_fn = lambda cfg,i: brain(cfg.brain_name,cfg.agent_fn(cfg),cfg,env,it=i)

    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2)) 
    return config

def initConf_td3(state_size,action_size,brain_name,env):
    
    # select_device(0)
    config = Config()
    config.brain_name = brain_name

    config.update_fields = ['fc1','fc2','fc3','weight_decay','weight_decay_act','actor_lr','critic_lr','target_network_mix','discount','gradient_clip','td3_noise','td3_noise_clip','td3_delay']
    config.pritty_fields = {'weight_decay_act':'a_W','weight_decay':'c_W','actor_lr':'a_Lr','critic_lr':'c_Lr','target_network_mix':'tau','gradient_clip':'c_Clip','act_clip':'a_Clip','discount':'G', 'td3_noise':'N','td3_noise_clip':'N_Clip','td3_delay':'Delay'}
    #,'fc3'
    config.fc1 = 300
    config.fc2 = 200
    config.fc3 = 100

    config.device = Config.DEVICE
    config.state_dim = state_size
    config.action_dim = action_size

    config.max_steps = int(1e6)
    config.eval_episodes = 128
    config.batch_size = 128

    config.eval_interval = int(1e4)
    config.memory_size=int(1e5)

    config.gradient_clip = 5
    config.act_clip = 9
    config.weight_decay=0.012
    config.actor_lr = 1e-4
    config.critic_lr = 1e-4
    config.target_network_mix = 5e-3
    config.discount = 0.98

    config.td3_noise = 0.2
    config.td3_noise_clip = 0.5
    config.td3_delay = 2

    config.network_fn = lambda cfg: TD3Net(
        cfg.action_dim,
        actor_body_fn=lambda: FCBody(cfg.state_dim, (cfg.fc1, cfg.fc2, cfg.fc3), gate=F.relu),
        critic_body_fn=lambda: FCBody(cfg.state_dim + cfg.action_dim, (cfg.fc1, cfg.fc2, cfg.fc3), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=cfg.actor_lr, weight_decay=cfg.weight_decay_act),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=cfg.critic_lr,weight_decay=cfg.weight_decay))
    #     .to(config.device)

    replay_kwargs = dict(
        memory_size=config.memory_size,
        batch_size=config.batch_size,
    )

    config.replay_fn = lambda: ReplayWrapper(UniformReplay, replay_kwargs, async=False)
    config.random_process_fn = lambda: GaussianProcess(
        size=(config.action_dim,), std=LinearSchedule(0.1))

    config.warm_up = int(1e4)

    config.agent_fn = lambda conf: TD3Ag(conf)
    config.brain_fn = lambda cfg,i: brain(cfg.brain_name,cfg.agent_fn(cfg),cfg,env,it=i)
    
    return config

def a2c_feature(state_size,action_size,brain_name,env):
    config = Config()
    config.brain_name = brain_name

    config.update_fields = ['fc1','fc2','rollout_length', 'lr', 'gae_tau', 'entropy_weight', 'discount','gradient_clip','use_gae']
    config.pritty_fields = {'rollout_length':'rout_L','gae_tau':'G_tau','entropy_weight':'e_W','use_gae':'G_use','gradient_clip':'Clip','discount':'G'}
    config.max_steps = int(2e7)

#     config.eval_interval = int(1e4)
    config.eval_episodes = 400
    config.batch_size = 128
#     config.memory_size=int(1e6)
    
    config.device = Config.DEVICE
    config.state_dim = state_size
    config.action_dim = action_size
    # {'rollout_length': 8, 'discount': 0.9751201238174458, 'gradient_clip': 7, 'lr': 1.0802107431577493e-05, 'gae_tau': 0.9964673379102222, 'entropy_weight': 0.0015039861493695169, 'use_gae': False}
    # {'rollout_length': 10, 'discount': 0.9847894774762507, 'gradient_clip': 2, 'lr': 1.0071160478803324e-05, 'gae_tau': 0.9899540738644809, 'entropy_weight': 0.0011836478393662721, 'use_gae': True},
    config.discount = 0.9751201238174458
    config.use_gae = False
    config.gae_tau = 0.9964673379102222
    config.entropy_weight = 0.0015039861493695169
    config.rollout_length = 8
    config.gradient_clip = 7
    config.lr = 1.0802107431577493e-05

    config.num_workers = 20
    
    config.fc1 = 64
    config.fc2 = 64
    
    config.optimizer_fn = lambda params,cfg: torch.optim.RMSprop(params, lr=cfg.lr)
    config.network_fn = lambda cfg: GaussianActorCriticNet(
        cfg.state_dim, cfg.action_dim,
        actor_body=FCBody(cfg.state_dim, (cfg.fc1, cfg.fc2)), critic_body=FCBody(cfg.state_dim, (cfg.fc1, cfg.fc2)))
        
#     config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
#     config.network_fn = lambda: CategoricalActorCriticNet(
#         config.state_dim, config.action_dim, FCBody(config.state_dim, gate=F.tanh))
    
    config.storage_fn = lambda:  Storage(config.rollout_length)
    
    config.agent_fn = lambda cfg: A2CAg(cfg)
    config.brain_fn = lambda cfg,i: brain_multy(brain_name,cfg.agent_fn(cfg),cfg,env,it=i)
    
    return config

# === [ Tuners ] ===

def tune(
    config : Config,
    space_cfg, 
    max_iter=10, 
    hebo_cfg = None,
    greater_is_better : bool = True,
    verbose  = True,
    first = [],
    **kwargs
    ):

    if hebo_cfg is None:
        hebo_cfg = {}
    space = DesignSpace().parse(space_cfg)
    opt   = HEBO(space, **hebo_cfg)
    
    scoresz = []
    first_index = 0
    for i in range(max_iter):
        if len(first) > first_index:
            hyp = first[first_index]
            first_index += 1
            details = {}
            for n,v in hyp.items():
                details[n] = [v]
            rec = pd.DataFrame(details)
        else:
            rec     = opt.suggest()
            hyp     = rec.iloc[0].to_dict()
        for k in config.update_fields:
#         for k in hyp:
            if space.paras[k].is_numeric and space.paras[k].is_discrete:
#                 hyp[k] = int(hyp[k])
                setattr(config, k, int(hyp[k]))
            else:
                setattr(config, k, hyp[k])

        scores,reward,done,max_win,max_mean = config.brain_fn(config,i)
        # brain(brain_name,a,config,env)
        scoresz.append(scores)

        sign    = -1. if greater_is_better else 1.0
        opt.observe(rec, np.array([sign * reward]))
        if verbose:
            print('\rIter %d, best metric: %g' % (i, sign * opt.y.min()), flush = True, end="")
        if done:
            break
            
    best_id   = np.argmin(opt.y.reshape(-1))
    best_hyp  = opt.X.iloc[best_id]
    df_report = opt.X.copy()
    df_report['metric'] = sign * opt.y

    return best_hyp.to_dict(),scoresz[best_id]

def core_para(it, hyp, env = None):
    print("it:{} hyp:{}".format(it,hyp), flush = True)
    
    filename = os.path.join(base_path, '{}/Reacher'.format('Reacher_Windows'))
    
    if env is None:
        env = UnityEnvironment(file_name=filename, seed=0,no_graphics=False, worker_id=0+it)
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]

    config = initConf_ddpg(state_size,action_size)
    
    for k in config.update_fields:
        if space.paras[k].is_numeric and space.paras[k].is_discrete:
            setattr(config, k, int(hyp[k]))
        else:
            setattr(config, k, hyp[k])

    a = config.agent_fn(config)
    scores,reward = brain(brain_name,a,config,env,it=it)
    env.close()
    return (scores,reward)

def tune_para(
    space_cfg, 
    max_iter=10, 
    hebo_cfg = None,
    greater_is_better : bool = True,
    verbose  = True,
    **kwargs
    ):
    
    sign = -1. if greater_is_better else 1.0

    if hebo_cfg is None:
        hebo_cfg = {}
    space = DesignSpace().parse(space_cfg)
    opt   = HEBO(space, rand_sample = 1, **hebo_cfg)
    
#     envs = []
#     print("UnityEnvironment Start".format(), flush = True)
    
#     f = 'Reacher_Windows'
#     no_graphics = False
#     for i in range(1,3):
#         print("UnityEnvironment i:{}".format(i), flush = True)
#         done = False
#         failed = 0
#         while not done:
#             try:
#                 envs.append(UnityEnvironment(file_name='{}/Reacher'.format(f), seed=1,no_graphics=no_graphics, worker_id=0+i))
#                 done = True
#                 print("UnityEnvironment Done i:{}".format(i), flush = True)
#                 sleep(5)
#             except Exception as e:            
#                 print("it:{} UnityEnvironment failed:{}".format(i,e), flush = True)
#                 failed += 1
#                 if failed>5:
#                     done = True
#             return            
#     print("it:{} UnityEnvironment len envs:{}".format(i,len(envs)), flush = True)
    
    scoresz = []
    
    try:
    
        for i in range(max_iter):
            rec     = opt.suggest(n_suggestions=2)
    #         hyp     = rec.iloc[0].to_dict()
    #         verbose=100, pre_dispatch='1.5*n_jobs'
            outs = Parallel(n_jobs=2,verbose=100, prefer="processes")(delayed(core_para)(i,rc.to_dict()) for i,rc in enumerate(rec.iloc))
#         envs[i],

            outs
            if verbose:
                print('\routs:{}'.format(outs), flush = True)

            a = []
            for s,r in outs:
                a.append(sign * r)

            opt.observe(rec, np.array(a))
    #         opt.observe(rec, np.array([sign * reward]))
            if verbose:
                print('\nIter %d, best metric: %g' % (i, sign * opt.y.min()), flush = True, end="")
    except Exception as e:                
#         for env in envs:
#             env.close()
#         envs = []
        raise e
#     else:
#         for env in envs:
#             env.close()
#         envs = []
    
    best_id   = np.argmin(opt.y.reshape(-1))
    best_hyp  = opt.X.iloc[best_id]
    df_report = opt.X.copy()
    df_report['metric'] = sign * opt.y
#     if report:
#         return best_hyp.to_dict(), df_report
    return best_hyp.to_dict(),scoresz[best_id]