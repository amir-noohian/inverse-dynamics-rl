import torch
import argparse
import relod.utils as utils
import time
import numpy as np
import cv2
import os

from relod.logger import Logger
from relod.algo.comm import MODE
from relod.algo.local_wrapper import LocalWrapper
from relod.algo.sac_rad_agent import SACRADLearner, SACRADPerformer
from relod.envs.visual_franka_reacher.franka_env import FrankaPanda_Visual_Reacher

config = {
    
    'conv': [
        # in_channel, out_channel, kernel_size, stride
        [-1, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 1],
    ],
    
    'latent': 50,

    'mlp': [
        [-1, 1024], # first hidden layer
        [1024, 1024], 
        [1024, -1] # output layer
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Local Franka Visual Min Task Eval')
    # environment
    parser.add_argument('--setup', default='Visual-Franka')
    parser.add_argument('--env', default='Visual_Franka_Dense_Task', type=str) 
    parser.add_argument('--camera_id', default=0, type=int)
    parser.add_argument('--image_width', default=160, type=int)
    parser.add_argument('--image_height', default=90, type=int)
    parser.add_argument('--target_type', default='size', type=str)
    parser.add_argument('--image_history', default=3, type=int)
    parser.add_argument('--episode_length_time', default=20, type=float)
    parser.add_argument('--dt', default=0.04, type=float)

    parser.add_argument('--size_tol', default=0.1, type=float)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=1000, type=int)
    parser.add_argument('--rad_offset', default=0.01, type=float)
    # train
    parser.add_argument('--init_steps', default=500000, type=int) 
    parser.add_argument('--env_steps', default=500000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--async_mode', default=True, action='store_true')
    parser.add_argument('--async_buffer', default=True, action='store_true')
    parser.add_argument('--max_updates_per_step', default=1.0, type=float)
    parser.add_argument('--update_every', default=50, type=int)
    parser.add_argument('--update_epochs', default=50, type=int)
    # critic
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    parser.add_argument('--bootstrap_terminal', default=0, type=int)
    # actor
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    # encoder
    parser.add_argument('--encoder_tau', default=0.005, type=float)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=3e-4, type=float)
    # agent
    parser.add_argument('--remote_ip', default='localhost', type=str)
    parser.add_argument('--port', default=9876, type=int)
    parser.add_argument('--mode', default='e', type=str, help="Modes in ['r', 'l', 'rl', 'e'] ")
    # misc
    parser.add_argument('--description', default='test new remote script', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--plot_learning_curve', default=True, action='store_true')
    parser.add_argument('--xtick', default=2500, type=int)
    parser.add_argument('--display_image', default=False, action='store_true')
    parser.add_argument('--save_image', default=False, action='store_true')
    parser.add_argument('--save_model_freq', default=10000, type=int)
    parser.add_argument('--load_model', default=100000, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--lock', default=False, action='store_true')

    args = parser.parse_args()
    assert args.mode in ['r', 'l', 'rl', 'e']
    return args


def check_end_run():
    fl = open("run_end.log", "r")
    en = int(fl.read())
    fl.close()
    return en


def main(seed=-1):
    args = parse_args()

    if seed != -1:
        args.seed = seed

    if args.mode == 'r':
        mode = MODE.REMOTE_ONLY
    elif args.mode == 'l':
        mode = MODE.LOCAL_ONLY
    elif args.mode == 'rl':
        mode = MODE.REMOTE_LOCAL
    elif args.mode == 'e':
        mode = MODE.EVALUATION
    else:
        raise  NotImplementedError()

    assert args.async_mode == True

    if args.device is '':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    args.work_dir += f'/results/{args.env}/seed={args.seed}'
    args.model_dir = args.work_dir+'/models'
    args.return_dir = args.work_dir+'/returns'

    # os.makedirs(args.model_dir, exist_ok=False)
    # os.makedirs(args.return_dir, exist_ok=False)

    if mode == MODE.LOCAL_ONLY:
        L = Logger(args.return_dir, use_tb=args.save_tb)

    episode_length_step = int(args.episode_length_time / args.dt)

    env = FrankaPanda_Visual_Reacher(
        dt=args.dt,
        image_history_size=args.image_history,
        image_width=args.image_width,
        image_height=args.image_height,
        episode_steps=episode_length_step,
        camera_index=args.camera_id,
        seed=args.seed,
        experiment_type='eval', 
        size_tol=args.size_tol,
        print_target_info=True)

    utils.set_seed_everywhere(args.seed, env)

    image, prop = env.reset()
    
    args.image_shape = env.image_space.shape
    args.proprioception_shape = env.observation_space.shape

    print(args.image_shape, image.shape)
    print(args.proprioception_shape, prop.shape)

    args.action_shape = env.action_space.shape
    args.env_action_space = env.action_space
    args.net_params = config

    
    agent = LocalWrapper(episode_length_step, mode, remote_ip=args.remote_ip, port=args.port)
    agent.send_data(args)
    agent.init_performer(SACRADPerformer, args)
    agent.init_learner(SACRADLearner, args, agent.performer)

    # sync initial weights with remote
    agent.apply_remote_policy(block=True)

    if args.load_model > -1:
        agent.load_policy_from_file(args.model_dir, args.load_model)
    
    # First inference took a while (~1 min), do it before the agent-env interaction loop
    if mode != MODE.REMOTE_ONLY:
        agent.performer.sample_action((image, prop))
        agent.performer.sample_action((image, prop))
        agent.performer.sample_action((image, prop))

    rewards_fl = open(args.return_dir + '/rewards.txt', 'w')

    # Experiment block starts
    experiment_done = False
    total_steps = 0
    returns = []
    epi_lens = []
    start_time = time.time()
    print(f'Experiment starts at: {start_time}')

    while not experiment_done:
        if check_end_run() == 1:
            env.reset()
            agent.close()
            env.close()
            return -1
        image, prop = env.reset() 
        agent.send_init_ob((image, prop))
        ret = 0
        epi_steps = 0
        epi_done = 0
        rewards = []
        epi_start_time = time.time()

        while not experiment_done and not epi_done:
            # select an action
            action = agent.sample_action((image, prop))

            # step in the environment
            next_image, next_prop, reward, epi_done, _ = env.step(action)

            image = next_image
            prop = next_prop

            # Log
            total_steps += 1
            ret += reward
            epi_steps += 1
            rewards.append(reward)

            experiment_done = total_steps >= args.env_steps

        if epi_done: # episode done, save result
            returns.append(ret)
            epi_lens.append(epi_steps)
            rewards_fl.write(str(rewards)+'\n')
            rewards_fl.flush()

            utils.save_returns(args.return_dir+'/return.txt', returns, epi_lens)

            if mode == MODE.LOCAL_ONLY:
                L.log('train/duration', time.time() - epi_start_time, total_steps)
                L.log('train/episode_reward', ret, total_steps)
                L.log('train/episode', len(returns), total_steps)
                L.dump(total_steps)
                if args.plot_learning_curve:
                    utils.show_learning_curve(args.return_dir+'/learning curve.png', returns, epi_lens, xtick=args.xtick)

    duration = time.time() - start_time

    rewards_fl.close()
    # Clean up
    env.reset()
    agent.close()
    env.close()

    # always show a learning curve at the end
    # if mode == MODE.LOCAL_ONLY:
    utils.show_learning_curve(args.return_dir+'/learning curve.png', returns, epi_lens, xtick=args.xtick)
    print(f"Finished in {duration}s")

    return 0

if __name__ == '__main__':
    for i in range(2):
        fl = main(i)
        if fl == -1:
            break
        time.sleep(60)

