import os
import random
import csv
import argparse
import numpy as np
import time
from tqdm import trange
import torch
import torch.nn as nn

from env import GridWorldEnv
from agent_dqn import DQNAgent
# plotting helper (reads CSV and produces plots)
from plot_training import read_log, plot as plot_from_log
from collections import deque
from agent_ppo import PPOAgent
from agent_A2C import A2CAgent, Transition

def is_map_connected(obstacles, width, height):
    """Check if empty cells form a single connected component (4-neighbor).

    obstacles: iterable of (y,x)
    """
    grid = [[0] * width for _ in range(height)]
    for (y, x) in obstacles:
        grid[y][x] = 1

    # find a starting empty cell
    start = None
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 0:
                start = (y, x)
                break
        if start is not None:
            break
    if start is None:
        return False

    q = deque([start])
    seen = {start}
    while q:
        y, x = q.popleft()
        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
            ny, nx = y+dy, x+dx
            if 0 <= ny < height and 0 <= nx < width and grid[ny][nx] == 0 and (ny,nx) not in seen:
                seen.add((ny,nx))
                q.append((ny,nx))

    # count empty cells
    empty_count = sum(1 for y in range(height) for x in range(width) if grid[y][x] == 0)
    return len(seen) == empty_count


def generate_map_pool(n_maps, width=6, height=6, num_obstacles=5, seed=None, max_tries=100):
    """Generate a list of obstacle lists where each map's empty cells are connected.

    Returns list of lists of (y,x) obstacle coordinates.
    """
    rng = random.Random(seed)
    all_cells = [(y,x) for y in range(height) for x in range(width)]
    pools = []
    tries = 0
    while len(pools) < n_maps and tries < max_tries * n_maps:
        tries += 1
        k = max(0, min(len(all_cells)-2, int(num_obstacles)))
        obs = rng.sample(all_cells, k=k) if k>0 else []
        if is_map_connected(obs, width, height):
            pools.append(obs)
    # if we failed to generate enough connected maps, pad with fixed layout
    if len(pools) < n_maps:
        fixed = [(1,2),(2,2),(3,2),(4,4),(1,4)]
        while len(pools) < n_maps:
            pools.append(fixed)
    return pools


def train(num_episodes=500, save_path="grid_dqn.pth", distance_coef=0.0, log_csv=None,
        randomize_obstacles=True, num_obstacles=5, n_train_maps=20, n_val_maps=5, map_seed=None):
    # create a single env and swap obstacle layouts per episode from a pool
    env = GridWorldEnv(width=6, height=6, max_steps=100)

    # generate training/validation map pools
    train_maps = generate_map_pool(n_train_maps, width=env.width, height=env.height, num_obstacles=num_obstacles, seed=map_seed)
    # validation maps should be fixed and reproducible: use a different seed if provided
    val_seed = None if map_seed is None else map_seed + 1
    val_maps = generate_map_pool(n_val_maps, width=env.width, height=env.height, num_obstacles=num_obstacles, seed=val_seed)
    state_dim = 4 + env.width * env.height
    # agent = DQNAgent(state_dim=state_dim, distance_coef=distance_coef)
    # 初始化 agent
    if args.algo == "dqn":
        agent = DQNAgent(state_dim=state_dim, distance_coef=args.distance_coef)
    elif args.algo == "ppo":
        agent = PPOAgent(state_dim=state_dim)
    else:
        agent = A2CAgent(state_dim=state_dim)
    epsilon_start = 1.0
    epsilon_final = 0.05
    epsilon_decay = 5000

    batch_size = 64
    total_steps = 0

    best_mean = -1e9
    rewards_history = []

    # prepare CSV logging if requested
    if log_csv:
        os.makedirs(os.path.dirname(log_csv), exist_ok=True)
        csv_file = open(log_csv, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        # add val_reward and val_loss columns (may be empty for many rows)
        # also record success (1/0), path_length (steps), and motion_time (seconds)
        csv_writer.writerow(['episode', 'ep_reward', 'mean_distance', 'mean_loss', 'epsilon', 'val_reward', 'val_loss', 'success', 'path_length', 'motion_time'])
    else:
        csv_file = None
        csv_writer = None

    # Validation settings will be passed as attributes if set on the train function (CLI will set them)
    eval_interval = getattr(train, 'eval_interval', None)
    eval_episodes = getattr(train, 'eval_episodes', None)

    for ep in trange(num_episodes, desc="Episodes"):
        # pick a training map for this episode
        if randomize_obstacles:
            map_idx = random.randrange(len(train_maps))
            env.set_obstacles(train_maps[map_idx])
        else:
            obs = [(1, 2), (2, 2), (3, 2), (4, 4), (1, 4)]
            env.set_obstacles(obs)
        state = env.reset()
        ep_reward = 0.0
        done = False
        ep_losses = []
        traj = []
        sum_dist = 0.0
        steps = 0
        start_time = time.time()
        while not done:
            eps = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1.0 * total_steps / epsilon_decay)
            
            if isinstance(agent, DQNAgent):
                action = agent.select_action(state, epsilon=eps)
            else:
                action,logp,v = agent.select_action(state)
            
            next_state, reward, done, info = env.step(action)
            
            if isinstance(agent, DQNAgent):
                agent.push_transition(state, action, reward, next_state, done)
                loss = agent.update(batch_size=batch_size)
            else:
                traj.append(Transition(torch.FloatTensor(state), action, reward, torch.FloatTensor(next_state), done, logp))
                loss = agent.update(traj)

            if loss is not None:
                ep_losses.append(loss)
            # compute distance in normalized coords: indices 0,1 agent; 2,3 goal
            try:
                d = float(np.linalg.norm(next_state[2:4] - next_state[0:2]))
            except Exception:
                d = 0.0
            sum_dist += d
            steps += 1

            state = next_state
            ep_reward += reward
            total_steps += 1

        # episode metrics
        mean_dist = sum_dist / steps if steps > 0 else 0.0
        motion_time = time.time() - start_time
        path_length = steps
        success = int(env.agent_pos == env.goal_pos)
        mean_loss = float(np.mean(ep_losses)) if len(ep_losses) > 0 else 0.0

        rewards_history.append(ep_reward)

        # val_reward = ''
        # val_loss = ''
        if ep % 20 == 0:
            mean_reward = np.mean(rewards_history[-50:]) if len(rewards_history) > 0 else 0.0
            print(f"Episode {ep} reward {ep_reward:.1f} mean50 {mean_reward:.2f} eps {eps:.3f} mean_dist {mean_dist:.3f} mean_loss {mean_loss:.4f}")
            if mean_reward > best_mean:
                best_mean = mean_reward
                agent.save(save_path)

        # periodic evaluation on a validation set (greedy)
        if eval_interval and eval_episodes and (ep % eval_interval == 0):
            # run greedy evaluation episodes across validation maps (no exploration)
            val_rewards = []
            val_losses = []
            for vm in val_maps:
                val_env = GridWorldEnv(width=env.width, height=env.height, max_steps=env.max_steps)
                val_env.set_obstacles(vm)
                for _ in range(eval_episodes):
                    s = val_env.reset()
                done_v = False
                rsum = 0.0
                # accumulate per-step losses for validation
                while not done_v:
                    if isinstance(agent, DQNAgent):
                        a = agent.select_action(s, epsilon=0.0)
                    else:
                        a, _, _ = agent.select_action(s)
                    next_s, r, done_v, _ = val_env.step(a)
                    rsum += r
                    # compute per-transition loss (without updating)
                    try:
                        s_t = torch.FloatTensor(s).unsqueeze(0).to(agent.device)
                        ns_t = torch.FloatTensor(next_s).unsqueeze(0).to(agent.device)
                        a_t = torch.LongTensor([a]).unsqueeze(1).to(agent.device)
                        r_t = torch.FloatTensor([r]).unsqueeze(1).to(agent.device)
                        done_t = torch.FloatTensor([float(done_v)]).unsqueeze(1).to(agent.device)

                        with torch.no_grad():
                            q_val = agent.q_net(s_t).gather(1, a_t)
                            next_a = agent.q_net(ns_t).argmax(dim=1, keepdim=True)
                            next_q = agent.target_net(ns_t).gather(1, next_a)

                            # potential-based shaping if enabled
                            if agent.distance_coef != 0.0:
                                agent_pos = s_t[:, 0:2]
                                goal_pos = s_t[:, 2:4]
                                dist_s = torch.norm(goal_pos - agent_pos, dim=1, keepdim=True)
                                agent_pos_n = ns_t[:, 0:2]
                                goal_pos_n = ns_t[:, 2:4]
                                dist_s_n = torch.norm(goal_pos_n - agent_pos_n, dim=1, keepdim=True)
                                phi_s = - agent.distance_coef * dist_s
                                phi_s_n = - agent.distance_coef * dist_s_n
                                shaped_r = r_t + (agent.gamma * phi_s_n - phi_s)
                            else:
                                shaped_r = r_t
                            if isinstance(agent, DQNAgent):
                                target_v = shaped_r + (1.0 - done_t) * agent.gamma * next_q
                                loss_v = nn.SmoothL1Loss()(q_val, target_v)
                            else:
                                # for PPO/A2C, we can compute value loss only
                                _, value_v = agent.net(s_t)
                                value_v = value_v.squeeze(1)
                                loss_v = nn.SmoothL1Loss()(value_v, shaped_r + (1.0 - done_t) * agent.gamma * next_q)
                            val_losses.append(float(loss_v.item()))
                            
                    except Exception:
                        pass

                    s = next_s
                val_rewards.append(rsum)

            val_reward = float(np.mean(val_rewards)) if len(val_rewards) > 0 else float('nan')
            val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else float('nan')
            print(f"  [Eval] mean reward over {eval_episodes}: {val_reward:.2f} mean val_loss {val_loss:.4f}")

        if csv_writer:
            csv_writer.writerow([ep, ep_reward, mean_dist, mean_loss, eps, val_reward, val_loss, success, path_length, motion_time])

    # final save
    agent.save(save_path)
    if csv_file:
        csv_file.close()

    print("Training finished. Model saved to", save_path)

    # Auto-generate training plots if a CSV log was produced
    if log_csv and os.path.isfile(log_csv):
        try:
            plots_dir = os.path.join(os.path.dirname(log_csv))
            data = read_log(log_csv)
            plot_from_log(data, out_dir=plots_dir)
            print(f"Saved training plots to {plots_dir}")
        except Exception as e:
            print("Warning: failed to generate plots:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",choices=["dqn","ppo","a2c"],default="ppo")
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--save-path', type=str, default=os.path.join(os.path.dirname(__file__), f'saving_model/grid_{parser.parse_args().algo}.pth'))
    parser.add_argument('--distance-coef', type=float, default=1, help='Phi coefficient (alpha) for potential-based shaping')
    parser.add_argument('--log-csv', type=str, default=os.path.join(os.path.dirname(__file__), f'train_{parser.parse_args().algo}_results_usedis/train_log.csv'), help='CSV path to save training logs')
    parser.add_argument('--eval-interval', type=int, default=20, help='Run evaluation every N episodes (0 = disabled)')
    parser.add_argument('--eval-episodes', type=int, default=5, help='Number of episodes to run for each evaluation')
    # obstacle randomization options
    parser.add_argument('--num-obstacles', type=int, default=5, help='Number of obstacles to place when randomizing')
    parser.add_argument('--no-randomize-obstacles', dest='randomize_obstacles', action='store_false', help='Disable random obstacle generation (use fixed layout)')
    parser.set_defaults(randomize_obstacles=False)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    # attach eval settings onto the train function so internals can read them
    if args.eval_interval > 0:
        train.eval_interval = args.eval_interval
        train.eval_episodes = args.eval_episodes
    else:
        train.eval_interval = None
        train.eval_episodes = None


    # pass obstacle randomization options into training
    train(num_episodes=args.episodes, save_path=args.save_path, distance_coef=args.distance_coef, log_csv=args.log_csv,
          randomize_obstacles=args.randomize_obstacles, num_obstacles=args.num_obstacles)
