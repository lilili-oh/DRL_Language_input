import os
import argparse
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

from env import GridWorldEnv
from agent_dqn import DQNAgent
from agent_ppo import PPOAgent
from agent_A2C import A2CAgent
from NLP import NLP_input

def plot_episode(env, trajectory, save_path=None, title=None):
    h, w = env.height, env.width
    obs = env.obstacles

    fig, ax = plt.subplots(figsize=(w, h))
    # draw grid
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, w, 1))
    ax.set_yticks(np.arange(-0.5, h, 1))
    ax.grid(color='gray')

    # draw obstacles
    for y in range(h):
        for x in range(w):
            if obs[y, x] == 1:
                rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='black')
                ax.add_patch(rect)

    # start and goal - use the initial recorded trajectory position for start if available
    if trajectory and len(trajectory) > 0:
        sy, sx = trajectory[0]
    else:
        sy, sx = env.agent_pos
    gy, gx = env.goal_pos
    ax.plot(sx, sy, marker='o', color='green', markersize=12, label='start')
    ax.plot(gx, gy, marker='*', color='red', markersize=14, label='goal')

    # trajectory: list of (y,x)
    ys = [p[0] for p in trajectory]
    xs = [p[1] for p in trajectory]
    ax.plot(xs, ys, marker='.', color='blue', linewidth=2, markersize=6, label='path')

    ax.set_title(title or 'Episode')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def visualize(model_path, episodes=5, out_dir=None, seed=0, dynamic=False, delay=0.2, save_gif=False, algo="dqn", start_pos=(0,0), goal_history=[]):
    np.random.seed(seed)
    env = GridWorldEnv(width=6, height=6, max_steps=200)
    obs = [(1, 2), (2, 2), (3, 2), (4, 4), (1, 4)]
    env.set_obstacles(obs)

    state_dim = 4 + env.width * env.height
    if algo == "dqn":
        agent = DQNAgent(state_dim=state_dim)
    elif algo == "a2c":
        agent = A2CAgent(state_dim=state_dim)
    elif algo == "ppo":
        agent = PPOAgent(state_dim=state_dim)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    agent.load(model_path)

    results = []
    for ep in range(episodes):
        # reset and record the positions
        if ep == 0 and start_pos and goal_history:
            if goal_history[0] in obs:
                raise ValueError("The first goal position cannot be on an obstacle.")
            state = env.reset(start=start_pos, goal=goal_history[0])
        else:
            if goal_history[ep % len(goal_history)] in obs:
                raise ValueError(f"Goal position {goal_history[ep % len(goal_history)]} cannot be on an obstacle.")
            state = env.reset(start=goal_history[ep % len(goal_history)-1], goal=goal_history[ep % len(goal_history)])
        # print(f"Episode {ep}: Start: {start_pos}, Goal: {goal_history[ep % len(goal_history)]}")
        traj = [env.agent_pos]
        done = False
        ep_reward = 0.0
        steps = 0
        start_time = time.time()

        if dynamic:
            # prepare interactive plot for this episode
            h, w = env.height, env.width
            fig, ax = plt.subplots(figsize=(w, h))
            ax.set_xlim(-0.5, w - 0.5)
            ax.set_ylim(h - 0.5, -0.5)
            ax.set_xticks(np.arange(-0.5, w, 1))
            ax.set_yticks(np.arange(-0.5, h, 1))
            ax.grid(color='gray')
            for y in range(h):
                for x in range(w):
                    if env.obstacles[y, x] == 1:
                        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='black')
                        ax.add_patch(rect)

            # start from the initial recorded position (traj[0]) so the plotted start
            # does not get overwritten by env.agent_pos changing during the episode
            sy, sx = traj[0]
            gy, gx = env.goal_pos
            ax.plot(sx, sy, marker='o', color='green', markersize=12, label='start')
            ax.plot(gx, gy, marker='*', color='red', markersize=14, label='goal')

            ys = [p[0] for p in traj]
            xs = [p[1] for p in traj]
            path_line, = ax.plot(xs, ys, marker='.', color='blue', linewidth=2, markersize=6, label='path')
            current_scatter, = ax.plot([xs[-1]], [ys[-1]], marker='o', color='cyan', markersize=8)
            ax.set_title(f"Episode {ep}")
            ax.set_aspect('equal')
            ax.legend(loc='upper right')
            plt.pause(0.001)

        while not done:
            if algo == "dqn":
                action = agent.select_action(state, epsilon=0.0)
            else:
                action, _, _ = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            traj.append(env.agent_pos)
            state = next_state
            ep_reward += reward
            steps += 1

            if dynamic:
                ys = [p[0] for p in traj]
                xs = [p[1] for p in traj]
                path_line.set_data(xs, ys)
                current_scatter.set_data([xs[-1]], [ys[-1]])
                ax.set_title(f"Episode {ep} reward={ep_reward:.1f} steps={steps}")
                plt.draw()
                plt.pause(delay)

            if steps > env.max_steps:
                break

        if dynamic:
            # small pause so final frame is visible
            plt.pause(0.5)
            plt.close()

        results.append({'traj': traj, 'reward': ep_reward, 'steps': steps, 'success': traj[-1] == env.goal_pos})

        # record motion time and path length in results
        motion_time = time.time() - start_time
        results[-1]['motion_time'] = motion_time
        results[-1]['path_length'] = steps

        title = f"Episode {ep} reward={ep_reward:.1f} steps={steps} success={results[-1]['success']}"

        # Priority: save GIF if requested; otherwise save PNG to out_dir when provided; otherwise show plot
        if save_gif:
            if out_dir is None:
                raise ValueError('To save GIFs you must specify --out-dir')
            gif_path = os.path.join(out_dir, f'episode_{ep:03d}.gif')
            # build animation from trajectory
            h, w = env.height, env.width
            fig, ax = plt.subplots(figsize=(w, h))
            ax.set_xlim(-0.5, w - 0.5)
            ax.set_ylim(h - 0.5, -0.5)
            ax.set_xticks(np.arange(-0.5, w, 1))
            ax.set_yticks(np.arange(-0.5, h, 1))
            ax.grid(color='gray')
            for y in range(h):
                for x in range(w):
                    if env.obstacles[y, x] == 1:
                        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='black')
                        ax.add_patch(rect)

            # Use the recorded initial position for start when building GIFs/static plots
            sy, sx = traj[0]
            gy, gx = env.goal_pos
            ax.plot(sx, sy, marker='o', color='green', markersize=12, label='start')
            ax.plot(gx, gy, marker='*', color='red', markersize=14, label='goal')

            ys_all = [p[0] for p in traj]
            xs_all = [p[1] for p in traj]
            path_line, = ax.plot([], [], marker='.', color='blue', linewidth=2, markersize=6, label='path')
            current_scatter, = ax.plot([], [], marker='o', color='cyan', markersize=8)
            ax.set_aspect('equal')
            ax.legend(loc='upper right')

            def update(i):
                xs = xs_all[: i + 1]
                ys = ys_all[: i + 1]
                path_line.set_data(xs, ys)
                current_scatter.set_data([xs[-1]], [ys[-1]])
                return path_line, current_scatter

            anim = animation.FuncAnimation(fig, update, frames=len(traj), interval=max(1, int(delay * 1000)), blit=True)
            try:
                os.makedirs(os.path.dirname(gif_path), exist_ok=True)
                writer = PillowWriter(fps=max(1, int(1.0 / delay)))
                anim.save(gif_path, writer=writer)
            except Exception as e:
                print(f"Warning: failed to save GIF {gif_path}: {e}")
            finally:
                plt.close(fig)
        elif out_dir and not dynamic:
            save_path = os.path.join(out_dir, f'episode_{ep:03d}.png')
            plot_episode(env, traj, save_path=save_path, title=title)
        elif not out_dir and not dynamic:
            plot_episode(env, traj, save_path=None, title=title)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize DQN agent trajectories')
    parser.add_argument('--algo', type=str, default="a2c", help='Algorithm to visualize: dqn, a2c, ppo')
    parser.add_argument('--model', type=str, default=os.path.join(os.path.dirname(__file__), f'saving_model/grid_{parser.parse_args().algo}.pth'))
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--out-dir', type=str, default=f"/Users/yuemingli/Desktop/workspace/language_RL/grid_path_planning/{parser.parse_args().algo}_results", help='If set, saves PNGs into this directory')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true', help='If set, shows dynamic plotting of the agent moving')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay between frames in dynamic mode (seconds)')
    parser.add_argument('--save-gif', dest='save_gif', action='store_true', help='If set, save each episode as a GIF into --out-dir')
    args = parser.parse_args()
    input_text = argparse.ArgumentParser(description='NLP Command Input')
    input_text.add_argument('--command_test', type=str, default="第一向上走三步第二向右走一步第三向后走三步开始执行", help='NLP command input for path planning')
    args_nlp = input_text.parse_args()

    args.save_gif = f"/Users/yuemingli/Desktop/workspace/language_RL/grid_path_planning/{parser.parse_args().algo}_results"
    start_pos = (0, 0)
    goal_history = NLP_input(command_test=args_nlp.command_test, init_pos=start_pos)
    args.episodes = len(goal_history)
    res = visualize(model_path=args.model, episodes=args.episodes, out_dir=args.out_dir, seed=args.seed, dynamic=args.dynamic, delay=args.delay, save_gif=args.save_gif, algo=args.algo,start_pos=start_pos,goal_history=goal_history)
    # print a short summary
    for i, r in enumerate(res):
        print(f"Episode {i}: reward={r['reward']:.1f} steps={r['steps']} success={r['success']}")

    # aggregate metrics
    try:
        successes = [float(r.get('success', 0)) for r in res]
        path_lengths = [float(r.get('path_length', float('nan'))) for r in res]
        motion_times = [float(r.get('motion_time', float('nan'))) for r in res]
        success_rate = float(np.nansum(successes)) / len(successes) if len(successes) > 0 else float('nan')
        mean_path = float(np.nanmean(path_lengths)) if len(path_lengths) > 0 else float('nan')
        mean_motion = float(np.nanmean(motion_times)) if len(motion_times) > 0 else float('nan')
        print(f"Summary over {len(res)} episodes: success_rate={success_rate:.3f} mean_path_length={mean_path:.2f} mean_motion_time={mean_motion:.3f}s")
    except Exception:
        pass
    out_dir = args.out_dir
    # if out_dir provided, save metrics CSV and produce simple plots here
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, 'visualize_metrics.csv')
            # write only the columns we actually have to avoid parse errors
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['episode', 'ep_reward', 'success', 'path_length', 'motion_time'])
                for i, r in enumerate(res):
                    writer.writerow([i,
                                     float(r.get('reward', float('nan'))),
                                     int(bool(r.get('success', False))),
                                     int(r.get('path_length', 0)),
                                     float(r.get('motion_time', 0.0))])

            print(f"Saved visualization metrics to {csv_path}")
        except Exception as e:
            print(f"Warning: failed to save visualization metrics CSV: {e}")