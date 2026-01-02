import os
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt


def read_log(path):
    data = {'episode': [], 'ep_reward': [], 'mean_distance': [], 'mean_loss': [], 'epsilon': [], 'val_reward': [], 'val_loss': [], 'success': [], 'path_length': [], 'motion_time': []}

    def _parse_float(val, default=np.nan):
        # treat None or empty string as default (usually np.nan)
        if val is None:
            return default
        if isinstance(val, str):
            v = val.strip()
            if v == '':
                return default
        try:
            return float(val)
        except Exception:
            return default

    def _parse_int(val, default=0):
        if val is None:
            return default
        if isinstance(val, str):
            v = val.strip()
            if v == '':
                return default
        try:
            return int(float(val))
        except Exception:
            return default

    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['episode'].append(_parse_int(row.get('episode', 0), default=0))
            data['ep_reward'].append(_parse_float(row.get('ep_reward', np.nan)))
            data['mean_distance'].append(_parse_float(row.get('mean_distance', np.nan)))
            data['mean_loss'].append(_parse_float(row.get('mean_loss', np.nan)))
            data['epsilon'].append(_parse_float(row.get('epsilon', np.nan)))

            # val_reward / val_loss may be empty
            data['val_reward'].append(_parse_float(row.get('val_reward', np.nan)))
            data['val_loss'].append(_parse_float(row.get('val_loss', np.nan)))

            # optional columns: success, path_length, motion_time
            data['success'].append(_parse_float(row.get('success', np.nan)))
            data['path_length'].append(_parse_float(row.get('path_length', np.nan)))
            data['motion_time'].append(_parse_float(row.get('motion_time', np.nan)))

    return data


def plot(data, out_dir=None):
    eps = np.array(data['episode'])
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    
    # 1. Train reward vs Val reward
    if 'ep_reward' in data and 'val_reward' in data:
        axs[0].plot(eps, data['ep_reward'], label='Train Reward')
        # axs[0].plot(eps, data['val_reward'], label='Val Reward')
        axs[0].set_ylabel('Reward')
        axs[0].legend()
        axs[0].grid(True)
    else:
        print("Warning: ep_reward or val_reward missing")
    
    # 2. Train loss vs Val loss
    if 'mean_loss' in data and 'val_loss' in data:
        axs[1].plot(eps, data['mean_loss'], label='Train Loss')
        # axs[1].plot(eps, data['val_loss'], label='Val Loss')
        axs[1].set_ylabel('Loss')
        axs[1].legend()
        axs[1].grid(True)
    else:
        print("Warning: mean_loss or val_loss missing")
    
    # 3. Mean distance
    if 'mean_distance' in data:
        axs[2].plot(eps, data['mean_distance'], label='Mean Distance')
        axs[2].set_ylabel('Distance')
        axs[2].legend()
        axs[2].grid(True)
    
    # 4. Epsilon
    if 'epsilon' in data:
        axs[3].plot(eps, data['epsilon'], label='Epsilon')
        axs[3].set_ylabel('Epsilon')
        axs[3].set_xlabel('Episode')
        axs[3].legend()
        axs[3].grid(True)
    
    plt.tight_layout()
    
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'training_metrics_plot.png')
        fig.savefig(out_path)
        print('Saved plot to', out_path)
    else:
        plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-csv', type=str, required=True, help='Path to training CSV log')
    parser.add_argument('--out-dir', type=str, default=None, help='If set, save plot to directory')
    args = parser.parse_args()

    data = read_log(args.log_csv)
    plot(data, out_dir=args.out_dir)
