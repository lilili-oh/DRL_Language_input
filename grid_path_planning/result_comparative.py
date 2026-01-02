import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
def read_csv(file_path):
    return pd.read_csv(file_path)

# 绘制并保存每个指标的图
def plot_and_save_comparison(data1, data2, data3, title, metric, file_name):
    plt.figure(figsize=(10, 6))
    
    # 绘制三个算法的曲线
    plt.plot(data1['episode'], data1[metric], label='A2C', color='r',alpha=0.7)
    plt.plot(data2['episode'], data2[metric], label='DQN', color='g')
    plt.plot(data3['episode'], data3[metric], label='PPO', color='b',alpha=0.5)
    
    # 添加标题和标签
    plt.xlabel('Episode')
    plt.ylabel(metric)
    plt.title(f'{title} - {metric}')
    plt.legend()
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

# 文件路径
file1 = '/Users/yuemingli/Desktop/workspace/language_RL/grid_path_planning/train_a2c_results_usedis/train_log.csv'
file2 = '/Users/yuemingli/Desktop/workspace/language_RL/grid_path_planning/train_dqn_results_usedis/train_log.csv'
file3 = '/Users/yuemingli/Desktop/workspace/language_RL/grid_path_planning/train_ppo_results_usedis/train_log.csv'

# 读取数据
data1 = read_csv(file1)
data2 = read_csv(file2)
data3 = read_csv(file3)

# 绘制并保存每个指标的图
plot_and_save_comparison(data1, data2, data3, 'Comparison of Algorithms', 'mean_loss', 'mean_loss_comparison.png')
plot_and_save_comparison(data1, data2, data3, 'Comparison of Algorithms', 'ep_reward', 'reward_comparison.png')
plot_and_save_comparison(data1, data2, data3, 'Comparison of Algorithms', 'epsilon', 'epsilon_comparison.png')
plot_and_save_comparison(data1, data2, data3, 'Comparison of Algorithms', 'motion_time', 'motion_time_comparison.png')

print("Comparison graphs saved successfully.")
