# DQN Grid Path Planning 示例

这是一个最小可运行的深度强化学习（DQN）示例，用于在格子地图上做路径规划。

文件说明：

- `env.py`：自定义 `GridWorldEnv` 环境（可选障碍），返回状态为归一化坐标加上障碍二值图。
- `agent_dqn.py`：基于 PyTorch 的 DQN 智能体与回放缓冲实现。
- `train.py`：训练脚本，包含训练循环、评估与模型保存。
- `requirements.txt`：运行示例所需的 Python 包清单。

快速运行（macOS zsh）：

```bash
# 建议在虚拟环境中运行
python3 -m pip install -r /Users/yuemingli/Desktop/workspace/language_RL/grid_path_planning/requirements.txt
python3 /Users/yuemingli/Desktop/workspace/language_RL/grid_path_planning/train.py
```
