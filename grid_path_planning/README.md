
文件说明：

- `env.py`：自定义 `GridWorldEnv` 环境（可选障碍），返回状态为归一化坐标加上障碍二值图。
- `agent_dqn.py`：基于 PyTorch 的 DQN 智能体与回放缓冲实现。
- `agent_A2C.py & agent_ppo.py` 两种不同的训练模型
- `NLP.py` 进行自然语言处理作为模型的输入
- `visual.py` 模型测试脚本
- `result_comparative.py` 训练结果可视化分析
- `train.py`：训练脚本，包含训练循环、评估与模型保存。
- `requirements.txt`：运行示例所需的 Python 包清单。

快速运行（macOS zsh）：

```bash
# 建议在虚拟环境中运行
python3 -m pip install -r ./grid_path_planning/requirements.txt
python3 ./grid_path_planning/train.py
```
