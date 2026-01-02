import re
from timeit import main

class Position:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def move(self, direction, steps):
        """根据指令更新坐标"""
        directions = {
            "up": (0, 1),  # 向上，y + 步数
            "down": (0, -1), # 向下，y - 步数
            "left": (-1, 0),     # 向左，x - 步数
            "right": (1, 0)      # 向右，x + 步数
        }
        
        if direction not in directions:
            print(f"Invalid direction: {direction}")
            return
        
        dx, dy = directions[direction]
        self.x += dx * steps
        self.y += dy * steps

    def __str__(self):
        return f"Position: ({self.x}, {self.y})"

# 中文数字转阿拉伯数字
def chinese_to_number(chinese_num):
    chinese_digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    return chinese_digits.get(chinese_num, 0)

# 解析文本指令并执行相应动作
def parse_and_execute(command, pos):
    print(f"Processing command: {command}")
    
    # 修改后的正则表达式匹配“向上走三步”或“向右走三步”
    match = re.match(r"(向[上下左右后]+走)([一二三四五六七八九\d]+)步", command)
    if match:
        direction = match.group(1)  # 获取方向部分
        steps_chinese = match.group(2)  # 获取步数部分（中文数字）
        
        # 将中文数字转换为阿拉伯数字
        steps = chinese_to_number(steps_chinese) if not steps_chinese.isdigit() else int(steps_chinese)
        
        # 根据方向转换为对应的符号
        if "上" in direction:
            direction = "up"
        elif "下" in direction:
            direction = "down"
        elif "左" in direction:
            direction = "left"
        elif "右" in direction:
            direction = "right"
        elif "后" in direction:
            direction = "down"  # 假设“后”就是“向下”

        # 执行移动
        pos.move(direction, steps)
        print(f"Executed: {command} -> {pos}")
        goal = (pos.x, pos.y)
        return goal
    else:
        print(f"Invalid command: {command}")
def NLP_input(command_test: str,init_pos=(0,0)):
    # 示例用法
    pos = Position(*init_pos)
    goal_history = []
    # 需要执行的命令
    if command_test:
        command = command_test
    else:
        command = "第一向上走三步第二向右走一步第三向后走一步开始执行"

    # 拆分指令（基于“第一”、“第二”、“第三”等标识符）
    individual_commands = re.split(r"(第一|第二|第三|第四|第五|第六)", command)

    # 清理拆分出的部分，去掉“第一”、“第二”等标识符
    individual_commands = [cmd.strip() for cmd in individual_commands if cmd.strip()]

    # 打印拆分后的指令
    print("Individual Commands:")
    for cmd in individual_commands:
        print(f"- {cmd}")

    # 执行每个指令
    for cmd in individual_commands:
        goal=parse_and_execute(cmd, pos)
        if goal:
            goal_history.append(goal)
    print("Goal History:", goal_history)
    # 输出最终位置
    print(f"Final position: ({pos.x}, {pos.y})")
    return goal_history

if __name__ == "__main__":
    NLP_input("第一向上走三步第二向右走一步第三向后走一步开始执行",init_pos=(0,0))