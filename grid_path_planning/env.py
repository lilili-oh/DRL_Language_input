import random
import numpy as np

class GridWorldEnv:
    """A small grid-world environment for path planning.

    State: concatenation of normalized agent (x,y), goal (x,y), and flattened obstacle map (0/1)
    Actions: 0=up,1=right,2=down,3=left
    """

    def __init__(self, width=6, height=6, obstacles=None, max_steps=100):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.obstacles = np.zeros((height, width), dtype=np.uint8)
        if obstacles is not None:
            self.set_obstacles(obstacles)
        self.reset()

    def set_obstacles(self, obstacles):
        # obstacles: iterable of (y,x) coordinates
        self.obstacles.fill(0)
        for (y, x) in obstacles:
            if 0 <= y < self.height and 0 <= x < self.width:
                self.obstacles[y, x] = 1

    def sample_empty_cell(self):
        empties = list(zip(*np.where(self.obstacles == 0)))
        return random.choice(empties)

    def reset(self, start=None, goal=None):
        # Ensure start and goal are on empty cells and are not the same
        # Cases:
        # - both None: sample both and ensure different
        # - start provided, goal None: sample goal until different
        # - start None, goal provided: sample start until different
        # - both provided: raise ValueError if identical
        # normalize provided positions to tuples (y,x) if needed
        def _to_pos(p):
            if p is None:
                return None
            # allow list/tuple/numpy array inputs
            if isinstance(p, np.ndarray):
                p = tuple(int(x) for x in p.tolist())
            elif isinstance(p, (list, tuple)):
                p = tuple(int(x) for x in p)
            else:
                # unknown type - keep as-is and allow later validation to catch
                p = p
            return p

        start = _to_pos(start)
        goal = _to_pos(goal)

        # helper to validate a provided position is within bounds and on empty cell
        def _valid_empty(pos):
            if pos is None:
                return False
            if not (isinstance(pos, tuple) and len(pos) == 2):
                return False
            y, x = pos
            if not (0 <= y < self.height and 0 <= x < self.width):
                return False
            return self.obstacles[y, x] == 0

        if start is None and goal is None:
            self.agent_pos = self.sample_empty_cell()
            self.goal_pos = self.sample_empty_cell()
            while self.goal_pos == self.agent_pos:
                self.goal_pos = self.sample_empty_cell()
        elif start is None and goal is not None:
            # provided goal must be valid empty cell
            if not _valid_empty(goal):
                raise ValueError("provided goal is invalid or not on an empty cell")
            self.goal_pos = goal
            # sample start but avoid the provided goal
            empties = list(zip(*np.where(self.obstacles == 0)))
            if len(empties) < 2:
                raise ValueError("not enough empty cells to place distinct start and goal")
            self.agent_pos = self.sample_empty_cell()
            while self.agent_pos == self.goal_pos:
                self.agent_pos = self.sample_empty_cell()
        elif start is not None and goal is None:
            # provided start must be valid empty cell
            if not _valid_empty(start):
                raise ValueError("provided start is invalid or not on an empty cell")
            self.agent_pos = start
            # sample goal until different
            empties = list(zip(*np.where(self.obstacles == 0)))
            if len(empties) < 2:
                raise ValueError("not enough empty cells to place distinct start and goal")
            self.goal_pos = self.sample_empty_cell()
            while self.goal_pos == self.agent_pos:
                self.goal_pos = self.sample_empty_cell()
        else:
            # both provided - validate types and emptiness
            if start == goal:
                raise ValueError("start and goal positions must be different")
            if not _valid_empty(start):
                raise ValueError("provided start is invalid or not on an empty cell")
            if not _valid_empty(goal):
                raise ValueError("provided goal is invalid or not on an empty cell")
            self.agent_pos = start
            self.goal_pos = goal
        self.steps = 0
        print(f"Environment reset: start={self.agent_pos}, goal={self.goal_pos}")
        return self._get_state()

    def _get_state(self):
        ay, ax = self.agent_pos
        gy, gx = self.goal_pos
        # normalized coordinates in [0,1]
        coords = np.array([ax / (self.width - 1), ay / (self.height - 1),
                           gx / (self.width - 1), gy / (self.height - 1)], dtype=np.float32)
        obs_flat = self.obstacles.flatten().astype(np.float32)
        return np.concatenate([coords, obs_flat])

    def step(self, action):
        # action: 0 up,1 right,2 down,3 left
        y, x = self.agent_pos
        if action == 0:
            ny, nx = y - 1, x
        elif action == 1:
            ny, nx = y, x + 1
        elif action == 2:
            ny, nx = y + 1, x
        elif action == 3:
            ny, nx = y, x - 1
        else:
            ny, nx = y, x

        hit_wall = False
        # bounds check
        if not (0 <= ny < self.height and 0 <= nx < self.width):
            ny, nx = y, x
            hit_wall = True

        # collision with obstacle
        collision = False
        if self.obstacles[ny, nx] == 1:
            collision = True
            ny, nx = y, x  # don't move into obstacle

        self.agent_pos = (ny, nx)
        self.steps += 1

        done = False
        reward = -0.1  # step penalty
        if collision:
            reward -= 1.0
        if (ny, nx) == self.goal_pos:
            reward += 100.0
            done = True
        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), float(reward), done, {"collision": collision, "hit_wall": hit_wall}

    def render(self):
        grid = np.full((self.height, self.width), fill_value=".", dtype=object)
        for (y, x) in zip(*np.where(self.obstacles == 1)):
            grid[y, x] = "#"
        ay, ax = self.agent_pos
        gy, gx = self.goal_pos
        grid[gy, gx] = "G"
        grid[ay, ax] = "A"
        for row in grid:
            print(" ".join(row))
