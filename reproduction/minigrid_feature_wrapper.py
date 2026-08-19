"""
MiniGrid Feature Wrapper — Converts grid observations to named numeric vectors.

MiniGrid environments produce dict observations with a partial-view image
(7×7×3 uint8 grid), agent direction (int), and a text mission string.
This wrapper extracts ~14 compact, semantically meaningful features that
CBS / DT surrogates can consume directly.

The wrapper accesses the underlying grid to compute exact positions and
distances rather than relying on the limited partial view.

Usage:
    import gymnasium as gym
    from reproduction.minigrid_feature_wrapper import MiniGridFeatureWrapper

    env = gym.make("MiniGrid-Dynamic-Obstacles-8x8-v0")
    env = MiniGridFeatureWrapper(env)
    obs, info = env.reset(seed=42)
    # obs is now a flat float32 vector of shape (14,)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# Feature names exported for CBS / replay saving
MINIGRID_FEATURE_NAMES = [
    "agent_x",               # 0: agent column position (normalized 0..1)
    "agent_y",               # 1: agent row position (normalized 0..1)
    "agent_dir",             # 2: direction (0=right,1=down,2=left,3=up) / 3
    "goal_dx",               # 3: signed dx to goal (normalized by grid size)
    "goal_dy",               # 4: signed dy to goal (normalized by grid size)
    "dist_to_goal",          # 5: Chebyshev distance to goal (normalized)
    "front_is_free",         # 6: 1 if cell in front is walkable, else 0
    "front_is_obstacle",     # 7: 1 if cell in front is an obstacle (ball)
    "left_free",             # 8: 1 if left-turn-then-forward cell is free
    "right_free",            # 9: 1 if right-turn-then-forward cell is free
    "nearest_obs_dist",      # 10: Chebyshev distance to nearest obstacle (norm.)
    "nearest_obs_dx",        # 11: signed dx to nearest obstacle (normalized)
    "nearest_obs_dy",        # 12: signed dy to nearest obstacle (normalized)
    "num_obstacles_visible", # 13: count of obstacles in the 7×7 partial view
]

# Direction vectors: 0=right(+x), 1=down(+y), 2=left(-x), 3=up(-y)
_DIR_VEC = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int32)


class MiniGridFeatureWrapper(gym.ObservationWrapper):
    """Convert MiniGrid dict obs → flat float32 feature vector."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        n_features = len(MINIGRID_FEATURE_NAMES)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_features,), dtype=np.float32,
        )
        # Cache grid size (available after make, before reset)
        self._grid_size = None

    def observation(self, obs):
        """Extract feature vector from MiniGrid observation dict."""
        uw = self.env.unwrapped
        grid = uw.grid
        w, h = grid.width, grid.height
        self._grid_size = max(w, h)
        gs = float(self._grid_size)

        ax, ay = uw.agent_pos
        a_dir = uw.agent_dir

        # ── Goal position ──
        gx, gy = self._find_object(grid, "goal", w, h)
        if gx is None:  # shouldn't happen, but be safe
            gx, gy = w - 2, h - 2

        goal_dx = (gx - ax) / gs
        goal_dy = (gy - ay) / gs
        dist_goal = max(abs(gx - ax), abs(gy - ay)) / gs

        # ── Obstacle positions ──
        obstacles = self._find_all_objects(grid, "ball", w, h)

        # Nearest obstacle
        if obstacles:
            dists = [max(abs(ox - ax), abs(oy - ay)) for ox, oy in obstacles]
            idx_near = int(np.argmin(dists))
            near_ox, near_oy = obstacles[idx_near]
            near_dist = dists[idx_near] / gs
            near_dx = (near_ox - ax) / gs
            near_dy = (near_oy - ay) / gs
        else:
            near_dist = 1.0
            near_dx = 0.0
            near_dy = 0.0

        # ── Directional checks ──
        fwd = _DIR_VEC[a_dir]
        fx, fy = ax + fwd[0], ay + fwd[1]
        front_free = float(self._is_free(grid, fx, fy, w, h))
        front_obs = float(self._is_obstacle(grid, fx, fy, w, h))

        # Left = turn left then forward
        left_dir = (a_dir - 1) % 4
        lv = _DIR_VEC[left_dir]
        lx, ly = ax + lv[0], ay + lv[1]
        left_free = float(self._is_free(grid, lx, ly, w, h))

        # Right = turn right then forward
        right_dir = (a_dir + 1) % 4
        rv = _DIR_VEC[right_dir]
        rx, ry = ax + rv[0], ay + rv[1]
        right_free = float(self._is_free(grid, rx, ry, w, h))

        # ── Obstacles visible in partial view ──
        if isinstance(obs, dict) and "image" in obs:
            img = obs["image"]
            # channel 0 = object type; ball = 6
            num_vis = int(np.sum(img[:, :, 0] == 6))
        else:
            num_vis = 0

        features = np.array([
            ax / gs,                    # agent_x
            ay / gs,                    # agent_y
            a_dir / 3.0,               # agent_dir (0..1)
            goal_dx,                    # goal_dx
            goal_dy,                    # goal_dy
            dist_goal,                  # dist_to_goal
            front_free,                 # front_is_free
            front_obs,                  # front_is_obstacle
            left_free,                  # left_free
            right_free,                 # right_free
            near_dist,                  # nearest_obs_dist
            near_dx,                    # nearest_obs_dx
            near_dy,                    # nearest_obs_dy
            float(num_vis),             # num_obstacles_visible
        ], dtype=np.float32)

        return features

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _find_object(grid, obj_type, w, h):
        for i in range(w):
            for j in range(h):
                cell = grid.get(i, j)
                if cell is not None and cell.type == obj_type:
                    return i, j
        return None, None

    @staticmethod
    def _find_all_objects(grid, obj_type, w, h):
        results = []
        for i in range(w):
            for j in range(h):
                cell = grid.get(i, j)
                if cell is not None and cell.type == obj_type:
                    results.append((i, j))
        return results

    @staticmethod
    def _is_free(grid, x, y, w, h):
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        cell = grid.get(x, y)
        return cell is None or cell.can_overlap()

    @staticmethod
    def _is_obstacle(grid, x, y, w, h):
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        cell = grid.get(x, y)
        return cell is not None and cell.type == "ball"


def make_minigrid_env(env_id="MiniGrid-Dynamic-Obstacles-8x8-v0", **kwargs):
    """Create a MiniGrid env wrapped with feature extraction."""
    import minigrid  # noqa: F401 — registers envs
    env = gym.make(env_id, **kwargs)
    return MiniGridFeatureWrapper(env)
