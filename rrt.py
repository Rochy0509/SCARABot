import numpy as np
from config import (D2, D3, D4, MAX_LIFT,
                    THETA1_MIN, THETA1_MAX, THETA2_MIN, THETA2_MAX,
                    THETA3_MIN, THETA3_MAX)
from collision import is_colliding
from planning import interpolate_joint

REACH = D2 + D3 + D4

GOAL_BIAS = 0.10 #10% of samples
STEP_SIZE = 0.20 #max length of one tree edge
EDGE_RES = 0.05 #spacing of collision checks along edge
MAX_ITERS = 5000
GOAL_TOL = 0.10 

def _wrap(angles):
    return np.arctan2(np.sin(angles), np.cos(angles))

def config_distance(q1, q2):
    q1, q2 = np.asarray(q1), np.asarray(q2)
    dtheta = _wrap(q1[:3] - q2[:3])
    dh = q1[3] - q2[3]

    return np.sqrt(REACH**2 * np.dot(dtheta, dtheta) + dh**2)

def sample_config(rng):
    t1 = rng.uniform(THETA1_MIN, THETA1_MAX)
    t2 = rng.uniform(THETA2_MIN, THETA2_MAX)
    t3 = rng.uniform(THETA3_MIN, THETA3_MAX)
    h1 = rng.uniform(0.0, MAX_LIFT)
    return np.array([t1, t2, t3, h1])

def nearest(nodes, q):
    dists = [config_distance(node, q) for node in nodes]
    return int(np.argmin(dists))

def steer(q_near, q_rand, step_size):
    dtheta = _wrap(q_rand[:3] - q_near[:3])
    dh = q_rand[3] - q_near[3]

    d = config_distance(q_near, q_rand)
    if d <= step_size:
        return np.array([*_wrap(q_rand[:3]), q_rand[3]])
    
    frac = step_size / d
    new_theta = _wrap(q_near[:3] + frac * dtheta)
    new_h = q_near[3] + frac + dh
    return np.array([*new_theta, new_h])

def motion_collision_free(q_a, q_b, obstacles):
    n = max(2, int(config_distance(q_a, q_b)) / EDGE_RES + 1)
    for q in interpolate_joint(q_a, q_b, n):
        if is_colliding(q, obstacles):
            return False
    return True

def reconstruct_path(nodes, parents, goal_index):
    path = []
    i = goal_index
    while i != -1:
        path.append(nodes[i])
        i = parents[i]

    path.reverse()
    return np.array(path)


def rrt(start, goal, obstacles, seed=None, step_size=STEP_SIZE, max_iters=MAX_ITERS,
        goal_tol=GOAL_TOL):
    rng = np.random.default_rng(seed)

    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    if is_colliding(start, obstacles) or is_colliding(goal, obstacles):
        return None
    
    nodes = [start]
    parents = [-1]

    for _ in range(max_iters):
        q_rand = goal if rng.random() < GOAL_BIAS else sample_config(rng)
        i_near = nearest(nodes, q_rand)
        q_new  = steer(nodes[i_near], q_rand, step_size)

        if not motion_collision_free(nodes[i_near], q_new, obstacles):
            continue

        nodes.append(q_new)
        parents.append(i_near)

        if config_distance(q_new, goal) < goal_tol:
            if motion_collision_free(q_new, goal, obstacles):
                nodes.append(goal)
                parents.append(len(nodes) - 2)
                return reconstruct_path(nodes, parents, len(nodes) - 1)
            
    return None