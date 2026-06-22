import numpy as np
from kinematics import forward_kinematics, inverse_kinematics, pose_from_matrix
from config import D2, D3, D4, H2, H3, MAX_LIFT

def interpolate_joint(start_config, target_config, steps):
    start_config = np.array(start_config)
    target_config = np.array(target_config)
    trajectory = np.linspace(start_config, target_config, steps) #linear interpolation
    return trajectory

def nearest_branch(solutions, reference_config):
    if not solutions: #checking for empty solutions
        return None
    
    reference = np.array(reference_config) #current pose of the arm
    
    #calculating the distance of the solutions
    distances = [np.linalg.norm(np.array(sol) - reference) for sol in solutions]

    best = int(np.argmin(distances)) #smallest distance is selected

    return solutions[best]

def interpolate_cartesian(start_pose, target_pose, steps, start_config):
    start_pose = np.array(start_pose)
    target_pose = np.array(target_pose)

    pose_sweep = np.linspace(start_pose, target_pose, steps) #straight line space

    trajectory = []
    prev_config = start_config #seed continuity from arm's current pose

    for pose in pose_sweep:
        x, y, z, gamma = pose
        solutions = inverse_kinematics(x, y, z, gamma)
        if not solutions:
            return None, False
        config = nearest_branch(solutions, prev_config)
        trajectory.append(config)
        prev_config = config

    return np.array(trajectory), True

def max_joint_jump(trajectory):
    angle_steps = np.diff(trajectory[:, :3], axis=0) #per-step change of the 3 angles
    return np.max(np.abs(angle_steps)) #single biggest swing anywhere on the path

def choose_trajectory(start_config, target_pose, steps, threshold):
    T = forward_kinematics(start_config[:3], start_config[3])
    start_pose = pose_from_matrix(T) #where the arm is located 

    target_solutions = inverse_kinematics(*target_pose)
    if not target_solutions:
        return None, "unreachable"
    target_config = nearest_branch(target_solutions, start_config) #target as joints

    cartesian, feasible = interpolate_cartesian(start_pose, target_pose, steps, start_config)
    if feasible and max_joint_jump(cartesian) <= threshold:
        return cartesian, "cartesian-space solution found!"
    
    #fallback to joint space
    joint = interpolate_joint(start_config, target_config, steps)
    return joint, "joint-space solution found!"

  
def sample_target_poses(n, seed=None):
    rng = np.random.default_rng(seed)
    r_min = abs(D2 - D3) + D4
    r_max = D2 + D3 + D4
    r   = rng.uniform(r_min, r_max, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    z   = rng.uniform(H2 + H3, MAX_LIFT + H2 + H3, n)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z, phi])