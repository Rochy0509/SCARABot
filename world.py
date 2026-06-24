import numpy as np
from config import D2, D3, D4, H2, H3, MAST_RADIUS, MAX_LIFT

REACH = D2 + D3 + D4
Z_TOP = MAX_LIFT + H2 + H3


def sample_obstacles(n, seed=None, radius_range=(0.03, 0.08), keepout=None):
    
    rng = np.random.default_rng(seed)
    obstacles = []
    tries = 0
    max_tries = 100 * n

    while len(obstacles) < n and tries < max_tries:
        tries += 1

        #draw one candidate: cx, cy, cz, r
        cx, cy = rng.uniform(-REACH, REACH, 2)
        cz = rng.uniform(0, Z_TOP)
        r = rng.uniform(*radius_range)

        horizontal_dist = np.hypot(cx, cy) #distance from candidate center to z-axis
        if horizontal_dist < MAST_RADIUS + r: 
            continue #if obstacles is generated into the column then rejects it

        if keepout is not None:
            valid = True
            for k in keepout: #k = [kx, ky, kz, kr]
                center_dist = np.sqrt((cx-k[0])**2 + (cy-k[1])**2 + (cz-k[2])**2)
                if center_dist < r + k[3]:
                    valid = False
                    break
            if not valid:
                continue
        obstacles.append([cx, cy, cz, r])

    return np.array(obstacles)


