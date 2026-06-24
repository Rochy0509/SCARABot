import numpy as np
from kinematics import joint_positions
from config import (LINK_RADIUS, MAST_RADIUS, MAX_LIFT, THETA1_MIN, THETA1_MAX, THETA2_MIN, THETA2_MAX,
                    THETA3_MIN, THETA3_MAX)

EPSILON = 1e-12

def point_segment_dist(p, a, b):
    """based on Real-time Collision Detection by Christer Ericson"""

    ab = b - a

    #checking zero-length
    denom = np.dot(ab, ab)
    if denom < 1e-12:   # a == b: "segment" is a point
        return np.linalg.norm(p - a)

    #project p onto ab, d(t) = a + t*(b-a)
    t = np.dot(p - a, ab) / denom

    #if outside segment, clamp t to the closest endpoint
    t = np.clip(t, 0.0, 1.0)

    #Compute projected position from clamped t
    d = a + t * ab # closest point

    return np.linalg.norm(p - d)

def clamp(n, min, max):
    if (n < min):
        return min
    elif (n > max):
        return max
    else:
        return n

def segment_segment_dist(p1, q1, p2, q2):
    """based on Real-time Collision Detection by Christer Ericson
    Chapter 5, 5.1.9"""

    d1 = q1 - p1 # direction vector of segment S1
    d2 = q2 - p2 # direction vector of segment S2

    r = p1 - p2

    a = np.dot(d1, d1) #squared length of segment S1, always nonnegative
    e = np.dot(d2, d2) #squared length of segment S2, always nonnegative
    f = np.dot(d2, r)

    #checking if either or both segments degenerate into points
    if (a <= EPSILON and e <= EPSILON):
        #both segments degenerate into points
        s = t = 0.0
        c1 = p1
        c2 = p2

        return np.linalg.norm(c1 - c2)
    
    elif (a <= EPSILON):
        #only first segment degenerates into a point
        s = 0.0
        t = f / e # s = 0 => t = (b*s + f) / e = f / e
        t = clamp(t, 0.0, 1.0)
    
    else:
        
        c = np.dot(d1, r)
        if (e <= EPSILON):
            #second segment degenerates into a point
            t = 0.0
            s = clamp(-c / a, 0.0, 1.0) # t = 0 => s = (b*t - c) / a = -c / a

        else: 

            #general nondegenerate case starts here
            b = np.dot(d1, d2)
            denom = a * e - b * b #always nonnegative

            #if not parallel, compute closest point on L1 to L2
            if (denom != 0.0):
                s = clamp((b * f - c * e )/ denom, 0.0, 1.0)
            else: s = 0.0

            #compute point on L2 closest to S1(s)
            #t = dot(P1 + D1*s) - P2, D2) / dot(D2, D2) = (b*s + f) / e

            t = (b * s + f) / e

            #if t in [0, 1] done. else clamp t
            # s = dot((P2 + D2*t) - P1, D1) / dot(D1, D1) = (t * b - c) / a
            if (t < 0.0):
                t = 0.0
                s = clamp(-c / a, 0.0, 1.0)
            elif (t > 1.0):
                t = 1.0
                s = clamp((b - c) / a, 0.0, 1.0)
            
    c1 = p1 + d1 * s
    c2 = p2 + d2 * t
    return np.linalg.norm(c1 - c2)

def robot_capsules(config):
    pts = joint_positions(config[:3], config[3])
    return [(pts[i], pts[i + 1], LINK_RADIUS) for i in range(len(pts) - 1)]

def mast_capsule():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, MAX_LIFT])
    return (a, b, MAST_RADIUS)

def capsule_sphere(capsule, sphere):
    a, b, r_cap = capsule
    center, r_sph = sphere[:3], sphere[3]
    return point_segment_dist(center, a, b) < r_cap + r_sph

def capsule_capsule(cap1, cap2):
    a1, b1, r1 = cap1
    a2, b2, r2 = cap2
    return segment_segment_dist(a1, b1, a2, b2) < r1 + r2

def world_collision(config, obstacles):
    if obstacles is None or len(obstacles) == 0:
        return False                       # no obstacles
    caps = robot_capsules(config)
    return any(capsule_sphere(cap, obs) for cap in caps for obs in obstacles)

def self_collision(config):
    L0, L1, L2 = robot_capsules(config)
    mast = mast_capsule()
    return (capsule_capsule(L0, L2)
            or capsule_capsule(L1, mast)
            or capsule_capsule(L2, mast))

def in_limits(config):
    t1, t2, t3, h1 = config
    return (THETA1_MIN <= t1 <= THETA1_MAX and
            THETA2_MIN <= t2 <= THETA2_MAX and
            THETA3_MIN <= t3 <= THETA3_MAX and
            0.0 <= h1 <= MAX_LIFT)

def is_colliding(config, obstacles):
    if not in_limits(config):
        return True
    if self_collision(config):
        return True
    return world_collision(config, obstacles)
