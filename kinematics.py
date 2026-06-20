import numpy as np 
from config import D2, D3, D4, H2, H3

def link_transform(theta, a, d):
    T = np.array([[np.cos(theta), -np.sin(theta), 0, a],
                  [np.sin(theta), np.cos(theta), 0, 0],
                  [0, 0, 1, d],
                  [0, 0, 0, 1]])
    return T

def forward_kinematics(thetas, h1):
    T_0_t = (
        link_transform(0, 0, h1) #lift elevator (prismatic)
        @ link_transform(thetas[0], 0, 0) #theta 1 (revolute)
        @ link_transform(thetas[1], D2, H2) #theta 2 (revolute) - D2, H2 are constant
        @ link_transform(thetas[2], D3, H3) #theta 3 (revolute) - D3, H3 are constant
        @ link_transform(0, D4, 0) #tool - D4 is constant
    )

    return T_0_t

def joint_positions(thetas, h1):
    P_0 = (0, 0)
    P_1 = ((D2*np.cos(thetas[0])), (D2*np.sin(thetas[0]))) #elbow
    P_2 = (((D2*np.cos(thetas[0]))+(D3*np.cos(thetas[0]+thetas[1]))), ((D2*np.sin(thetas[0]))+(D3*np.sin(thetas[0]+thetas[1])))) #wrist
    P_3 = (((D2*np.cos(thetas[0]))+(D3*np.cos(thetas[0] + thetas[1])) + (D4*np.cos(thetas[0] + thetas[1] + thetas[2]))), 
           ((D2*np.sin(thetas[0])) + (D3*np.sin(thetas[0] + thetas[1])) + (D4*np.sin(thetas[0] + thetas[1] + thetas[2])))) #tool

    return np.array([P_0, P_1, P_2, P_3])

def inverse_kinematics(x, y, z, gamma):
    h1 = z - H2 - H3 #lift elevator position
    W_x = x - (D4*np.cos(gamma)) #wrist x
    W_y = y - (D4*np.sin(gamma)) #wrist y

    # law of cosines on the triangle
    cos_theta2 = (W_x**2 + W_y**2 - D2**2 - D3**2) / (2*D2*D3) #
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0) # clamp so float overshoot at full reach can't NaN the sqrt
    sin_theta2 = np.sqrt(1 - cos_theta2**2) # matching sine magnitude;

    solutions = []
    for sign in (+1, -1):
        theta_2 = np.arctan2(sign * sin_theta2, cos_theta2)
        theta_1 = np.arctan2(W_y, W_x) - np.arctan2(D3*np.sin(theta_2), D2 +
                                                    D3 * np.cos(theta_2))
        theta_3 = gamma - theta_1 - theta_2
        solutions.append((theta_1, theta_2, theta_3, h1))

    return solutions
