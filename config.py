import numpy as np 

#meters
D2 = 1.0
D3 = 0.9
D4 = 0.9
H2 = 0.15
H3 = 0.15
MAX_LIFT = 1.2
LINK_RADIUS = 0.1
MAST_RADIUS = 0.15

# joint limits (radians).
THETA1_MIN, THETA1_MAX = np.radians(-90), np.radians(90)
THETA2_MIN, THETA2_MAX = np.radians(-180), np.radians(180)
THETA3_MIN, THETA3_MAX = np.radians(-180), np.radians(180)