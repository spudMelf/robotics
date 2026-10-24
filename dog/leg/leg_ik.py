import numpy as np

def leg_ik(L1, L2, x, y, z):
    theta1 = np.arctan2(y, -z)
    D = np.sqrt(y**2 + z**2)

    r = np.sqrt(x**2 + D**2)
    theta3 = np.arccos((r*r - L1**2 - L2**2)/(2*L1*L2))

    A = L1 + L2*np.cos(theta3)
    B = L2*np.sin(theta3)
    theta2 = np.arctan2(A*x - B*D, A*D + B*x)

    return np.rad2deg(np.array([theta1, theta2, theta3]))



