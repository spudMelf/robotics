import numpy as np

def leg_fk(l1, l2, theta1, theta2, theta3):
    theta1 = np.deg2rad(theta1)
    theta2 = np.deg2rad(theta2)
    theta3 = np.deg2rad(theta3)

    x = l1*np.sin(theta2)+l2*np.sin(theta2+theta3)
    y = (l1*np.cos(theta2)+l2*np.cos(theta2+theta3))*np.sin(theta1)
    z = -(l1*np.cos(theta2)+l2*np.cos(theta2+theta3))*np.cos(theta1)

    return np.array([x, y, z])



