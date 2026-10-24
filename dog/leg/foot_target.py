from leg.leg_ik import leg_ik
import numpy as np

def foot_target(L1, L2, t, T):

    positions = []
    angles = []
    predicted_positions = []

    t %= T

    if t < 1:
        x = 0.05 - 0.10*smoothstep(t)
        y = 0
        z = -0.15

    else:
        u = (t - 1.0) / 1.0
        x = -0.05 + 0.10*smoothstep(u)
        y = 0
        z = (0.48 * u**2 * (1 - u)**2) - 0.15

    angles = leg_ik(L1, L2, x, y, z)
    positions = np.array([x, y, z])

    return (positions, angles)

def smoothstep(s):
    return 3*s**2 - 2*s**3

