import numpy as np

def smoothstep(a, b, T, step):

    t = np.linspace(0.0, T, int(T * step))

    s = t/T
    f = (b - a) * (3*s**2 - 2*s**3) + a

    return f