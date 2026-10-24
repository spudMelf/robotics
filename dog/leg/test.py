import numpy as np
from leg_fk import leg_fk
from leg_ik import leg_ik

np.set_printoptions(precision=2, suppress=True)

l1 = 0.1
l2 = 0.1

x = 0.1
y = 0.1
z = -0.1

angles = leg_ik(l1, l2, x, y, z)
positions = leg_fk(l1, l2, angles[0], angles[1], angles[2])

print(f"angles:\t\t {angles}")
print(f"positions:\t {positions}")
