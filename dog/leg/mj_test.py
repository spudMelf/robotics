import mujoco
import numpy as np
from leg_ik import leg_ik

model = mujoco.MjModel.from_xml_path("leg.xml")
data = mujoco.MjData(model)

angles = leg_ik(0.1, 0.1, 0.1, 0.1, -0.1)
angles_rad = np.deg2rad(angles)
data.joint("theta1").qpos[0] = angles_rad[0]
data.joint("theta2").qpos[0] = angles_rad[1]
data.joint("theta3").qpos[0] = angles_rad[2]

mujoco.mj_forward(model, data)

foot_pos = data.site("foot").xpos.copy()

print("MuJoCo:", foot_pos)
