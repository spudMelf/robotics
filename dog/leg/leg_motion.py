import numpy as np
import mujoco
import mujoco.viewer
import time

from leg_ik import leg_ik
from leg_fk import leg_fk

L1 = 0.1
L2 = 0.1

times = np.arange(0, 2, (1/60))

positions = []
angles = []
predicted_positions = []

T = 2

with open("output.txt", "w") as file:
    for t in times:
        x = 0.1
        y = 0
   
        z = -0.12 + 0.03 * np.sin((2*(np.pi)*t)/T)
        positions.append([x, y, z])

        angle = leg_ik(L1, L2, x, y, z)
        pred_position = leg_fk(L1, L2, angle[0], angle[1], angle[2])

        angles.append(angle)
        predicted_positions.append(pred_position)
        
        file.write(str(angle) + "\n")
        file.write(str(pred_position) + "\n")
        file.write("\n")

model = mujoco.MjModel.from_xml_path('leg.xml')
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.perf_counter()
    while viewer.is_running():
        frame_start = time.perf_counter()
        elapsed_seconds = frame_start - start_time
        frame = int(elapsed_seconds * 60) % 120

        angle_rad = np.deg2rad(angles[frame])
        data.joint('theta1').qpos[0] = angle_rad[0]
        data.joint('theta2').qpos[0] = angle_rad[1]
        data.joint('theta3').qpos[0] = angle_rad[2]

        mujoco.mj_forward(model, data)

        viewer.sync()

        work_time = time.perf_counter() - frame_start
        time.sleep(max(0.0, 1/60 - work_time))
