import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("leg.xml")
data = mujoco.MjData(model)

joint_names = ["theta1", "theta2", "theta3"]
motor_names = ["motor1", "motor2", "motor3"]


target = np.deg2rad([20, 30, 60])

# Initial gains for this experiment; we may adjust them.
kp = np.array([5.0, 5.0, 5.0])
kd = np.array([0.15, 0.15, 0.15])

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    playback_speed = 0.1
    while viewer.is_running():
        frame_start = time.perf_counter()

        with viewer.lock():
            actual = np.array([
                data.joint(name).qpos[0] for name in joint_names
            ])

            print("Target angles:", np.rad2deg(target))
            print("Actual angles:", np.rad2deg(actual))
            print("Error (degrees):", np.rad2deg(target - actual))
            # 20 × 0.001 s = 0.02 s of simulated motion.
            for _ in range(20):
                for i, (joint, motor) in enumerate(
                    zip(joint_names, motor_names)
                ):
                    angle = data.joint(joint).qpos[0]
                    velocity = data.joint(joint).qvel[0]

                    torque = (
                        kp[i] * (target[i] - angle)
                        - kd[i] * velocity
                    )

                    data.actuator(motor).ctrl[0] = torque

                mujoco.mj_step(model, data)

        viewer.sync()

        work_time = time.perf_counter() - frame_start
        time.sleep(max(0.0, 0.02 / playback_speed - work_time))
