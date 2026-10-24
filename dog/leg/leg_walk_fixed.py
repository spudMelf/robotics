import time
import numpy as np
import mujoco
import mujoco.viewer
from foot_target import foot_target
from leg_fk import leg_fk
from leg_ik import leg_ik

L1 = 0.1
L2 = 0.1
joint_names = ["theta1", "theta2", "theta3"]
motor_names = ["motor1", "motor2", "motor3"]

kp = np.array([5.0, 5.0, 5.0])
kd = np.array([0.15, 0.15, 0.15])

model = mujoco.MjModel.from_xml_path("leg.xml")
data = mujoco.MjData(model)

T = 2

with mujoco.viewer.launch_passive(model, data) as viewer: 
    errors = []
    foot_errors = []
    positions = np.array([0.0, 0.0, -0.15])
    target = np.deg2rad(leg_ik(L1, L2, *positions))

    # Set the starting pose once.
    for name, angle in zip(joint_names, target):
        data.joint(name).qpos[0] = angle

    mujoco.mj_forward(model, data)
    while viewer.is_running():
        with viewer.lock():
            actual = np.array([
                data.joint(name).qpos[0] for name in joint_names
            ])

            # print("Target angles:", np.rad2deg(target))
            # print("Actual angles:", np.rad2deg(actual))
            # print("Error (degrees):", np.rad2deg(target - actual))
            # 20 × 0.001 s = 0.02 s of simulated motion.
            # target = np.deg2rad(angles[frame])i

            start_time = time.perf_counter()
            for _ in range(20):
                # Wait 1 second, then lower the target over 2 seconds.
                s = np.clip((data.time - 1.0) / 2.0, 0.0, 1.0)
                blend = 3*s**2 - 2*s**3

                positions = np.array([
                    0.0,
                    0.0,
                    -0.15 - 0.012 * blend
                ])
                target = np.deg2rad(leg_ik(L1, L2, *positions))
                for i, (joint, motor) in enumerate(zip(joint_names, motor_names)):
                    angle = data.joint(joint).qpos[0]
                    velocity = data.joint(joint).qvel[0]

                    torque = (
                        kp[i] * (target[i] - angle)
                        - kd[i] * velocity
                    )

                    data.actuator(motor).ctrl[0] = torque

                actual = np.array([
                    data.joint(name).qpos[0] for name in joint_names
                ])
                if data.time >= 4.0:
                    errors.append(np.rad2deg(target - actual))

                    actual_deg = np.rad2deg(actual)
                    actual_position = leg_fk(L1, L2, *actual_deg)
                    foot_errors.append(positions - actual_position)
                
                mujoco.mj_step(model, data)

        viewer.sync()

        work_time = time.perf_counter() - start_time
        time.sleep(max(0.0, 0.02 - work_time))

    actual = np.array([
        data.joint(name).qpos[0] for name in joint_names
    ])
    actual_position = leg_fk(L1, L2, *np.rad2deg(actual))

    print("Target foot position (m):", positions)
    print("Actual foot position (m):", actual_position)
    print("Target minus actual [x, y, z] (mm):", 1000 * (positions - actual_position))

    if errors:
        errors = np.asarray(errors)
        print("RMS error per joint (degrees):",
              np.sqrt(np.mean(errors**2, axis=0)))
        print("Maximum absolute error per joint (degrees):",
              np.max(np.abs(errors), axis=0))

        foot_errors = np.asarray(foot_errors)

        # One Euclidean distance per sample, converted to millimetres.
        foot_distances_mm = 1000 * np.linalg.norm(foot_errors, axis=1)

        print("RMS foot-position error (mm):",
              np.sqrt(np.mean(foot_distances_mm**2)))
        print("Maximum foot-position error (mm):",
              np.max(foot_distances_mm))

mujoco.mj_forward(model, data)
print("Actual foot position in world (m):",
      data.site("foot").xpos)

floor_id = model.geom("floor").id
normal_force = 0.0

for contact_id in range(data.ncon):
    contact = data.contact[contact_id]

    if floor_id in (contact.geom1, contact.geom2):
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, contact_id, force)
        normal_force += force[0]

print("Upward ground force (N):", normal_force)
