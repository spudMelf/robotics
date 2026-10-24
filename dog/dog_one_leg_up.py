import time
import sys
import numpy as np
import mujoco
import mujoco.viewer
from leg.foot_target import foot_target
from leg.leg_fk import leg_fk
from leg.leg_ik import leg_ik
from smoothstep import smoothstep
import robot_structure as robot


kp = np.array([5.0, 5.0, 5.0])
kd = np.array([0.15, 0.15, 0.15])

model = mujoco.MjModel.from_xml_path("dog.xml")
data = mujoco.MjData(model)

T_up = 2
T_down = 1
step = 1000

last_positions = []

with mujoco.viewer.launch_passive(model, data) as viewer: 

    start_positions = np.array([
        [0.0, 0.0, -0.16],
        [0.0, 0.0, -0.16],
        [0.0, 0.0, -0.16],
        [0.0, 0.0, -0.16],
    ])
    start_angles = np.deg2rad([leg_ik(robot.L1, robot.L2, *(start_positions[i])) for i in range(len(start_positions))])


    target_positions_down = np.array([
        [0.00, 0.0, -0.16],  # front left
        [0.00, 0.0, -0.16],  # front right
        [0.00, 0.0, -0.16],  # rear left
        [0.00, 0.0, -0.16],  # rear right
    ])

    target_positions_up = np.array([
        [0.00, 0.0, -0.14],  # front left
        [0.00, 0.0, -0.16],  # front right
        [0.00, 0.0, -0.16],  # rear left
        [0.00, 0.0, -0.16],  # rear right
    ])
    # not needed since we compute angles on demand now
    # target_angles = np.deg2rad([leg_ik(robot.L1, robot.L2, *(target_positions[i])) for i in range(len(target_positions))])

    # Set the starting pose once.
    for i, (j1, j2, j3) in enumerate(robot.ALL_JOINTS):
        data.joint(j1).qpos[0] = start_angles[i][0] # set theta1 to start_angle
        data.joint(j2).qpos[0] = start_angles[i][1] # set theta2
        data.joint(j3).qpos[0] = start_angles[i][2] # set theta3

    # calculate trajectories
    position_trajectories_down = [[None] * 3 for _ in range(4)]
    for i in range(4):
        for j in range(3):
            position_trajectories_down[i][j] = smoothstep(start_positions[i][j], target_positions_down[i][j], T_up, step)

    position_trajectories_up = [[None] * 3 for _ in range(4)]
    for i in range(4):
            for j in range(3):
                position_trajectories_up[i][j] = smoothstep(target_positions_down[i][j], target_positions_up[i][j], T_down, step)


    mujoco.mj_forward(model, data)
    while viewer.is_running():
        with viewer.lock():
            start_time = time.perf_counter()
            for _ in range(20):
                # update our target
                time_step = int(data.time * step) 
                if(time_step < T_up * step):
                    for i, (joint_names, motor_names) in enumerate(zip(robot.ALL_JOINTS, robot.ALL_MOTORS)):
                        # target_angles for one leg
                        angle_trajectory = np.deg2rad(leg_ik(robot.L1, robot.L2, 
                                                        position_trajectories_down[i][0][time_step],
                                                        position_trajectories_down[i][1][time_step],
                                                        position_trajectories_down[i][2][time_step]))

                        for j, (joint, motor) in enumerate(zip(joint_names, motor_names)):
                            angle = data.joint(joint).qpos[0]
                            velocity = data.joint(joint).qvel[0]

                            torque = (
                                kp[j] * (angle_trajectory[j] - angle)
                                - kd[j] * velocity
                            )

                            data.actuator(motor).ctrl[0] = torque
                elif(time_step < (T_up + T_down) * step):
                    time_step_adj = (time_step - 2000) % 2000
                    # update our target
                    for i, (joint_names, motor_names) in enumerate(zip(robot.ALL_JOINTS, robot.ALL_MOTORS)):
                        # target_angles for one leg
                        angle_trajectory = np.deg2rad(leg_ik(robot.L1, robot.L2, 
                                                        position_trajectories_up[i][0][time_step_adj],
                                                        position_trajectories_up[i][1][time_step_adj],
                                                        position_trajectories_up[i][2][time_step_adj]))

                        for j, (joint, motor) in enumerate(zip(joint_names, motor_names)):
                            angle = data.joint(joint).qpos[0]
                            velocity = data.joint(joint).qvel[0]

                            torque = (
                                kp[j] * (angle_trajectory[j] - angle)
                                - kd[j] * velocity
                            )

                            data.actuator(motor).ctrl[0] = torque
                else:
                    continue
                mujoco.mj_step(model, data)

        viewer.sync()

        work_time = time.perf_counter() - start_time
        time.sleep(max(0.0, 0.02 - work_time))

    for joint_names in robot.ALL_JOINTS:
        leg_angles = np.array([
            data.joint(name).qpos[0] for name in joint_names
        ])
        actual_position = leg_fk(robot.L1, robot.L2, *np.rad2deg(leg_angles))
        last_positions.append(actual_position)

ground_id = model.geom("floor").id

total_normal = 0
for i in range(data.ncon):
    contact = data.contact[i]

    if contact.geom1 == ground_id:
        print(f"{model.geom(contact.geom1).name} is touching {model.geom(contact.geom2).name}")
        forces = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, forces)
        normal = forces[0]
        total_normal += normal
        print(f"force is: {normal}\n")

print(total_normal)

print(data.site("front_left_foot").xpos.copy())
print(data.site("front_right_foot").xpos.copy())
print(data.site("rear_left_foot").xpos.copy())
print(data.site("rear_right_foot").xpos.copy())

for i in range(model.nu):
    name = model.actuator(i).name
    force = data.actuator_force[i]

    print(f"{name}: {force:.4f}")

hip = data.body("front_left_hip")
foot = data.site("front_left_foot")
chassis = data.body("chassis")

R = hip.xmat.reshape(3, 3)

actual_local = R.T @ (foot.xpos - hip.xpos)
target_local = np.array([0.0, 0.0, -0.14])

print("Target in hip frame:", target_local)
print("Actual in hip frame:", actual_local)
print("Tracking error:", target_local - actual_local)

print("Hip world position:", hip.xpos)
print("Chassis world position:", chassis.xpos)
print("Chassis quaternion:", chassis.xquat)