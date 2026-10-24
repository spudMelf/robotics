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

T = 2
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


    target_positions = np.array([
        [0.00, 0.0, -0.16],  # front left
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
    position_trajectories = [[None] * 3 for _ in range(4)]
    for i in range(4):
        for j in range(3):
            position_trajectories[i][j] = smoothstep(start_positions[i][j], target_positions[i][j], T, step)

    mujoco.mj_forward(model, data)
    while viewer.is_running():
        with viewer.lock():
            start_time = time.perf_counter()
            for _ in range(20):
                # update our target
                time_step = min(int(data.time * step), 1999) # since this is not cyclic

                for i, (joint_names, motor_names) in enumerate(zip(robot.ALL_JOINTS, robot.ALL_MOTORS)):
                    # target_angles for one leg
                    angle_trajectory = np.deg2rad(leg_ik(robot.L1, robot.L2, 
                                                       position_trajectories[i][0][time_step],
                                                       position_trajectories[i][1][time_step],
                                                       position_trajectories[i][2][time_step]))

                    for j, (joint, motor) in enumerate(zip(joint_names, motor_names)):
                        angle = data.joint(joint).qpos[0]
                        velocity = data.joint(joint).qvel[0]

                        torque = (
                            kp[j] * (angle_trajectory[j] - angle)
                            - kd[j] * velocity
                        )

                        data.actuator(motor).ctrl[0] = torque

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

joints = np.array(robot.ALL_JOINTS).flatten()

joint_to_angle = {}
for joint_name in joints:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_address = model.jnt_qposadr[joint_id]
    joint_angle = data.qpos[qpos_address]

    joint_to_angle[joint_name] = joint_angle

print(joint_to_angle)