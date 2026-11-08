# joints
THETA1_JOINTS = (
    "front_left_theta1",
    "front_right_theta1",
    "rear_left_theta1",
    "rear_right_theta1",
)
THETA2_JOINTS = (
    "front_left_theta2",
    "front_right_theta2",
    "rear_left_theta2",
    "rear_right_theta2",
)
THETA3_JOINTS = (
    "front_left_theta3",
    "front_right_theta3",
    "rear_left_theta3",
    "rear_right_theta3",
)

# Tuple containing 4-tuples for each leg: (FL, FR, RL, RR)
ALL_JOINTS = (
    ("front_left_theta1", "front_left_theta2", "front_left_theta3"),
    ("front_right_theta1", "front_right_theta2", "front_right_theta3"),
    ("rear_left_theta1", "rear_left_theta2", "rear_left_theta3"),
    ("rear_right_theta1", "rear_right_theta2", "rear_right_theta3"),
)

# motors
MOTOR1_LIST = (
    "front_left_motor1",
    "front_right_motor1",
    "rear_left_motor1",
    "rear_right_motor1",
)
MOTOR2_LIST = (
    "front_left_motor2",
    "front_right_motor2",
    "rear_left_motor2",
    "rear_right_motor2",
)
MOTOR3_LIST = (
    "front_left_motor3",
    "front_right_motor3",
    "rear_left_motor3",
    "rear_right_motor3",
)

# Tuple containing 4-tuples for each leg: (FL, FR, RL, RR)
ALL_MOTORS = (
    ("front_left_motor1", "front_left_motor2", "front_left_motor3"),
    ("front_right_motor1", "front_right_motor2", "front_right_motor3"),
    ("rear_left_motor1", "rear_left_motor2", "rear_left_motor3"),
    ("rear_right_motor1", "rear_right_motor2", "rear_right_motor3"),
)

L1 = 0.1
L2 = 0.1

# Nominal standing joint angles (radians)
STANDING_ANGLES = {
    "front_left_theta1": 0.0,
    "front_left_theta2": -0.6167622816246616,
    "front_left_theta3": 1.337089161190517,

    "front_right_theta1": 0.0,
    "front_right_theta2": -0.6167622816246616,
    "front_right_theta3": 1.337089161190517,

    "rear_left_theta1": 0.0,
    "rear_left_theta2": -0.6037718322490597,
    "rear_left_theta3": 1.3872085844435758,

    "rear_right_theta1": 0.0,
    "rear_right_theta2": -0.6037718322490597,
    "rear_right_theta3": 1.3872085844435758,
}