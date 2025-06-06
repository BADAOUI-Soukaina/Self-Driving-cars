from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

distance = 0.0
speed = 3.0
direction = 1

while robot.step(timestep) != -1:
    distance += 0.01 * direction

    if abs(distance) >= 2.0:
        direction *= -1
        distance = 0.0
        left_motor.setVelocity(-speed * direction)
        right_motor.setVelocity(speed * direction)
    else:
        left_motor.setVelocity(speed * direction)
        right_motor.setVelocity(speed * direction)
