from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Accès aux moteurs
left_motor = robot.getDevice('left_rear_wheel_joint')
right_motor = robot.getDevice('right_rear_wheel_joint')

# Mode de contrôle : vitesse
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Vitesse initiale
left_motor.setVelocity(3.0)
right_motor.setVelocity(3.0)

while robot.step(timestep) != -1:
    # Tu peux ajouter ici une IA, détection de panneaux, etc.
    pass
