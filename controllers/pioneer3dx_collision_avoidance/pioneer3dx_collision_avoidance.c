#include <webots/robot.h>
#include <webots/motor.h>

#define TIME_STEP 32
#define DURATION 200  // 200 * 32 ms = 6.4 secondes

int main() {
  wb_robot_init();

  WbDeviceTag left_motor = wb_robot_get_device("left wheel");
  WbDeviceTag right_motor = wb_robot_get_device("right wheel");

  wb_motor_set_position(left_motor, INFINITY);
  wb_motor_set_position(right_motor, INFINITY);

  double speed = 3.0;
  int counter = 0;

  while (wb_robot_step(TIME_STEP) != -1) {
    if (counter >= DURATION) {
      speed = -speed;
      counter = 0;
    }

    wb_motor_set_velocity(left_motor, speed);
    wb_motor_set_velocity(right_motor, speed);
    counter++;
  }

  wb_robot_cleanup();
  return 0;
}
