import numpy as np


class Controller:
    def __init__(self, angle_gain=0):
        # Controller gain parameters
        self.angle_gain = angle_gain

    def angle_control_commands(self, dist, angle, pid_type='p'):
        # Return the angular velocity in order to control the Duckiebot so that it follows the lane.
        # Parameters:
        #     dist: distance from the center of the lane. Left is negative, right is positive.
        #     angle: angle from the lane direction, in rad. Left is negative, right is positive.
        # Outputs:
        #     omega: angular velocity, in rad/sec. Right is negative, left is positive.
        
        omega = 0. 
        
        #######
        #
        # MODIFY ANGULAR VELOCITY
        #
        # YOUR CODE HERE
        #
        #######

        # Restrict angle and distance
        dist_threshold = np.clip(dist, -np.abs(np.pi / (3 * self.angle_gain)), np.abs(np.pi / (3 * self.angle_gain)))
        angle_threshold = np.clip(angle, -np.pi / 6, np.pi / 6)

        # Add terms to control depending on type
        if 'p' in pid_type:
            omega += self.angle_gain * angle_threshold
        if 'd' in pid_type:
            omega += ((self.angle_gain ** 2) / 2) * dist_threshold

        return omega

    def pure_pursuit(self, env, pos, angle, v, follow_dist=0.25):
        # Return the angular velocity in order to control the Duckiebot using a pure pursuit algorithm.
        # Parameters:
        #     env: Duckietown simulator
        #     pos: global position of the Duckiebot
        #     angle: global angle of the Duckiebot
        # Outputs:
        #     v: linear veloicy in m/s.
        #     omega: angular velocity, in rad/sec. Right is negative, left is positive.
        
        closest_curve_point = env.unwrapped.closest_curve_point
        
        # Find the curve point closest to the agent, and the tangent at that point
        closest_point, closest_tangent = closest_curve_point(pos, angle)

        iterations = 0

        lookup_distance = follow_dist
        multiplier = 0.5
        curve_point = None

        while iterations < 10:
            ########
            #
            # TODO 1: Modify follow_point so that it is a function of closest_point, closest_tangent, and lookup_distance
            #
            ########
            follow_point = closest_point + (lookup_distance * closest_tangent)

            curve_point, _ = closest_curve_point(follow_point, angle)

            # If we have a valid point on the curve, stop
            if curve_point is not None:
                break

            iterations += 1
            lookup_distance *= multiplier

        # Agent angle pointing opposed to driving direction
        agent_angle = angle + 2.8942716086132663

        # Invert x axis to make agent look towards the curves
        cos_agent_angle = -np.cos(agent_angle)
        sin_agent_angle = np.sin(agent_angle)

        # Transform goal to agent coordinates
        t = np.linalg.inv(np.array([[cos_agent_angle, -sin_agent_angle, pos[0]],
                                   [sin_agent_angle, cos_agent_angle, pos[-1]],
                                   [0, 0, 1]]))
        goal = np.matmul(t, np.array([[curve_point[0]], [curve_point[-1]], [1]]))[:-1]

        # Calculate angular speed and invert to follow convention
        omega = -2 * v * goal[0][0] / np.linalg.norm(goal) ** 2

        return v, omega
