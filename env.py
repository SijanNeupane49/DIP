import gym
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

from math import cos, sin
from scipy import signal
from PIL import Image
from io import BytesIO
from gym import spaces



#import pygame




# Enable interactive mode
plt.ion()

class DoubleInvertedPendulumCartEnv(gym.Env):
    def __init__(self):
        self.max_cart_position = 2.4
        self.max_cart_velocity = 2
        self.max_angle = np.pi/15
        self.max_angular_velocity = np.pi
        self.cart_mass = 1.0
        self.pendulum_mass_1 = 0.1
        self.pendulum_mass_2 = 0.01
        self.pendulum_length_1 = 1
        self.pendulum_length_2 = 0.5
        self.gravity = 9.81
        self.inertia_1 = self.pendulum_mass_1 * (self.pendulum_length_1 ** 2 + 0.25 * self.cart_mass * (2 * self.pendulum_length_1) ** 2)
        self.inertia_2 = self.pendulum_mass_2 * (self.pendulum_length_2 ** 2 + 0.25 * self.cart_mass * (2 * self.pendulum_length_2) ** 2)
        self.delta_t = 0.01
        self.reference_velocity = 0.0

        self.x_threshold = self.max_cart_position * 1.2
        
        self.viewer = None
        self.scale = 100  # Scale factor for converting coordinates to pixels

        self.state = None




        self.cart_position = None
        self.cart_velocity = None
        self.angle_1 = None
        self.angle_2 = None
        self.angular_velocity_1 = None
        self.angular_velocity_2 = None

        high = np.array([self.max_cart_position, self.max_cart_velocity, self.max_angle, self.max_angle, self.max_angular_velocity, self.max_angular_velocity])
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

    def reset(self):
        self.cart_position = np.random.uniform(low=-0.05, high=0.05)
        self.cart_velocity = np.random.uniform(low=-0.05, high=0.05)
        self.angle_1 = np.random.uniform(low=-0.1, high=0.1)
        self.angle_2 = np.random.uniform(low=-0.1, high=0.1)
        self.angular_velocity_1 = np.random.uniform(low=-0.1, high=0.1)
        self.angular_velocity_2 = np.random.uniform(low=-0.1, high=0.1)
        return self._get_observation()

    def step(self, action):
        force = action[0]
        self.cart_position, self.cart_velocity, self.angle_1, self.angle_2, self.angular_velocity_1, self.angular_velocity_2 = self._update_state(force)
        observation = self._get_observation()
        reward = self._get_reward()
        done = self._get_done()
        info = {}
        return observation, reward, done, info

    def _update_state(self, force):
        friction = -0.1 * np.sign(self.cart_velocity)
        cart_acceleration = (force + friction) / (self.cart_mass + self.pendulum_mass_1 + self.pendulum_mass_2)
        new_cart_velocity = self.cart_velocity + cart_acceleration * self.delta_t

        # Calculate inertia forces
        inertia_1_force = self.inertia_1 * self.angular_velocity_1 ** 2 * np.sin(self.angle_1) / (self.pendulum_length_1 * 2)
        inertia_2_force = self.inertia_2 * self.angular_velocity_2 ** 2 * np.sin(self.angle_2) / (self.pendulum_length_2 * 2)

        new_cart_velocity += (inertia_1_force + inertia_2_force) / (self.cart_mass + self.pendulum_mass_1 + self.pendulum_mass_2) * self.delta_t

        new_cart_position = self.cart_position + new_cart_velocity * self.delta_t
        new_angle_1, new_angle_2, new_angular_velocity_1, new_angular_velocity_2 = self._get_new_angles_and_angular_velocities()
        return new_cart_position, new_cart_velocity, new_angle_1, new_angle_2, new_angular_velocity_1, new_angular_velocity_2

    def _get_new_angles_and_angular_velocities(self):
        torque_1, torque_2 = self._get_torques()
        new_angular_velocity_1 = self.angular_velocity_1 + torque_1 / self.inertia_1 * self.delta_t
        new_angular_velocity_2 = self.angular_velocity_2 + torque_2 / self.inertia_2 * self.delta_t
        new_angle_1 = self.angle_1 + new_angular_velocity_1 * self.delta_t
        new_angle_2 = self.angle_2 + new_angular_velocity_2 * self.delta_t
        return new_angle_1, new_angle_2, new_angular_velocity_1, new_angular_velocity_2

    def _get_torques(self):
        k1, k2 = self._get_gain_matrix()
        phi_1 = self.angle_1 - np.pi
        phi_2 = self.angle_2 - np.pi
        phi_dot_1 = self.angular_velocity_1
        phi_dot_2 = self.angular_velocity_2
        phi = np.array([phi_1, phi_2, phi_dot_1, phi_dot_2])
        A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [k1, 0, 0, 0], [k2, k1, 0, 0]])
        B = np.array([0, 0, -1, -1])
        torques = np.linalg.solve(A, B)
        torque_1, torque_2 = torques[0], torques[1]
        return torque_1, torque_2

    def _get_gain_matrix(self):
        s = signal.StateSpace(self._get_A_matrix(), self._get_B_matrix(), self._get_C_matrix(), self._get_D_matrix())
        k = signal.place_poles(s.A, s.B, np.array([-5, -5.1, -5.2, -5.3]))
        k1, k2 = k.gain_matrix[0][0], k.gain_matrix[0][1]
        return k1, k2

    def _get_A_matrix(self):
        A = np.zeros((4, 4))
        A[0][2] = 1
        A[1][3] = 1
        A[2][0] = self.gravity * (self.pendulum_mass_1 + self.pendulum_mass_2) / self.inertia_1
        A[2][1] = -self.pendulum_mass_2 * self.gravity / self.inertia_1
        A[2][2] = -self.cart_mass * self.gravity / self.inertia_1
        A[3][0] = -self.pendulum_mass_2 * self.gravity / self.inertia_2
        A[3][1] = (self.pendulum_mass_1 + self.pendulum_mass_2) * self.gravity / self.inertia_2
        A[3][3] = -self.cart_mass * self.gravity / self.inertia_2
        return A

    def _get_B_matrix(self):
        B = np.zeros((4, 1))
        B[2][0] = -1 / self.inertia_1
        B[3][0] = 1 / self.inertia_2
        return B

    def _get_C_matrix(self):
        C = np.zeros((2, 4))
        C[0][0] = 1
        C[1][1] = 1
        return C

    def _get_D_matrix(self):
        return np.zeros((2, 1))

    def _get_observation(self):
        observation = np.array([self.cart_position, self.cart_velocity, self.angle_1, self.angle_2, self.angular_velocity_1, self.angular_velocity_2])
        return observation

    def _get_reward(self):
        reward = 0.0
        if abs(self.cart_position) < 0.1 and abs(self.cart_velocity) < 0.1:
            reward += 100.0
        reward -= 0.1 * ((self.cart_position / self.max_cart_position) ** 2 + (self.cart_velocity / self.max_cart_velocity) ** 2)
        reward -= 0.1 * ((self.angle_1 / self.max_angle) ** 2 + (self.angle_2 / self.max_angle) ** 2)
        reward -= 0.1 * ((self.angular_velocity_1 / self.max_angular_velocity) ** 2 + (self.angular_velocity_2 / self.max_angular_velocity) ** 2)
        return reward

    def _get_done(self):
        if abs(self.cart_position) > self.max_cart_position or abs(self.angle_1) > self.max_angle or abs(self.angle_2) > self.max_angle:
            return True
        else:
            return False
    
    # def render(self): #rendering with pygame
    #     if self.viewer is None:
    #         pygame.init()
    #         self.viewer = pygame.display.set_mode((800, 600))
    #         pygame.display.set_caption("Double Inverted Pendulum")

    #     cart_x = int(self.cart_position * self.scale) + 400
    #     cart_y = 300

    #     pendulum_1_x = int(cart_x + self.pendulum_length_1 * np.sin(self.angle_1) * self.scale)
    #     pendulum_1_y = int(cart_y - self.pendulum_length_1 * np.cos(self.angle_1) * self.scale)

    #     pendulum_2_x = int(pendulum_1_x + self.pendulum_length_2 * np.sin(self.angle_2) * self.scale)
    #     pendulum_2_y = int(pendulum_1_y - self.pendulum_length_2 * np.cos(self.angle_2) * self.scale)

    #     self.viewer.fill((255, 255, 255))  # Fill the background with white color

    #     pygame.draw.line(self.viewer, (255, 0, 0), (cart_x, cart_y), (pendulum_1_x, pendulum_1_y), 2)
    #     pygame.draw.line(self.viewer, (0, 0, 255), (pendulum_1_x, pendulum_1_y), (pendulum_2_x, pendulum_2_y), 2)

    #     pygame.draw.circle(self.viewer, (0, 0, 0), (cart_x, cart_y), 10)
    #     pygame.draw.circle(self.viewer, (255, 0, 0), (pendulum_1_x, pendulum_1_y), 10)
    #     pygame.draw.circle(self.viewer, (0, 0, 255), (pendulum_2_x, pendulum_2_y), 10)

    #     pygame.display.flip()

    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             self.close()
    #             sys.exit()

    # def close(self):
    #     if self.viewer is not None:
    #         pygame.quit()
    #         self.viewer = None
    
    def render(self, return_image=False):#rendering using matplotlib
        plt.cla()
        plt.xlim(-self.max_cart_position, self.max_cart_position)
        plt.ylim(-3, 3)

        cart_x = self.cart_position
        cart_y = 0

        pendulum_1_x = cart_x + self.pendulum_length_1 * np.sin(self.angle_1)
        pendulum_1_y = cart_y - self.pendulum_length_1 * np.cos(self.angle_1)

        pendulum_2_x = pendulum_1_x + self.pendulum_length_2 * np.sin(self.angle_2)
        pendulum_2_y = pendulum_1_y - self.pendulum_length_2 * np.cos(self.angle_2)

        plt.plot([cart_x, pendulum_1_x], [cart_y, pendulum_1_y], 'r-')
        plt.plot([pendulum_1_x, pendulum_2_x], [pendulum_1_y, pendulum_2_y], 'b-')
        plt.plot(cart_x, cart_y, 'ko', markersize=10)
        plt.plot(pendulum_1_x, pendulum_1_y, 'ro', markersize=10)
        plt.plot(pendulum_2_x, pendulum_2_y, 'bo', markersize=10)

        plt.pause(0.01)
        plt.draw()

        if return_image:
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            return img
    def close(self):
        plt.close()
    
    # def render(self, mode='human'): #rendering using gym.env
    #     screen_width = 600
    #     screen_height = 400

    #     world_width = self.x_threshold * 2
    #     scale = screen_width /world_width
    #     carty = 100  # TOP OF CART
    #     polewidth = 10.0
    #     polelen = scale * 1.0
    #     cartwidth = 50.0
    #     cartheight = 30.0

    #     if self.viewer is None:
    #         from gym.envs.classic_control import rendering
    #         self.viewer = rendering.Viewer(screen_width, screen_height)
    #         #cart
    #         l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
    #         axleoffset = cartheight / 4.0
    #         cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    #         self.carttrans = rendering.Transform()
    #         cart.add_attr(self.carttrans)
    #         self.viewer.add_geom(cart)
    #         #pole 1
    #         l, r, t, b = -polewidth / 2, polewidth / 2, polelen-polewidth / 2, -polewidth / 2
    #         pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    #         pole.set_color(.8, .6, .4)
    #         self.poletrans = rendering.Transform(translation=(0, axleoffset))
    #         pole.add_attr(self.poletrans)
    #         pole.add_attr(self.carttrans)
    #         self.viewer.add_geom(pole)
    #         self.axle = rendering.make_circle(polewidth / 2)
    #         self.axle.add_attr(self.poletrans)
    #         self.axle.add_attr(self.carttrans)
    #         self.axle.set_color(.5, .5, .8)
    #         self.viewer.add_geom(self.axle)
    #         # pole2
    #         l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
    #         pole2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    #         pole2.set_color(.2, .8, .2)
    #         self.poletrans2 = rendering.Transform(translation=(0, polelen))
    #         pole2.add_attr(self.poletrans2)
    #         pole2.add_attr(self.poletrans)
    #         pole2.add_attr(self.carttrans)
    #         self.viewer.add_geom(pole2)
    #         self.axle2 = rendering.make_circle(polewidth / 2)
    #         self.axle2.add_attr(self.poletrans2)
    #         self.axle2.add_attr(self.poletrans)
    #         self.axle2.add_attr(self.carttrans)
    #         self.axle2.set_color(.5, .5, .8)
    #         self.viewer.add_geom(self.axle2)

    #         self.track = rendering.Line((0, carty), (screen_width, carty))
    #         self.track.set_color(0, 0, 0)
    #         self.viewer.add_geom(self.track)

    #     if self.state is None:
    #         return None

    #     x = self.state
    #     cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
    #     self.carttrans.set_translation(cartx, carty)
    #     self.poletrans.set_rotation(-x[2])
    #     self.poletrans.set_rotation(-x[2])

    #     return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    # def close(self):
    #     if self.viewer:
    #         self.viewer.close()
    #         self.viewer = None
    
#     def render(self, mode='human'):
#         if self.viewer is None:
#             self.viewer = plt.figure()

#         plt.clf()
#         plt.xlim(-self.x_threshold, self.x_threshold)
#         plt.ylim(-1.5, 1.5)

#         xt = self._get_observation()
#         image = self.viewer.gca()

#         image = draw(xt, image)

#         plt.draw()
#         plt.pause(0.001)

#         if mode == 'rgb_array':
#             buf = BytesIO()
#             self.viewer.savefig(buf, format='png')
#             buf.seek(0)
#             img_array = np.array(Image.open(buf))
#             buf.close()
#             return img_array

# def draw(xt, image):
#     x = xt[0]
#     phi1 = xt[2]
#     phi2 = xt[4]

#     cart_width = 0.4
#     cart_height = 0.2
#     pendulum_length1 = 1.0
#     pendulum_length2 = 1.0
#     pendulum_width = 0.1

#     cart_top_y = cart_height / 2.0
#     pendulum1_base = np.array([x, cart_top_y])
#     pendulum1_tip = pendulum1_base + np.array([-pendulum_length1 * np.sin(phi1), -pendulum_length1 * np.cos(phi1)])
#     pendulum2_base = pendulum1_tip
#     pendulum2_tip = pendulum2_base + np.array([-pendulum_length2 * np.sin(phi2), -pendulum_length2 * np.cos(phi2)])

#     # Draw cart
#     cart = patches.Rectangle((x - cart_width / 2.0, 0), cart_width, cart_height, fc='blue')
#     image.add_patch(cart)

#     # Draw pendulum1
#     pendulum1 = lines.Line2D([pendulum1_base[0], pendulum1_tip[0]], [pendulum1_base[1], pendulum1_tip[1]], lw=pendulum_width, color='red')
#     image.add_line(pendulum1)

#     # Draw pendulum1 mass
#     pendulum1_mass = patches.Circle(pendulum1_tip, radius=0.1, fc='red')
#     image.add_patch(pendulum1_mass)

#     # Draw pendulum2
#     pendulum2 = lines.Line2D([pendulum2_base[0], pendulum2_tip[0]], [pendulum2_base[1], pendulum2_tip[1]], lw=pendulum_width, color='green')
#     image.add_line(pendulum2)

#     # Draw pendulum2 mass
#     pendulum2_mass = patches.Circle(pendulum2_tip, radius=0.1, fc='green')
#     image.add_patch(pendulum2_mass)

#     return image





    


