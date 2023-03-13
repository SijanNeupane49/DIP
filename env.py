import gym
import numpy as np
from scipy import signal

class DoubleInvertedPendulumCartEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.gravity = 9.81
        self.mass_cart = 1.0
        self.mass_pendulum1 = 0.1
        self.mass_pendulum2 = 0.05
        self.length_pendulum1 = 0.5
        self.length_pendulum2 = 0.25
        self.total_mass = self.mass_cart + self.mass_pendulum1 + self.mass_pendulum2
        self.m1_p1 = self.mass_pendulum1 * self.length_pendulum1
        self.m2_p2 = self.mass_pendulum2 * self.length_pendulum2
        self.dt = 0.02  # seconds between state updates

        # State variables
        self.cart_position = 0.0
        self.cart_velocity = 0.0
        self.theta1 = 0.0
        self.theta1_dot = 0.0
        self.theta2 = 0.0
        self.theta2_dot = 0.0

        # Action space
        self.action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(1,))

        # Observation space
        self.observation_space = gym.spaces.Box(low=np.array([-2.4, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf]),
                                                high=np.array([2.4, np.inf, np.pi, np.inf, np.pi, np.inf]), dtype=np.float32)

        # Initialize state space matrices
        A = np.array([[0, 1, 0, 0, 0, 0],
                      [0, 0, -self.m1_p1*self.gravity/self.total_mass, 0, -self.m2_p2*self.gravity/self.total_mass, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, -(self.total_mass+self.m1_p1)*self.gravity/(self.length_pendulum1*self.total_mass), 0,
                       -(self.m2_p2*self.gravity*self.length_pendulum2)/(self.length_pendulum1*self.total_mass), 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, -(self.m1_p1+self.m2_p2)*self.gravity/(self.length_pendulum1*self.total_mass), 0,
                       -(self.gravity*self.total_mass*self.length_pendulum2)/(self.length_pendulum1*self.total_mass), 0]])

        B = np.array([[0], [self.m1_p1/self.total_mass], [0], [(self.total_mass+self.m1_p1)/(self.length_pendulum1*self.total_mass)],
                      [0], [(self.m1_p1+self.m2_p2)/(self.length_pendulum1*self.total_mass)]])
        
        

        # Convert to state space format
        self.sys = signal.StateSpace(A, B, np.eye(6), np.zeros((6,1)))

        self.viewer = None

    def get_state_space_model(self):
        A = np.array([[0, 1, 0, 0, 0, 0],
                      [0, 0, -self.m1_p1*self.gravity/self.total_mass, 0, -self.m2_p2*self.gravity/self.total_mass, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, -(self.total_mass+self.m1_p1)*self.gravity/(self.length_pendulum1*self.total_mass), 0,
                       -(self.m2_p2*self.gravity*self.length_pendulum2)/(self.length_pendulum1*self.total_mass), 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, -(self.m1_p1+self.m2_p2)*self.gravity/(self.length_pendulum1*self.total_mass), 0,
                       -(self.gravity*self.total_mass*self.length_pendulum2)/(self.length_pendulum1*self.total_mass), 0]])

        B = np.array([[0], [self.m1_p1/self.total_mass], [0], [(self.total_mass+self.m1_p1)/(self.length_pendulum1*self.total_mass)],
                      [0], [(self.m1_p1+self.m2_p2)/(self.length_pendulum1*self.total_mass)]])
        
        

        # Convert to state space format
        #A = signal.StateSpace(A)
        #B = signal.StateSpace(B)
        C = np.eye(6)#signal.StateSpace(np.eye(6))
        D = np.zeros((6,1))#signal.StateSpace(np.zeros((6,1)))

        #self.sys = signal.StateSpace(A, B, np.eye(6), np.zeros((6,1))) 'original line'
        ss = signal.StateSpace(A, B, C, D)
        
        return A, B, C, D

        print(A)

        #print(abcd)

        #return abcd





    def step(self, action):
        # Clamp the action to the valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)[0]

        # Compute the reference
        x_ref = np.array([self.cart_position, self.cart_velocity, self.theta1, self.theta1_dot, self.theta2, self.theta2_dot])
        x_ref_dot = np.array([0, action, 0, 0, 0, 0])

        # Compute the force
        Kp = np.array([-100, -10, -300, -10, -300, -10])
        Kd = np.array([-20, -2, -30, -2, -30, -2])
        force = np.dot(Kp, (x_ref - self.state)) + np.dot(Kd, (x_ref_dot - self.derivatives))

        # Simulate the system
        self.cart_velocity += force / self.total_mass * self.dt
        self.cart_position += self.cart_velocity * self.dt

        q = [self.theta1, self.theta2]
        q_dot = [self.theta1_dot, self.theta2_dot]
        q_ddot = np.linalg.inv(self.sys.B) * (force - np.dot(self.sys.A, [self.cart_position, self.cart_velocity] + q + q_dot))
        self.theta1_dot += q_ddot[3] * self.dt
        self.theta2_dot += q_ddot[5] * self.dt
        self.theta1 += self.theta1_dot * self.dt
        self.theta2 += self.theta2_dot * self.dt

        # Update state and derivatives
        self.state = np.array([self.cart_position, self.cart_velocity, self.theta1, self.theta1_dot, self.theta2, self.theta2_dot])
        self.derivatives = np.array([0, action, 0, 0, 0, 0])

        # Compute the reward and check if the episode is over
        done = abs(self.cart_position) > 2.4 or abs(self.theta1) > 12 * np.pi / 180 or abs(self.theta2) > 12 * np.pi / 180
        reward = -1000 if done else 1.0

        return self.state, reward, done, {}


    def reset(self):
        self.cart_position = 0
        self.cart_velocity = 0
        self.theta1 = np.pi
        self.theta2 = 0
        self.theta1_dot = 0
        self.theta2_dot = 0
        self.state = np.array([self.cart_position, self.cart_velocity, self.theta1, self.theta1_dot, self.theta2, self.theta2_dot])
        self.derivatives = np.zeros_like(self.state)
        return self.state
    
    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = 4.8
        scale = screen_width / world_width
        cart_y = screen_height / 2  # Top of cart
        pole_width = 10.0
        pole_len = scale * 1.0
        cart_width = 50.0
        cart_height = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cart_width / 2, cart_width / 2, cart_height / 2, -cart_height / 2
            axleoffset = cart_height / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -pole_width / 2, pole_width / 2, pole_len - pole_width / 2, -pole_width / 2
            pole1 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole1.set_color(0.8, 0.6, 0.4)
            pole2.set_color(0.5, 0.3, 0.1)
            self.poletrans1 = rendering.Transform(translation=(0, axleoffset))
            self.poletrans2 = rendering.Transform(translation=(0, axleoffset))
            pole1.add_attr(self.poletrans1)
            pole2.add_attr(self.poletrans2)
            self.viewer.add_geom(pole1)
            self.viewer.add_geom(pole2)
            self.axle = rendering.make_circle(pole_width / 2)
            self.axle.add_attr(self.poletrans1)
            self.axle.add_attr(self.poletrans2)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)

        if self.state is None:
            return None

        cartx = self.state[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, cart_y)
        self.poletrans1.set_rotation(-self.state[2])
        self.poletrans2.set_rotation(-self.state[4])
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


