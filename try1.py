import numpy as np
import gym
from scipy import signal

def abcd():
    gravity = 9.81
    mass_cart = 1.0
    mass_pendulum1 = 0.1
    mass_pendulum2 = 0.05
    length_pendulum1 = 0.5
    length_pendulum2 = 0.25
    total_mass = mass_cart + mass_pendulum1 + mass_pendulum2
    m1_p1 = mass_pendulum1 * length_pendulum1
    m2_p2 = mass_pendulum2 * length_pendulum2
    dt = 0.02  # seconds between state updates

    # State variables
    cart_position = 0.0
    cart_velocity = 0.0
    theta1 = 0.0
    theta1_dot = 0.0
    theta2 = 0.0
    theta2_dot = 0.0

    # Action space
    action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(1,))

    # Observation space
    observation_space = gym.spaces.Box(low=np.array([-2.4, -np.inf, -np.pi, -np.inf, -np.pi, -np.inf]),
                                            high=np.array([2.4, np.inf, np.pi, np.inf, np.pi, np.inf]), dtype=np.float32)

    # Initialize state space matrices
    A = np.array([[0, 1, 0, 0, 0, 0],
                    [0, 0, -m1_p1*gravity/total_mass, 0, -m2_p2*gravity/total_mass, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, -(total_mass+m1_p1)*gravity/(length_pendulum1*total_mass), 0,
                    -(m2_p2*gravity*length_pendulum2)/(length_pendulum1*total_mass), 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, -(m1_p1+m2_p2)*gravity/(length_pendulum1*total_mass), 0,
                    -(gravity*total_mass*length_pendulum2)/(length_pendulum1*total_mass), 0]])

    B = np.array([[0], [m1_p1/total_mass], [0], [(total_mass+m1_p1)/(length_pendulum1*total_mass)],
                    [0], [(m1_p1+m2_p2)/(length_pendulum1*total_mass)]])
    

   
    


    # Convert to state space format

    C = np.eye(6)
    D =np.zeros((6,1))
   
    A = signal.StateSpace(A)
    B = signal.StateSpace(B)
    C = signal.StateSpace(C)
    D = signal.StateSpace(D)
    

   
    
    return A, B, C, D
