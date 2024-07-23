import logging
import sys, math
from collections import namedtuple
import numpy as np
import pandas as pd
import pygame
import random
import lzma
import os

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import colorize, seeding

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] In %(pathname)s:%(lineno)d:\n%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

if __name__ == "__main__":
    # Relative path durving development
    DATA_DIRECTORY = "../data/2023_2/"
else:
    # Relative path when training
    DATA_DIRECTORY = "data/2023_2/"

VIDEO_FPS = 60     # Source data FPS (60Hz)
SIM_FPS = 30    # Simulation environment FPS

INITIAL_RANDOM = 5

SCREEN_W = 900
SCREEN_H = 900
BOUNDARY_SCALE = 0.02

vec2d = namedtuple('vec2d', ['x', 'y'])

# Global parameters for agent control
TIMESTEP = 1./SIM_FPS       # Not sure if this will be necessary, given the fixed FPS?
# TIME_LIMIT = SIM_FPS * 60   # 60 seconds
TIME_LIMIT = SIM_FPS * 30   # 60 seconds

ANT_DIM = vec2d(5, 5)
AGENT_SPEED = 25*3.25       # Taken from slimevolley, will need to adjust based on feeling
TURN_RATE = 200 * 2 * math.pi / 360
VISION_RANGE = 100  # No idea what is a reasonable value for this.

DRAW_ANT_VISION = True
vision_segments = [
    # Front arc: Directly in front of the agent
    ((-math.pi / 2, -3*math.pi / 10), (180, 100, 100)),
    ((-3*math.pi / 10, -math.pi / 10), (180, 180, 190)),
    ((-math.pi / 10, math.pi / 10), (100, 180, 100)),
    ((math.pi / 10, 3*math.pi / 10), (100, 180, 100)),
    ((3*math.pi / 10, math.pi / 2), (180, 180, 190)),
    # Right arc: To the right of the agent
    ((-9 * math.pi / 6, -7 * math.pi / 6), (180, 100, 100)),
    # Back arc: Directly behind the agent
    ((-7 * math.pi / 6, -5 * math.pi / 6), (180, 180, 190)),
    # Left arc: To the left of the agent
    ((-5 * math.pi / 6, -math.pi / 2), (180, 180, 190)),
]

REWARD_TYPE = 'trail' # 'trail', 'action'
TRACK_TRAIL = 'all' # 'all', 'fade', 'none'
MOVEMENT_THRESHOLD = 10
FADE_DURATION = 5 # seconds

##########################################
#          Assistance functions          #
##########################################

FILE_PREFIX = "KA050_10cm_5h_20230614"
PP_FILE_PREFIX = "KA050_processed"
OUTPUT_FILE = '_'.join([PP_FILE_PREFIX, *FILE_PREFIX.split('_')[1:]]) + '.pkl.xz'

def load_data(source_dir, scale = None, arena_dim = None):
    data = None
    if os.path.exists(os.path.join(source_dir, OUTPUT_FILE)):
        with lzma.open(os.path.join(source_dir, OUTPUT_FILE)) as file:
            data = pd.read_pickle(file)
            logger.info(msg=f"Processed data file found: {OUTPUT_FILE}")
        return data.iloc[::int(scale)] if scale else data
    else:
        logger.info(msg=f"No processed file found. Looking for ")
        return load_combined_files(source_dir, arena_dim, scale)


def process_data(data, arena_dim):
    data_len = len(data)
    logger.info(msg=f"Ant trail data loaded. Total records: {data_len}")
    arena_bb = find_bounding_box(data)
    origin_arena = calculate_circle(*arena_bb)

    translation, scale = circle_transformation(origin_arena, arena_dim)

    logger.info(msg=f"Processing data now. This will take a while...")
    apply_transform_scale(data, translation, scale)
    logger.info(msg=f"Finished processing.")

    logger.info(msg=f"Translation: {translation}, scale: {scale}")
    logger.info(msg=f"Original: ({origin_arena[0][0] + translation[0]}, {origin_arena[0][1] + translation[1]}), scale: {origin_arena[1]*scale}")
    logger.info(msg=f"Simulated: {arena_dim[0]}, scale: {arena_dim[1]}")

    return data


def load_combined_files(source_dir, arena_dim, scale = None):
    input_files = []
    data = []

    for file in os.listdir(source_dir):
        if FILE_PREFIX in file and file.endswith('.pkl.xz'):
            input_files.append(file)

    for input_file in input_files:
        with lzma.open(os.path.join(source_dir, input_file)) as file:
            data.append(pd.read_pickle(file))

    data = process_data(pd.concat(data, ignore_index=True), arena_dim)
    data.to_pickle(os.path.join(source_dir, OUTPUT_FILE), compression='xz')

    return data.iloc[::int(scale)] if scale else data


def find_bounding_box(data):
    # Separate all x and y values into slices
    all_x_values = data[[col for col in data.columns if 'x' in col]]
    all_y_values = data[[col for col in data.columns if 'y' in col]]
    # Calculating the minimum and maximum for x and y values efficiently
    min_x = all_x_values.min(axis=None)
    max_x = all_x_values.max(axis=None)
    min_y = all_y_values.min(axis=None)
    max_y = all_y_values.max(axis=None)

    return min_x, min_y, max_x, max_y


def calculate_circle(min_x, min_y, max_x, max_y):
    """
    Calculate the circle that fits perfectly within a bounding box.

    Parameters:
    min_x (float): The minimum x value of the bounding box.
    max_x (float): The maximum x value of the bounding box.
    min_y (float): The minimum y value of the bounding box.
    max_y (float): The maximum y value of the bounding box.

    Returns:
    tuple: A tuple containing the center coordinates (x, y) and the radius of the circle.
    """
    # Calculate the center of the bounding box
    x_centre = (min_x + max_x) / 2
    y_centre = (min_y + max_y) / 2

    # Calculate the radius of the circle
    radius = min(max_x - min_x, max_y - min_y) / 2

    return ((x_centre, y_centre), radius)


def circle_transformation(circle_a, circle_b):
    """
    Calculate the transformation from one circle to another.

    Parameters:
    circle_a (tuple): A tuple (x_a, y_a, r_a) representing Circle A's center and radius.
    circle_b (tuple): A tuple (x_b, y_b, r_b) representing Circle B's center and radius.

    Returns:
    tuple: A tuple containing the translation vector (dx, dy) and the scaling factor.
    """
    scale_factor = 0.99
    (x_a, y_a), r_a = circle_a
    (x_b, y_b), r_b = circle_b

    # Scaling
    scale = r_b / r_a

    x_a *= scale*scale_factor
    y_a *= scale*scale_factor

    # Translation vector
    dx = x_b - x_a
    dy = y_b - y_a

    return (dx, dy), scale*scale_factor


def apply_transform_scale(data, trans, scale):
    data[[col for col in data.columns if 'x' in col]] = data[[col for col in data.columns if 'x' in col]].transform(
        np.vectorize(lambda x : np.round((x * scale) + trans[0]))
    )
    data[[col for col in data.columns if 'y' in col]] = data[[col for col in data.columns if 'y' in col]].transform(
        np.vectorize(lambda x : np.round((x * scale) + trans[1]))
    )


def euclidean_distances(data):
    a = np.array(data)
    b = a.reshape(a.shape[0], 1, a.shape[1])
    distances = np.sqrt(np.einsum('ijk, ijk->ij', a-b, a-b))
    np.fill_diagonal(distances, np.NaN)

    return distances

def is_rectangle_in_circle(x, y, circle_center, circle_radius):
    """
    Check if a pygame.Rect is completely contained within a circle.

    Parameters:
    rect (pygame.Rect): The rectangle to check.
    circle_center (tuple): The (x, y) coordinates of the center of the circle.
    circle_radius (float): The radius of the circle.

    Returns:
    bool: True if the rectangle is completely contained within the circle, False otherwise.
    """
    rect = pygame.Rect(x - ANT_DIM.x / 2., y - ANT_DIM.y / 2.,
                       ANT_DIM.x, ANT_DIM.y)
    rect_corners = [
        (rect.left, rect.top),    # Top-left
        (rect.left, rect.bottom), # Bottom-left
        (rect.right, rect.top),   # Top-right
        (rect.right, rect.bottom) # Bottom-right
    ]

    for x, y in rect_corners:
        dx = x - circle_center[0]
        dy = y - circle_center[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance > circle_radius:
            return False

    return True


##########################################
#            Ant Environment             #
##########################################

class Ant():
    """Agent class for the ant"""

    def __init__(self, pos):
        self.pos = vec2d(*pos)
        self.speed = 0.0
        self.theta = 0.0
        self.theta_dot = 0.0
        self.trail = []

        # Detection scalar:
        # num of ants in cone, or distance to closes ant
        self.V_f_l1 = None
        self.V_f_l2 = None
        self.V_f = None
        self.V_f_r2 = None
        self.V_f_r1 = None
        self.V_b_r = None
        self.V_b = None
        self.V_b_l = None
        self.vision_range = VISION_RANGE

    def _detect_vision(self, detected_ants: dict):
        v_f_l1 = len(detected_ants['forward_l1'])
        v_f_l2 = len(detected_ants['forward_l2'])
        v_f = len(detected_ants['forward'])
        v_f_r2 = len(detected_ants['forward_r2'])
        v_f_r1 = len(detected_ants['forward_r1'])
        v_b_r = len(detected_ants['backward_r'])
        v_b = len(detected_ants['backward'])
        v_b_l = len(detected_ants['backward_l'])

        vision = [v_f_l1, v_f_l2, v_f, v_f_r2, v_f_r1, v_b_r, v_b, v_b_l]
        denominator = np.sum([v_f_l1, v_f_l2, v_f_r2, v_f_r1, v_b_r, v_b, v_b_l])
        if denominator != 0: vision /= denominator

        self.V_f_l1, self.V_f_l2, self.V_f, self.V_f_r2, self.V_f_r1,  self.V_b_r, self.V_b, self.V_b_l = vision


    def _detect_nearby_ants(self, other_ants):
        """
        Detects other ants within a specified radius and identifies their relative
        position segment based on this ant's orientation.

        Parameters:
        - other_ants (list of tuples): The (x, y) positions of other ants.

        Returns:
        - dict: A dictionary mapping segment names to a list of ants
                (represented by their positions) that are within the specified
                radius and fall into that relative segment.
        """
        # Define the segments
        detected_ants = {
            'forward_l1': [], 'forward_l2': [], 'forward': [], 'forward_r2': [], 'forward_r1': [],
            'backward_r': [], 'backward': [], 'backward_l': []
        }
        segment_boundaries = {
            'forward_l1': vision_segments[0][0],
            'forward_l2': vision_segments[1][0],
            'forward': vision_segments[2][0],
            'forward_r2': vision_segments[3][0],
            'forward_r1': vision_segments[4][0],
            'backward_r': vision_segments[5][0],
            'backward': vision_segments[6][0],
            'backward_l': vision_segments[7][0]
        }

        for other_ant in other_ants:
            dx = other_ant[0] - self.pos.x
            dy = other_ant[1] - self.pos.y
            distance = math.sqrt(dx**2 + dy**2)

            if distance <= self.vision_range:
                # Calculate angle from self.pos to other_ant.pos, adjusting with self.theta
                angle_to_ant = math.atan2(dy, dx) % (2 * np.pi)
                # Determine segment based on relative_angle
                for segment, (start_angle, stop_angle) in segment_boundaries.items():
                    # Adjust bounds to be within -pi to pi
                    start_angle = (self.theta + start_angle) % (2 * math.pi)
                    stop_angle = (self.theta + stop_angle) % (2 * math.pi)
                    if start_angle < stop_angle:
                        if start_angle <= angle_to_ant < stop_angle:
                            detected_ants[segment].append(other_ant)
                            break
                    else:  # Angle wraps around the -pi/pi boundary
                        if start_angle <= angle_to_ant or angle_to_ant > stop_angle:
                            detected_ants[segment].append(other_ant)
                            break

        return detected_ants



    def _turn(self):
        self.theta += (self.theta_dot * TIMESTEP)
        self.theta = self.theta % (2 * np.pi)


    def _move(self, arena):
        """
        Move an agent from its current position (x, y) according to desired_speed
        and angle theta using matrix multiplication.
        """
        # Calculate the desired direction of travel (rotate to angle theta)
        direction = np.array([np.cos(self.theta), np.sin(self.theta)]) * self.desired_speed * TIMESTEP
        # Set the desired position based on direction and speed relative to timestep
        desired_pos = np.add(np.array(self.pos), direction)
        # If leaving the cirle, push agent back into circle.
        if is_rectangle_in_circle(desired_pos[0], desired_pos[1], arena[0], arena[1]):
            self.pos = vec2d(desired_pos[0], desired_pos[1])
        # Otherwise, slightly adjust the agent's angle theta towards tangent at the
        # circle's circumference.
        # else:
        #     # Calculate the angle from the center of the circle to the agent
        #     angle_to_center = math.atan2(self.pos.y - arena[0][1], self.pos.x - arena[0][0])
        #     if angle_to_center < 0: angle_to_center += np.pi
        #     if (self.theta >= angle_to_center) and (self.theta < angle_to_center + np.pi):
        #         d_theta = np.pi / 2 - self.desired_turn_speed
        #     elif (self.theta < angle_to_center) or (self.theta >= angle_to_center - np.pi):
        #         d_theta = -np.pi / 2 + self.desired_turn_speed

        #     theta = self.theta + d_theta * TIMESTEP
        #     theta = theta % (2 * np.pi)
        #     self.theta = theta


    def set_action(self, action):
        forward    = False
        backward   = False
        turn_left  = False
        turn_right = False

        if action[0] > 0.25: forward    = True
        if action[1] > 0.25: backward   = True
        if action[2] > 0.25: turn_left  = True
        if action[3] > 0.25: turn_right = True

        self.desired_speed = 0
        self.desired_turn_speed = 0

        if (forward and (not backward)):
            self.desired_speed = AGENT_SPEED
        if (backward and (not forward)):
            self.desired_speed = -AGENT_SPEED
        if (turn_left and (not turn_right)):
            self.desired_turn_speed = -TURN_RATE
        if (turn_right and (not turn_left)):
            self.desired_turn_speed = TURN_RATE

        return [int(x) for x in [forward, backward, turn_left, turn_right]]

    def get_obs(self, others=None):
        if others is not None:
            self._detect_vision(self._detect_nearby_ants(others))
        result = [
            self.pos.x, self.pos.y, self.speed,
            self.theta, self.theta_dot,
            self.V_f_l1, self.V_f_l2, self.V_f, self.V_f_r2, self.V_f_r1,
            self.V_b_r, self.V_b, self.V_b_l,
        ]
        return result


    def update(self, arena, noise=0.0):
        self.pos   = vec2d(
            self.pos.x + np.random.randn() * noise,
            self.pos.y - np.random.randn() * noise
        )
        self.theta += (np.random.randn() * noise)
        self.theta = self.theta % (2 * np.pi)

        self.speed = self.desired_speed
        self.theta_dot = self.desired_turn_speed

        self._turn()
        self._move(arena)


class AntDynamicsEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps' : SIM_FPS
    }

    ant_trail_data = None

    def __init__(self, render_mode=None):
        self.force_mag = 10.0
        self.ant = None
        self.ant_trail = None

        self.target_trail = None
        self.target_data = None

        self.other_ants = None

        self.seed()
        self.viewer = None
        self.state = None
        self.noise = 0

        self.t = 0
        self.t_limit = TIME_LIMIT

        assert render_mode is None or render_mode in type(self).metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        # circular arena
        self.ant_arena = (
            (SCREEN_W/2.0, SCREEN_H/2.0),
            min(SCREEN_W, SCREEN_H)/2.0 - min(SCREEN_W, SCREEN_H) * BOUNDARY_SCALE
        )

        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max
        ])

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=float)
        self.observation_space = spaces.Box(-high, high, dtype=float)

        # Load the ant trail dataset
        if not type(self).ant_trail_data:
            self._get_ant_trails()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _get_ant_trails(self):
        type(self).ant_trail_data = load_data(
            DATA_DIRECTORY,
            VIDEO_FPS / SIM_FPS,
            self.ant_arena
        )


    def _select_target(self, others=False, trail_len=SIM_FPS*60):
        """
        Select an ant trail as the target trail for the current trial.
        At the moment, we will just select a single target trail, but we should
        also provide positions of other ants within a given radius for feeding
        into the ant's internal state.
        """
        trail_length = int(trail_len)+2
        target = np.zeros((trail_length, 2), dtype=float)
        trail_data = type(self).ant_trail_data
        num_ants = len(trail_data.columns.levels[0])
        # If showing positions of other ants during the trail
        other_ants = None
        if others:
            other_ants = np.zeros(
                (num_ants - 1, trail_length, 2),
                dtype=float
            )

        start = np.random.randint(len(trail_data) - trail_length)
        indices = list(np.random.permutation(num_ants))
        indices_set = set(indices)
        contains_null = True
        while contains_null and len(indices) > 0:
            ant_index = indices.pop()
            if np.isnan(np.array(trail_data[ant_index][start:start + trail_length])).any():
                continue
            else:
                x1 = trail_data.iloc[start][ant_index].x
                y1 = trail_data.iloc[start][ant_index].y
                x2 = trail_data.iloc[start + trail_length-1][ant_index].x
                y2 = trail_data.iloc[start + trail_length-1][ant_index].y
                x1, y1, x2, y2 = [int(x) for x in [x1, y1, x2, y2]]
                dx, dy = x2-x1, y2-y1
                # If this trail is too short to be used, continue the search.
                if (np.sqrt(dx**2 + dy**2)) < MOVEMENT_THRESHOLD:
                    continue
                target[0:trail_length] = trail_data[ant_index][start:start + trail_length]
                contains_null = False
                indices_set.discard(ant_index)
        if others and not contains_null:
            trail_index = 0
            for other_ant_index in indices_set:
                if np.isnan(np.array(trail_data[other_ant_index][start:start + trail_length])).any():
                    np.resize(other_ants, (np.shape(other_ants)[1]-1, trail_length, 2))
                    continue
                other_ants[trail_index][0:trail_length] = trail_data[other_ant_index][start:start + trail_length]
                trail_index += 1

        return Ant(target[0]), target, other_ants


    def _get_angle_from_trajectory(self, trail, start_time, interval=False):
        theta = -1
        threshold = 5
        time = start_time + 1
        while time != len(trail) and theta < 0:
            try:
                dx, dy = trail[time] - trail[start_time]
            except IndexError:
                break
            if threshold < (np.sqrt(dx**2 + dy**2)):
                theta = math.atan2(dy, dx) % (2 * np.pi)
            time += 1

        if interval: return theta, (time - start_time)
        else: return theta


    def _smallest_angle_diff(self, theta1, theta2):
        diff1 = (theta2 - theta1) % (2 * math.pi)
        diff2 = (theta1 - theta2) % (2 * math.pi)
        return min(diff1, diff2)


    def _calculate_target_data(self, trail):
        target_data = {
            'angle': [],
            'motion': [],
            'action': []
            # ACTION SET:
            # forward, forward-left, forward-right
            # backward, backward-left, backward-right
            # turn-left, turn-right
            # stop
        }
        time = 0
        prev_angle = -1
        # Get the polarity time series based on movement of the ant
        while time != len(trail):
            angle, interval = self._get_angle_from_trajectory(trail, time, interval=True)
            if angle >= 0:
                prev_angle = angle
                for i in range(interval):
                    target_data['angle'].append(angle)
            else:
                for i in range(interval):
                    target_data['angle'].append(prev_angle)
            time += interval
        # Now that we know movement angle, invert any sudden angle changes
        # due to moving backwards. (I think this is a valid assumption & correction.)
        for p in range(1, len(target_data['angle'])):
            # print("1:", target_data['angle'][p], target_data['angle'][p-1])
            # print("2:", target_data['angle'][p] - target_data['angle'][p-1], abs(target_data['angle'][p] - target_data['angle'][p-1]) % (2 * np.pi))
            diff = self._smallest_angle_diff(target_data['angle'][p], target_data['angle'][p-1])
            print(diff)

            # if 

            # if abs(target_data['angle'][p] - target_data['angle'][p-1]) % (2 * np.pi) > np.pi/1.2:
            #     print("sudden theta:", target_data['angle'][p])
            #     target_data['angle'][p] = abs(target_data['angle'][p] - target_data['angle'][p-1] + np.pi) % (2 * np.pi)
            #     print("new theta:", target_data['angle'][p])
            
        return target_data


    def _calculate_area_between_trails(self, trail1, trail2):
        """
        Calculate the area between two trajectories.

        Parameters:
        - trail1: List of (x, y) tuples for the first trail.
        - trail2: List of (x, y) tuples for the second trail.

        Returns:
        - total_area: The total area between the two trails.
        """
        total_area = 0.0

        # Assuming both trails have the same number of points
        for i in range(1, len(trail1)):
            # Calculate the height (h) as the difference in x between successive points
            h = abs(trail1[i][0] - trail1[i-1][0])

            # Calculate the lengths of the parallel sides (b1 and b2)
            b1 = abs(trail1[i-1][1] - trail2[i-1][1])
            b2 = abs(trail1[i][1] - trail2[i][1])

            # Calculate the area of the trapezoid and add it to the total area
            trapezoid_area = 0.5 * (b1 + b2) * h
            # total_area += trapezoid_area
            total_area += 1 - (trapezoid_area / (np.sqrt(1 + trapezoid_area**2)))

        return total_area


    def _compare_actions(self, actions, trail, time):
        """
        Compare the agent's current action with the historical action of the target agent.
        """
        forward, backward, turn_left, turn_right = actions
        # Calculate the direction of movement required
        dx = trail[time+1][0] - trail[time][0]
        dy = trail[time+1][1] - trail[time][1]
        # Target direction in radians
        target_theta = math.atan2(dy, dx) % (2 * np.pi)
        # Calculate the smallest angle difference (target_theta - current_theta)
        angle_diff = (target_theta - self.ant.theta) % (2 * math.pi)

        return [forward, backward, turn_left, turn_right]


    def _reward_function(self, actions=None):
        """
        Calculate the reward given the focal ant and the accuracy of its behaviour
        over the trial, given the source data as the ground truth.
        """
        if REWARD_TYPE.lower() == 'trail':
            reward = self._calculate_area_between_trails(
                self.ant_trail,
                self.target_trail
            )

            return reward * -1
        elif REWARD_TYPE.lower() == 'action':
            return np.mean(self._compare_actions(actions,
                                         self.target_trail,
                                         self.t))

    def get_observations(self, others=None):
        return np.sum(self.ant.get_obs(others))


    def _track_trail(self, pos: vec2d):
        self.ant_trail.append(pos)


    def _destroy(self):
        self.ant = None
        self.ant_trail = []
        self.target_trail = []
        self.other_ants = None

        self.viewer = None
        self.state = None

        self.window = None
        self.clock = None


    def reset(self):
        self._destroy()

        self.t = 0      # timestep reset
        self.steps_beyond_done = None

        self.ant, self.target_trail, self.other_ants = self._select_target(
            others=True,
            trail_len=TIME_LIMIT
        )
        self.target_data = self._calculate_target_data(self.target_trail)
        self.ant.theta = self._get_angle_from_trajectory(self.target_trail, self.t)
        obs = self.get_observations(self.other_ants[:,self.t])

        if self.render_mode == 'human':
            self._render_frame()

        return obs


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


    def step(self, action):
        """
        Each step, take the given action and return observations, reward, done (bool)
        and any other additional information if necessary.
        """
        done = False
        self.t += 1

        # Pygame controls and resources if render_mode == 'human'
        if self.render_mode == "human": self._render_frame()

        action_set = self.ant.set_action(action)
        self.ant.update(self.ant_arena)
        obs = self.get_observations(self.other_ants[:,self.t])
        self._track_trail(self.ant.pos)

        if self.t >= self.t_limit:
            done = True

        info = {}

        reward = 0
        if done and REWARD_TYPE == 'trail':
            reward = self._reward_function()
        if REWARD_TYPE == 'action':
            reward = self._reward_function(action_set)

        return obs, reward, done, info


    def render(self, mode=None):
        if mode is not None:
            self.render_mode = mode
        # if self.render_mode == 'rgb_array':
        return self._render_frame()


    def _render_frame(self):
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption('WANNts')
            self.window = pygame.display.set_mode(
                (SCREEN_W, SCREEN_H)
            )

        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()
            return

        canvas = pygame.Surface((SCREEN_W, SCREEN_H))
        canvas.fill((200, 190, 210))

        # Project the circular arena
        pygame.draw.circle(
            canvas,
            (230, 230, 230),
            self.ant_arena[0],
            self.ant_arena[1]
        )

        if DRAW_ANT_VISION:
            pygame.draw.circle(
                canvas,
                (225, 225, 230),
                (self.ant.pos.x,
                self.ant.pos.y),
                VISION_RANGE
            )
            for (start_angle, stop_angle), colour in vision_segments:
                # Calculate start and stop angles, normalized to 0 to 2*pi
                start_angle = (self.ant.theta + start_angle) % (2 * math.pi)
                stop_angle = (self.ant.theta + stop_angle) % (2 * math.pi)
                
                pygame.draw.line(canvas, colour,
                    (self.ant.pos.x, self.ant.pos.y),
                    np.add(
                        np.array(self.ant.pos),
                        np.array([np.cos(start_angle), np.sin(start_angle)]) * VISION_RANGE
                    )
                )
            # pygame.draw.arc(
            #     canvas,
            #     colour,
            #     (self.ant.pos.x - VISION_RANGE,
            #     self.ant.pos.y - VISION_RANGE,
            #     VISION_RANGE*2, VISION_RANGE*2),
            #     arc_definitions[2][0][1], arc_definitions[2][0][0], 1
            # )


        ### DRAW TRAILS FIRST

        # Draw projected target trail
        if REWARD_TYPE.lower() == 'trail':
            for pos in self.target_trail:
                pygame.draw.rect(canvas, (220, 180, 180),
                                (pos[0] - ANT_DIM.x/2.,
                                pos[1] - ANT_DIM.y/2.,
                                ANT_DIM.x, ANT_DIM.y))
        else:
            try:
                for pos in self.target_trail[:self.t+1]:
                    pygame.draw.rect(canvas, (220, 180, 180),
                                    (pos[0] - ANT_DIM.x/2.,
                                    pos[1] - ANT_DIM.y/2.,
                                    ANT_DIM.x, ANT_DIM.y))
            except IndexError:
                for pos in self.target_trail:
                    pygame.draw.rect(canvas, (220, 180, 180),
                                    (pos[0] - ANT_DIM.x/2.,
                                    pos[1] - ANT_DIM.y/2.,
                                    ANT_DIM.x, ANT_DIM.y))

        # Draw ant trail
        trail_length = len(self.ant_trail)
        if TRACK_TRAIL == 'all':
            self.ant_trail_segment = self.ant_trail
        elif TRACK_TRAIL == 'fade':
            if trail_length > FADE_DURATION * SIM_FPS:
                self.ant_trail_segment = self.ant_trail[trail_length - FADE_DURATION * SIM_FPS:]
        else:
            self.ant_trail_segment = []
        for pos in self.ant_trail_segment:
            pygame.draw.rect(canvas, (180, 180, 220),
                        (pos.x - ANT_DIM.x/2.,
                        pos.y - ANT_DIM.y/2.,
                        ANT_DIM.x, ANT_DIM.y))

        ### THEN DRAW ANTS AT THEIR CURRENT POSITIONS

        # Draw other ants' positions
        if self.other_ants is not None:
            try:
                for other_ant in self.other_ants[:,self.t]:
                    pygame.draw.rect(canvas, (180, 180, 180),
                                    (other_ant[0] - ANT_DIM.x/2.,
                                    other_ant[1] - ANT_DIM.y/2.,
                                    ANT_DIM.x, ANT_DIM.y))

            except IndexError:
                print(other_ant)
                logger.error("End of time series reached for other ants.")
            except TypeError:
                print(other_ant)
                logger.error("Cannot draw ant with provided coordinates.")

        # Draw target ant
        pygame.draw.rect(canvas, (180, 0, 0),
                        (self.target_trail[-1][0] - ANT_DIM.x/2.,
                         self.target_trail[-1][1] - ANT_DIM.y/2.,
                         ANT_DIM.x, ANT_DIM.y))

        # Draw agent last; to ensure visibility.
        pygame.draw.rect(canvas, (0, 0, 180),
                        (self.ant.pos.x - ANT_DIM.x/2.,
                         self.ant.pos.y - ANT_DIM.y/2.,
                         ANT_DIM.x, ANT_DIM.y))
        pygame.draw.line(canvas, (0, 0, 180),
            (self.ant.pos.x, self.ant.pos.y),
            np.add(
                np.array(self.ant.pos),
                np.array([np.cos(self.ant.theta), np.sin(self.ant.theta)]) * ANT_DIM.x * 3
            )
        )


        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:   # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1,0,2)
            )


if __name__ == "__main__":
    env = AntDynamicsEnv(render_mode='human')
    total_reward = 0
    obs = env.reset()

    manual_mode = True
    manual_action = [0, 0, 0, 0]

    done = False
    while not done:
        if manual_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
                    if event.key == pygame.K_r:
                        env.reset()
                    if event.key == pygame.K_UP:    manual_action[0] = 1
                    if event.key == pygame.K_DOWN:  manual_action[1] = 1
                    if event.key == pygame.K_LEFT:  manual_action[2] = 1
                    if event.key == pygame.K_RIGHT: manual_action[3] = 1
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_UP:    manual_action[0] = 0
                    if event.key == pygame.K_DOWN:  manual_action[1] = 0
                    if event.key == pygame.K_LEFT:  manual_action[2] = 0
                    if event.key == pygame.K_RIGHT: manual_action[3] = 0
            action = manual_action

        if done: break

        obs, reward, done, _ = env.step(action)
        total_reward += reward


    env.close()
    print('Cumulative score:', total_reward)