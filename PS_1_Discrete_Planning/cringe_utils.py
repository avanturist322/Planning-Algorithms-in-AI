import numpy as np
import matplotlib.pyplot as plt
from utils import plotting_results

class heuristic_wrapper:
    def __init__(self, heuristic):
        self.heuristic = heuristic
        
    def wrap_heuristic(self):
        if self.heuristic == "0":
            return self.calculate_heuristic_0
        elif self.heuristic == "L1":
            return self.calculate_heuristic_l1
        elif self.heuristic == "L2":
            return self.calculate_heuristic_l2
        elif self.heuristic == "custom":
            return self.calculate_heuristic_custom
        elif self.heuristic == "cosine":
            return self.calculate_heuristic_cosine
        elif self.heuristic == "oriented1":
            return self.calculate_heuristic_oriented1
        elif self.heuristic == "oriented2":
            return self.calculate_heuristic_oriented2
        elif self.heuristic == "sine":
            return self.calculate_heuristic_sine
        elif self.heuristic == "random1":
            return self.calculate_random_heuristic_1
        elif self.heuristic == "random2":
            return self.calculate_random_heuristic_2
        else:
            raise NotImplementedError("This heuristic is not implemented. Try '0', 'L1', 'L2', 'custom', 'cosine', 'sine', 'oriented1', 'oriented2, 'random1', or 'random2' 🤗")
            
    def calculate_heuristic_0(self, state, goal_state):
        return 0
    
    def calculate_heuristic_l1(self, state, goal_state):
        # Calculate the L1 norm between the x, y components of the current state and the goal state
        x1, y1, _ = state
        x2, y2, _ = goal_state
        h = abs(x1 - x2) + abs(y1 - y2)
        return h
    
    def calculate_heuristic_l2(self, state, goal_state):
        # Calculate the L2 norm between the x, y components of the current state and the goal state
        x1, y1, _ = state
        x2, y2, _ = goal_state
        h = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        return h
    
    def calculate_heuristic_custom(self, state, goal_state):
        x1, y1, _ = state
        x2, y2, _ = goal_state
        h = (abs((x1 - x2) ** 3) + abs((y1 - y2) ** 3)) ** (1 / 3)
        return h
    
    def calculate_heuristic_cosine(self, state, goal_state):
        x1, y1, _ = state
        x2, y2, _ = goal_state
        h = abs(np.cos(x1 - x2) * np.cos(y1 - y2))
        return h
    
    def calculate_heuristic_sine(self, state, goal_state):
        x1, y1, angle1 = state
        x2, y2, angle2 = goal_state
        h = abs(np.sin(angle1) * np.sin(angle2) + np.sin(angle1 - angle2))
        return h
    
    def calculate_heuristic_oriented1(self, state, goal_state):
        x1, y1, angle1 = state
        x2, y2, angle2 = goal_state
        pos_distance = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
        angle_distance = 1 - np.dot(self.get_vector(angle1), self.get_vector(angle2))
        h = pos_distance + angle_distance
        return h
    
    def get_vector(self, angle):
        if angle == 0:
            return np.array([1, 0])
        elif angle == 1:
            return np.array([0, 1])
        elif angle == 2:
            return np.array([-1, 0])
        elif angle == 3:
            return np.array([0, -1])
        
    def calculate_heuristic_oriented2(self, state, goal_state):
        x1, y1, angle1 = state
        x2, y2, angle2 = goal_state
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle_diff = np.abs(angle2 - angle1)
        angle_diff = angle_diff % (2 * np.pi)
        
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
            
        distance_weight = 1.0
        angle_weight = 0.5
        h = distance_weight * distance + angle_weight * angle_diff
        return h

        
    def calculate_random_heuristic_1(self, state, goal_state):
        # A randomly defined heuristic 1
        return 42
    
    def calculate_random_heuristic_2(self, state, goal_state):
        # A randomly defined heuristic 2
        return 667


class a_super_star(heuristic_wrapper):
    def __init__(self, environment, rod, list_of_c_spaces, heuristic):
        super().__init__(heuristic)
        self.environment = environment
        self.rod = rod
        self.list_of_c_spaces = list_of_c_spaces
        self.calculate_heuristic = self.wrap_heuristic()  # Set the calculate_heuristic function based on the heuristic
        self.final_cost = None  # Variable to store the final cost
        self.states_visited = 0  # Counter to track the number of states visited
        
    def a_star(self, start_state, goal_state):
        print(f"Heuristic: {self.heuristic}")
        self.plan = None
        open_set = {start_state}
        g_scores = {start_state: 0}
        f_scores = {start_state: self.calculate_heuristic(start_state, goal_state)}
        came_from = {}

        it = 0
        while open_set:
            print(rf"Iteration: {it}", end="\r")
            current_state = min(open_set, key=lambda state: f_scores[state])
            if current_state == goal_state:
                self.plan = self.reconstruct_path(came_from, current_state)
                self.final_cost = f_scores[current_state]  # Set the final cost
                print("\n")
                print("Algorithm completed successfully! ✅✅✅")
                print(f"Final cost: {self.final_cost}")
                print(f"States visited: {self.states_visited}")
                return self.plan

            open_set.remove(current_state)

            for c_space in self.list_of_c_spaces:
                self.c_space = c_space
                successors = self.get_successors(current_state)
                for successor_state in successors:
                    tentative_g_score = g_scores[current_state] + self.calculate_cost(current_state, successor_state)

                    if successor_state not in g_scores or tentative_g_score < g_scores[successor_state]:
                        came_from[successor_state] = current_state
                        g_scores[successor_state] = tentative_g_score
                        f_scores[successor_state] = tentative_g_score + self.calculate_heuristic(successor_state, goal_state)
                        open_set.add(successor_state)
                        it += 1
                        self.states_visited += 1  # Increment the states visited counter

        return None
    
    def is_valid_state(self, state):
        x, y, angle = state
        if x < 0 or y < 0 or x >= self.environment.shape[0] or y >= self.environment.shape[1]:
            return False
        
        for ang, el in enumerate(self.list_of_c_spaces):
            if el[x, y] == 1 and ang == angle:
                return False
        
        if angle > 3 or angle < 0:
            return False
        
        return self.environment[x, y] == 0  
    
    def get_successors(self, state):
        successors = []
        x, y, angle = state
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                new_x, new_y, new_angle = x + dx, y + dy, angle
                if self.is_valid_state((new_x, new_y, new_angle)):
                    successors.append((new_x, new_y, new_angle))
                    
        for new_angle in range(3):
            if new_angle != angle:
                if self.is_valid_state((x, y, new_angle)):
                    successors.append((x, y, new_angle))

        return successors
    
    def reconstruct_path(self, came_from, current_state):
        total_path = [current_state]
        while current_state in came_from:
            current_state = came_from[current_state]
            total_path.append(current_state)
        return total_path[::-1]
    
    def calculate_cost(self, current_state, successor_state):
        return 1
        
    def get_trajectory_plot(self, start_state, goal_state):
        self.comment = self.heuristic
        for i, state in enumerate(self.plan):
            plt.plot(state[1], state[0], color="green", marker="o", markersize=4, label="Trajectory" if i == len(self.plan) - 1 else None)
            plt.imshow(self.environment)

        plt.plot(start_state[1], start_state[0], color="blue", marker="p", markersize=8, label="Initial position")
        plt.plot(goal_state[1], goal_state[0], color="red", marker="*", markersize=8, label="Final position")
        plt.title(f"Rod trajectory, heuristic: {self.comment}")
        plt.legend(bbox_to_anchor=(1.4, 0.92), loc='center right')
        plt.tight_layout()
        plt.show()
        
    def create_video(self):
        plotting_results(self.environment, self.rod, self.plan, f'rod_solve_{self.comment}.gif')