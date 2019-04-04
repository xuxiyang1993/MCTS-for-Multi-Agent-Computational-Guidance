import numpy as np


class MCTSState:
    def __init__(self, state):
        self.state = state

    def reward(self):
        raise NotImplemented("Implement game_result function")

    def is_terminal_state(self, search_depth):
        raise NotImplemented("Implement is_game_over function")

    def move(self, action):
        raise NotImplemented("Implement move function")

    def get_legal_actions(self):
        raise NotImplemented("Implement get_legal_actions function")


class MCTSNode:
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        self.q = 0.
        self.n = 0

    @property
    def untried_actions(self):
        raise NotImplemented()

    def expand(self):
        raise NotImplemented()

    def is_terminal_node(self, search_depth):
        raise NotImplemented()

    def rollout(self, search_depth):
        raise NotImplemented()

    def backpropagate(self, reward):
        raise NotImplemented()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[int(np.argmax(choices_weights))]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]
