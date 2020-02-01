class RolloutStorage:
    """ An object to store rollout information. """

    def __init__(self):
        """ init function for RolloutStorage class. """

        # HARDCODE
        self.obs = [None for _ in range(100)]

    def add_step(self, obs, action, value, reward):
        """ Add an environment step to storage. """

        pass

    def set_initial_obs(self, obs):
        """ Set the first observation in storage. """

        pass
