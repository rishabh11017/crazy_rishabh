import numpy as np
#this file is jsut to store all the details of the episode here by defa
class RolloutBuffer:
    def __init__(
        self,
        rollout_steps=100,
        state_dim=18,
        action_dim=3,
        image_height=224,
        image_width=224,
        image_channels=3
    ):
        self.max_steps = rollout_steps
        self.ptr = 0
        # ===========================
        # Numerical State
        # ===========================
        self.states = np.zeros((rollout_steps, state_dim),dtype=np.float32)
        self.next_states = np.zeros((rollout_steps, state_dim),dtype=np.float32 )
        # ===========================
        # Front Camera
        # ===========================

        self.front_images = np.zeros((rollout_steps, image_height,
                     image_width, image_channels ), dtype=np.uint8)
        # Downward Camera
        # ===========================
        self.down_images = np.zeros(  ( rollout_steps,image_height,
                image_width,image_channels ), dtype=np.uint8 )
        # PPO Data
        # ===========================
        self.actions = np.zeros((rollout_steps, action_dim),
            dtype=np.float32)
        
        self.rewards = np.zeros(rollout_steps, dtype=np.float32 )

        self.dones = np.zeros(rollout_steps,dtype=bool)

        self.values = np.zeros(
            rollout_steps,
            dtype=np.float32
        )

        self.log_probs = np.zeros(
            rollout_steps,
            dtype=np.float32
        )

        self.returns = np.zeros(
            rollout_steps,
            dtype=np.float32
        )

        self.advantages = np.zeros(
            rollout_steps,
            dtype=np.float32
        )

        # ===========================
        # Metadata
        # ===========================

        self.episode = np.zeros(
            rollout_steps,
            dtype=np.int32
        )

        self.step = np.zeros(
            rollout_steps,
            dtype=np.int32
        )

        self.timestamp = np.zeros(
            rollout_steps,
            dtype=np.float64
        )

        # ===========================
        # Future LSTM Support
        # ===========================

        self.hidden_state = None
        self.cell_state = None

    def store(
        self,
        state,
        next_state,
        front_image,
        down_image,
        action,
        reward,
        done,
        value,
        log_prob,
        episode,
        step,
        timestamp
    ):

        i = self.ptr

        self.states[i] = state
        self.next_states[i] = next_state

        self.front_images[i] = front_image
        self.down_images[i] = down_image

        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.log_probs[i] = log_prob

        self.episode[i] = episode
        self.step[i] = step
        self.timestamp[i] = timestamp

        self.ptr += 1

    def clear(self):

        self.ptr = 0

    def size(self):

        return self.ptr