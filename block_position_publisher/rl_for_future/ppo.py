import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 

from block_position_publisher.block_position_publisher.rl_for_future.actor_critic import ActorCritic
from block_position_publisher.block_position_publisher.rl_for_future.rollout_buffer import RolloutBuffer
class PPO:
    def __init__(
        self,
        state_dim=18,
        action_dim=3,
        rollout_steps=500,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        ppo_epochs=10,
        mini_batch_size=64,
        device="cuda"
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.rollout_steps = rollout_steps
        # ----------------------------------

        # Neural Network

        # ----------------------------------

        self.policy = ActorCritic().to(device)


        # ----------------------------------

        # Optimizer

        # ----------------------------------

        self.optimizer = optim.AdamW(

            self.policy.parameters(),

            lr=lr

        )


        # ----------------------------------

        # Rollout Buffer

        # ----------------------------------

        self.buffer = RolloutBuffer(

            rollout_steps=rollout_steps

        )
    def select_action(
    self,
    state,
    front_image=None,
    down_image=None,
    hidden_state=None,
    cell_state=None
):

        with torch.no_grad():

            action, log_prob, value, hidden_state, cell_state = self.policy.act(

                state,

                front_image,

                down_image,

                hidden_state,

                cell_state

            )

        return (

            action,

            log_prob,

            value,

            hidden_state,

            cell_state

        )
    def compute_returns_and_advantages(
    self,
    last_value=0.0
):

        advantages = []
        gae = 0.0
        values = np.append(self.buffer.values[:self.buffer.ptr], last_value)
        for step in reversed(range(len(self.buffer.rewards))):
            if self.buffer.dones[step]:
                mask = 0.0
            else:
                mask = 1.0
            delta = (
                self.buffer.rewards[step]+
                self.gamma*
                values[step+1]
                *
                mask
                -
                values[step]
            )

            gae = (

                delta

                +

                self.gamma

                *

                self.gae_lambda

                *

                mask

                *

                gae

            )

            advantages.insert(

                0,

                gae

            )

        returns = [
            adv + val
            for adv,val in zip(
                advantages,
                self.buffer.values
            )
        ]
        self.buffer.advantages = advantages
        self.buffer.returns = returns
    def get_mini_batches(
    self,
    states,
    front_images,
    down_images,
    actions,
    returns,
    advantages,
    old_log_probs
):

        N = states.size(0)

        ####################################################
        # MODE 1 : Standard PPO
        ####################################################

        if self.batch_mode == "transition":

            indices = torch.randperm(N, device=self.device)

            for start in range(0, N, self.mini_batch_size):

                end = start + self.mini_batch_size

                idx = indices[start:end]

                yield (
                    states[idx],
                    front_images[idx],
                    down_images[idx],
                    actions[idx],
                    returns[idx],
                    advantages[idx],
                    old_log_probs[idx]
                )

        ####################################################
        # MODE 2 : Sequence PPO
        ####################################################

        elif self.batch_mode == "sequence":

            seq = self.sequence_length

            num_sequences = N // seq

            sequence_indices = torch.randperm(
                num_sequences,
                device=self.device
            )

            for start in range(
                0,
                num_sequences,
                self.mini_batch_size
            ):

                end = start + self.mini_batch_size

                batch_sequences = sequence_indices[start:end]

                state_batch = []
                front_batch = []
                down_batch = []
                action_batch = []
                return_batch = []
                advantage_batch = []
                logprob_batch = []

                for s in batch_sequences:

                    a = s * seq
                    b = a + seq

                    state_batch.append(states[a:b])
                    front_batch.append(front_images[a:b])
                    down_batch.append(down_images[a:b])
                    action_batch.append(actions[a:b])
                    return_batch.append(returns[a:b])
                    advantage_batch.append(advantages[a:b])
                    logprob_batch.append(old_log_probs[a:b])

                yield (

                    torch.stack(state_batch),

                    torch.stack(front_batch),

                    torch.stack(down_batch),

                    torch.stack(action_batch),

                    torch.stack(return_batch),

                    torch.stack(advantage_batch),

                    torch.stack(logprob_batch)

                )

        else:

            raise ValueError(
                f"Unknown batch mode : {self.batch_mode}"
            )  
        
    def update(self):
        states = torch.FloatTensor(
        self.buffer.states[:self.buffer.ptr]
    ).to(self.device)

    front_images = torch.FloatTensor(
        self.buffer.front_images[:self.buffer.ptr]
    ).permute(0,3,1,2).to(self.device)

    down_images = torch.FloatTensor(
        self.buffer.down_images[:self.buffer.ptr]
    ).permute(0,3,1,2).to(self.device)

    actions = torch.FloatTensor(
        self.buffer.actions[:self.buffer.ptr]
    ).to(self.device)

    returns = torch.FloatTensor(
        self.buffer.returns[:self.buffer.ptr]
    ).to(self.device)

    advantages = torch.FloatTensor(
        self.buffer.advantages[:self.buffer.ptr]
    ).to(self.device)

    old_log_probs = torch.FloatTensor(
        self.buffer.log_probs[:self.buffer.ptr]
    ).to(self.device)

    #normalize images for resnet
    front_images /= 255.0
    down_images /= 255.0
    #normalize advantages 
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


    # one of the self thought ideas is to suffle the 
    #data first at timestep level later on sequence leavel
    #and then full episode left unshuffeled so that earlier optimization stable then lstm learn 
    #velocily and othr camerera pattern 
  
    for epoch in range(self.ppo_epochs):

        for (batch_states,  batch_front_images, batch_down_images,
            batch_actions, batch_returns, batch_advantages,
            batch_old_log_probs ) in self.get_mini_batches(states,
            front_images, down_images, actions, returns, advantages,
             old_log_probs):
            new_log_probs, entropy, state_values = self.policy.evaluate(batch_states,
            batch_actions,batch_front_images,batch_down_images)
            ratio = torch.exp(
                new_log_probs -
                batch_old_log_probs)
            
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - self.clip_epsilon,
                1.0 + self.clip_epsilon
            ) * batch_advantages

            actor_loss = -torch.min( surr1, surr2).mean()

            critic_loss = nn.functional.mse_loss(
                state_values.squeeze(-1),
                batch_returns
            )
            entropy_loss = entropy.mean()

            loss = (actor_loss + self.value_coef * critic_loss -
              self.entropy_coef * entropy_loss)
            


            self.optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(

                self.policy.parameters(),

                0.5

            )

            self.optimizer.step()
    self.buffer.clear()
            