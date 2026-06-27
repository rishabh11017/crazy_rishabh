import time
import torch

from environment import DroneEnvironment
from ppo import PPO


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = DroneEnvironment()

    agent = PPO(device=device)

    num_episodes = 10000

    save_interval = 100

    for episode in range(num_episodes):

        state, front_image, down_image = env.reset()

        hidden_state, cell_state = agent.policy.memory.initialize_memory(
            batch_size=1,
            device=device
        )

        done = False

        episode_reward = 0.0

        step = 0

        while not done:

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            front_tensor = (
                torch.FloatTensor(front_image)
                .permute(2,0,1)
                .unsqueeze(0)
                .to(device)
            )

            down_tensor = (
                torch.FloatTensor(down_image)
                .permute(2,0,1)
                .unsqueeze(0)
                .to(device)
            )

            action, log_prob, value, hidden_state, cell_state = agent.select_action(

                state_tensor,

                front_tensor,

                down_tensor,

                hidden_state,

                cell_state

            )

            action_np = action.squeeze(0).cpu().numpy()

            next_state, next_front, next_down, reward, done = env.step(action_np)

            agent.buffer.store(

                state,

                next_state,

                front_image,

                down_image,

                action_np,

                reward,

                done,

                value.item(),

                log_prob.item(),

                episode,

                step,

                time.time()

            )

            state = next_state

            front_image = next_front

            down_image = next_down

            episode_reward += reward

            step += 1

        agent.compute_returns_and_advantages()

        stats = agent.update()

        print(
            f"Episode {episode:5d} | "
            f"Reward {episode_reward:8.2f} | "
            f"Actor {stats['actor_loss']:.4f} | "
            f"Critic {stats['critic_loss']:.4f}"
        )

        if episode % save_interval == 0:

            torch.save(
                agent.policy.state_dict(),
                f"checkpoint_{episode}.pth"
            )


if __name__ == "__main__":

    train()