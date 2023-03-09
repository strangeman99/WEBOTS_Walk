import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import gym
from CustomEnv import WalkingEnv


def main():
    # seed = 42
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_max = 1.0
    epsilon_interval = (epsilon_max - epsilon_min)
    batch_size = 32
    max_steps_per_episode = 10000
    learning_rate = 0.00025
    clipnorm = 1.0

    env_name = 'WalkingEnv-v0'
    gym.envs.register(id=env_name, entry_point='CustomEnv:WalkingEnv')
    env = gym.make(env_name)
    # env.seed(seed)    We only need to seed if there's random numbers

    num_actions = 4

    def createModel():
        # Look at changing this network to not use the camera
        # Network for Walking Simulation
        # Create input layer for the first input
        inputs_1 = layers.Input(shape=(512, 512, 4))
        # Create input layer for the second input
        inputs_2 = layers.Input(shape=(12, 4))

        # reshape the inputs_2
        inputs_2_reshape = layers.Reshape((48,))(inputs_2)

        # Merge the two input layers
        merged = layers.concatenate([inputs_1, inputs_2_reshape])

        # Convolutions on the frames on the screen and motor inputs
        layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(merged)
        layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
        layer4 = layers.Flatten()(layer3)
        layer5 = layers.Dense(512, activation="relu")(layer4)

        # Output layer. Outputs percentage between the motor range
        action_out = layers.Dense(12, activation="sigmoid")(layer5)

        return keras.Model(inputs=[inputs_1, inputs_2], outputs=action_out)

    model = createModel()  # creating a network
    model_target = createModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)

    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    frame_count = 0

    epsilon_random_frames = 50000
    epsilon_greedy_frames = 1000000.0

    max_memory_length = 100000
    update_after_actions = 4
    update_target_network = 10000

    loss_function = keras.losses.Huber()

    # Motor parameters

    while True:  # going until it converges
        state = np.array(env.reset(), dtype=object)
        episode_reward = 0

        for timestep in range(1, max_steps_per_episode):
            # env.render()  # this shows the attempts made by the agent (slows down training)
            frame_count += 1

            # if the agent should explore
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # TODO change to random numbers within the motor range
                action = np.random.choice(num_actions)  # taking a random action to explore
            else:  # if the agent should take the best choice
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                # TODO See if this needs to be changed
                action_probs = model(state_tensor,
                                     training=False)  # pretty much calculating all of the action probabilities
                # TODO change to random numbers within the motor range
                action = tf.argmax(action_probs[0]).numpy()  # taking the best action possible

            # decrease the probability of exploring
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            # applying the chosen action to the environment
            state_next, reward, done, _ = env.step(action)
            state_next = np.array(state_next)

            episode_reward += reward

            # saving all of the states of the actions
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next  # moving to the next state

            print(frame_count)  # So the user can see the number of frames

            # update every fourth frame and once the batch size is over 32
            if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
                # get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                # sampling from the replay buffer
                state_sample = np.array([state_history[i] for i in indices]).astype("float64")
                state_next_sample = np.array([state_next_history[i] for i in indices]).astype("float64")
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )
                # update the Q-values
                future_rewards = model_target.predict(state_next_sample)
                # q val = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)

                # if the final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                # creating a mask to only calculate loss on updated q values
                masks = tf.one_hot(action_sample, num_actions)

                with tf.GradientTape() as tape:
                    # training the model
                    q_values = model(state_sample)

                    # applying the mask and calculating the loss
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if frame_count % update_target_network == 0:
                # update the network with new weights
                model_target.set_weights(model.get_weights())
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))

            # limit the state and reward count
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                break

        # check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]

        running_reward = np.mean(episode_reward_history)
        episode_count += 1

        # to consider the task solved
        if running_reward > 40:
            file = open("TheModel.txt", "w")
            file.write(model)
            file.close()
            print("Solved at episode {}!".format(episode_count))
            break


if __name__ == '__main__':
    main()
