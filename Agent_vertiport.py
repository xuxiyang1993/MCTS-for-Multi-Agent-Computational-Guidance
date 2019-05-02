import argparse
import numpy as np
import time

from config_vertiport import Config
from MultiAircraftVertiportEnv import MultiAircraftEnv
from nodes_multi import MultiAircraftNode, MultiAircraftState
from search_multi import MCTS


def run_experiment(env, render, save_path):
    text_file = open(save_path, "w")  # save all non-terminal print statements in a txt file
    episode = 0
    epi_returns = []
    conflicts_list = []
    num_aircraft = Config.num_aircraft
    time_dict = {}

    while episode < Config.no_episodes:
        # at the beginning of each episode, set done to False, set time step in this episode to 0
        # set reward to 0, reset the environment
        if render:
            env.render()
        episode += 1
        done = False
        episode_time_step = 0
        episode_reward = 0
        last_observation, id_list = env.reset()
        action = np.ones(num_aircraft)
        info = None
        near_end = False
        counter = 0 # avoid end episode initially

        while not done:
            env.render()
            if episode_time_step % 5 == 0:

                time_before = int(round(time.time() * 1000))
                num_existing_aircraft = last_observation.shape[0]
                action = np.ones(num_existing_aircraft, dtype=np.int32)
                action_by_id = {}
                
                for index in range(num_existing_aircraft):
                    state = MultiAircraftState(state=last_observation, index=index, init_action=action)
                    root = MultiAircraftNode(state=state)
                    mcts = MCTS(root)
                    if info[index] < 3 * Config.minimum_separation:
                        best_node = mcts.best_action(Config.no_simulations, Config.search_depth)
                    else:
                        best_node = mcts.best_action(Config.no_simulations_lite, Config.search_depth_lite)
                    action[index] = best_node.state.prev_action[index]
                    action_by_id[id_list[index]] = best_node.state.prev_action[index]

                time_after = int(round(time.time() * 1000))
                if num_existing_aircraft in time_dict:
                    time_dict[num_existing_aircraft].append(time_after - time_before)
                else:
                    time_dict[num_existing_aircraft] = [time_after - time_before]
            (observation, id_list), reward, done, info = env.step(action_by_id, near_end)

            episode_reward += reward
            last_observation = observation
            episode_time_step += 1

            if episode_time_step % 100 == 0:
                print('========================== Time Step: %d =============================' % episode_time_step, file=text_file)
                print('Number of conflicts:', env.conflicts / 2, file=text_file)
                print('Total Aircraft Genrated:', env.id_tracker, file=text_file)
                print('Goal Aircraft:', env.goals, file=text_file)
                print('NMACs:', env.NMACs, file=text_file)
                print('Current Aircraft Enroute:', env.aircraft_dict.num_aircraft, file=text_file)

            if env.id_tracker >= 10000:
                counter += 1
                near_end = True
            
            if counter > 0:
                done = num_existing_aircraft == 0
            
        print('========================== End =============================', file=text_file)
        print('========================== End =============================')
        print('Number of conflicts:', env.conflicts / 2)
        print('Total Aircraft Genrated:', env.id_tracker)
        print('Goal Aircraft:', env.goals)
        print('NMACs:', env.NMACs)
        print('Current Aircraft Enroute:', env.aircraft_dict.num_aircraft)
        for key, item in time_dict.items():
            print('%d aircraft: %.2f' % (key, np.mean(item)))
            
        # print training information for each training episode
        epi_returns.append(info)
        conflicts_list.append(env.conflicts)
        print('Training Episode:', episode)
        print('Cumulative Reward:', episode_reward)

    time_list = time_dict.values()
    flat_list = [item for sublist in time_list for item in sublist]
    print('----------------------------------------')
    print('Number of aircraft:', Config.num_aircraft)
    print('Search depth:', Config.search_depth)
    print('Simulations:', Config.no_simulations)
    print('Time:', sum(flat_list) / float(len(flat_list)))
    print('NMAC prob:', epi_returns.count('n') / Config.no_episodes)
    print('Goal prob:', epi_returns.count('g') / Config.no_episodes)
    print('Average Conflicts per episode:', sum(conflicts_list) / float(len(conflicts_list)) / 2) # / 2 to ignore duplication
    env.close()
    text_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--save_path', '-p', type=str, default='output/seed2.txt')
    parser.add_argument('--render', '-r', action='store_true')
    args = parser.parse_args()

    import random
    np.set_printoptions(suppress=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    env = MultiAircraftEnv(args.seed)
    run_experiment(env, args.render, args.save_path)


if __name__ == '__main__':
    main()
