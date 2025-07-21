import highway_env
import gymnasium
from gymnasium.wrappers import RecordVideo
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cma
from cma.optimization_tools import EvalParallel2
import argparse
import warnings
warnings.filterwarnings('ignore')

env = gymnasium.make('racetrack-v0', render_mode='rgb_array')
class PD:
    def __init__(self, kp, kd, kp_v, kd_v, goal, v_goal):
        self.kp = kp
        self.kd = kd
        self.kp_v = kp_v
        self.kd_v = kd_v
        self.goal = goal
        self.v_goal = v_goal
        self.last_error = 0
        self.last_v_error = 0


    def observe(self, obs, info):
        # obs = obs[0]
        feature1 = 0
        feature2 = 12
        for i in range(12, 13):
            ok = 0
            for j in range(13):
                if(obs[i][j] == 1):
                    ok = 1
                    feature1 = j
                    break
            if(not ok):
                feature1 = 6

        for i in range(12, 13):
            ok = 0
            for j in range(13):
                if(obs[i][12 - j] == 1):
                    ok = 1
                    feature2 = ((12 - j))
                    break
            if(not ok):
                feature2 = 6

        x = (int)(feature1 + feature2)/2
        # print(self.goal, x)
        
        error = self.goal - x
  
        d_error = error - self.last_error
        self.last_error = error
        X = -1*np.tanh(self.kp * error + self.kd * d_error)

        v_error = self.v_goal - info["speed"]
        d_v_error = v_error - self.last_v_error
        self.last_v_error = v_error
        Y = (1-np.abs(X))*np.tanh(self.kp_v * v_error + self.kd_v * d_v_error)
        return [Y, X]


def policy(state, info, controller, eval_mode = False, params = []):

    # The next 3 lines are used for reading policy parameters learned by training CMA-ES. Do not change them even if you don't use CMA-ES.
    if eval_mode:
        param_df = pd.read_json("cmaes_params.json")
        params = np.array(param_df.iloc[0]["Params"])

    """Replace the default policy given below by your policy"""
    # print(state)
    # acceleration = 0
    # steering = 0
    # # return np.random.uniform(low=-1.0, high=1.0, size=(2,))
    # return [acceleration, steering]
    action = controller.observe(state, info) 
    return action


def fitness(params):

    """This is the fitness function which is optimised by CMA-ES.
    Note that the cma library minimises the fitness function by default.
    You should make suitable adjustments to make sure fitness is maximised"""

    # Write your fitness function below. You have to write the code to interact with the environment and
    # use the information provided by the environment to formulate the fitness function in terms of CMA-ES params
    # which are provided as an argument to this function. You can refer the code provided in evaluation section of
    # the main function to see how to interact with the environment. You should invoke your policy using the following:
    # policy(state, info, False, params)

    fitness_value = 0.0
    return fitness_value

def call_cma(num_gen=2, pop_size=2, num_policy_params = 1):
  sigma0 = 1
  x0 = np.random.normal(0, 1, (num_policy_params, 1))  # Initialisation of parameter vector
  opts = {'maxiter':num_gen, 'popsize':pop_size}
  es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
  with EvalParallel2(fitness, es.popsize + 1) as eval_all:
    while not es.stop():
      X = es.ask()
      es.tell(X, eval_all(X))
      es.logger.add()  # write data to disc for plotting
      es.disp()
  es.result_pretty()
  return es.result

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true')  # For training using CMA-ES
    parser.add_argument("--eval", action='store_true')  # For evaluating a trained CMA-ES policy
    parser.add_argument("--numTracks", type=int, default=6, required=False)  # Number of tracks for evaluation
    parser.add_argument("--seed", type=int, default=2025, required=False)  # Seed for evaluation
    parser.add_argument("--render", action='store_true')  # For rendering the evaluations
    args = parser.parse_args()

    train_mode = args.train
    eval_mode = args.eval
    num_tracks = args.numTracks
    seed = args.seed
    rendering = args.render


    """CMA-ES code begins"""
    # You can skip this part if you don't intend to use CMA-ES

    if train_mode:
        num_gen = 2
        pop_size = 2
        num_policy_params = 1
        X = call_cma(num_gen, pop_size, num_policy_params)
        cmaes_params = X[0]  # Parameters returned by CMA-ES after training
        cmaes_params_df = pd.DataFrame({
            'Params': [cmaes_params]
        })
        cmaes_params_df.to_json("cmaes_params.json")  # Storing parameters for evaluation purpose

    """CMA-ES code ends"""

    """Evaluation code begins"""
    # Do not modify this part.

    if rendering:
        env = RecordVideo(env, video_folder="videos", name_prefix="eval", episode_trigger=lambda x: True)

    if not train_mode:
        track_score_list = []  # This list stores the scores for different tracks

        for t in range(num_tracks):
            env.unwrapped.config["track"] = t  # Configuring the environment to provide track associated with index t. There are 6 tracks indexed 0 to 5.
            (obs, info) = env.reset(seed=seed)  # Getting initial state information from the environment
            controller = PD(kp=0.7, kd=0.7, kp_v=0.5, kd_v=0.5, goal=6, v_goal = 10)
            state = obs[0]
            done = False

            while not done:  # While the episode is not done
                action = policy(state, info, controller, eval_mode)  # Call policy to produce action
                (obs, _, term, trunc, info) = env.step(action)  # Take action in the environment
                state = obs[0]
                done = term or trunc  # If episode has terminated or truncated, set boolean variable done to True

            track_score = np.round(info["distance_covered"], 4).item()  # .item() converts numpy float to python float
            print("Track " + str(t) + " score:", track_score)
            track_score_list.append(track_score)

        env.close()

        # The next 4 lines of code generate a performance file which is used by autograder for evaluation. Don't change anything here.
        perf_df = pd.DataFrame()
        perf_df["Track_number"] = [n for n in range(num_tracks)]
        perf_df["Score"] = track_score_list
        perf_df.to_json("Performance_" + str(seed) + ".json")

        # A scatter plot is generated for you to visualise the performance of your agent across different tracks
        plt.scatter(np.arange(len(track_score_list)), track_score_list)
        plt.xlabel("Track index")
        plt.ylabel("Scores")
        plt.title("Scores across various tracks")
        plt.savefig('Evaluation.jpg')
        plt.close()

    """Code to generate learning curve and logs of CMA-ES"""
    # To be used only if your policy has parameters which are optimised using CMA-ES
    if train_mode:
        datContent = [i.strip().split() for i in open("outcmaes/fit.dat").readlines()]

        generations = []
        evaluations = []
        bestever = []
        best = []
        median = []
        worst = []

        for i in range(1, len(datContent)):
            generations.append(int(datContent[i][0]))
            evaluations.append(int(datContent[i][1]))
            bestever.append(-float(datContent[i][4]))
            best.append(-float(datContent[i][5]))
            median.append(-float(datContent[i][6]))
            worst.append(-float(datContent[i][7]))

        logs_df = pd.DataFrame()
        logs_df['Generations'] = generations
        logs_df['Evaluations'] = evaluations
        logs_df['BestEver'] = bestever
        logs_df['Best'] = best
        logs_df['Median'] = median
        logs_df['Worst'] = worst

        logs_df.to_csv('logs.csv')

        plt.plot(generations, best, color='green')
        plt.plot(generations, median, color='blue')
        plt.xlabel("Number of generations")
        plt.ylabel("Fitness")
        plt.legend(["Best", "Median"])
        plt.title('Evolution of fitness across generations')
        plt.savefig('LearningCurve.jpg')
        plt.close()

