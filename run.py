from Environment import Stock
from Agent import DeepQNetwork
from data import *
import pandas as pd
from pandas_datareader import data


def game_step(env, observation, step=None, train=True, show_log=False, my_trick=False):
	# RL choose action based on observation
	action = RL.choose_action(observation, train)

	# RL take action and get next observation and reward
	observation_, reward, done = env.step(action, show_log=show_log, my_trick=my_trick)

	RL.store_transition(observation, action, reward, observation_)
	if step and (step > 3000) and (step % 8 == 0):
		RL.learn()

	# swap observation
	observation = observation_

	return observation, done


def run(env_list, max_round):
	step = 0
	total_profit_max = 0
	total_profit_round = 0
	for episode in range(max_round):
		profit = 0
		for env in env_list:  # iterate all the stocks
			# initial observation
			observation = env.reset()

			while True:
				observation, done = game_step(env, observation, step=step)
				# break while loop when end of this episode
				if done:
					break
				step += 1
			profit += env.total_profit
		profit = profit / len(env_list)
		print('epoch:%d, total_profit:%.3f' % (episode, env.total_profit))

		total_profit = 0
		for env in env_list2:
			BackTest(env, show_log=False, my_trick=True)
			total_profit += env.total_profit

		print(total_profit)

		if total_profit_max < total_profit:
			total_profit_max = total_profit
			total_profit_round = episode

	print(total_profit_round)


def BackTest(env, show_log=True, my_trick=False):
	observation = env.reset()
	while True:
		observation, done = game_step(env, observation, train=False,
									  show_log=show_log, my_trick=my_trick)
		if done:
			break


if __name__ == "__main__":

	start_date = "2011-01-01"
	end_date = "2020-12-31"

	panel_data = data.DataReader("TSLA", "yahoo", start_date, end_date)
	dataset_1 = pd.DataFrame(data=panel_data['Adj Close'])  # first step

	# training, testing stage_1
	for i in range(len(dataset_1)):
		if str(dataset_1.index[i]) == '2019-12-31 00:00:00':
			stage_1_training = dataset_1.iloc[:i + 1, :]
			stage_1_testing = dataset_1.iloc[i + 1:, :]

	index_data1 = Get_data(stage_1_training)

	env_list1 = []
	env_list2 = []
	env_list1.append(Stock(stage_1_training[31:], index_data1))
	env_list2.append(Stock(stage_1_testing[31:], index_data1))

	RL = DeepQNetwork(env_list1[0].n_actions, env_list1[0].n_features,
					  learning_rate=0.002,
					  reward_decay=0.9,
					  e_greedy=0.9,
					  replace_target_iter=300,
					  memory_size=7000,
					  batch_size=256,
					  )
	max_round = 20

	run(env_list1, max_round)

	i = 0
	for ele in env_list2:
		BackTest(ele, show_log=False)
		name1 = 'plots/trade1_' + str(i) + '.png'
		name2 = 'plots/profit1_' + str(i) + '.png'
		ele.draw(name1, name2)
		print('total_profit:%.3f' % ele.total_profit)
		i += 1
