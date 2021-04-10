import numpy as np
import matplotlib.pyplot as plt


class Stock:

    def __init__(self, df, df2, init_money=100000, window_size=16, risk_prefer=0.3, target=0.1):

        self.n_actions = 3  # size of action space
        self.n_features = window_size  # number of features
        self.trend = df['Adj Close'].values  # adjusted close price
        self.df = df  # data frame of stock info
        self.df2 = df2
        self.init_money = init_money  # initial asset

        self.window_size = window_size  # size of rolling window
        self.half_window = window_size // 2

        self.buy_rate = 0.0003  # service charge of buying
        self.buy_min = 5  # minimum purchase rate
        self.sell_rate = 0.0003  # service charge of selling
        self.sell_min = 5  # maximum purchase rate
        self.stamp_duty = 0.001  # stamp duty
        self.risk_prefer = risk_prefer
        self.weekly_takeout = 0.005 * init_money
        self.invest_targett = target # take out fix amount of money every week

        # redundant, put here to avoid earning
        self.hold_money = self.init_money  # initial asset
        self.buy_num = 0  # quantity bought
        self.hold_num = 0  # number of shares held
        self.stock_value = 0  # total market capitalization of stocks held
        self.market_value = 0  # total market value (including cash)
        self.last_value = self.init_money  # market value in the previous day
        self.total_profit = 0  # total profit
        self.t = self.window_size // 2  # time stamp
        self.reward = 0  # reward

        self.states_sell = []  # selling time
        self.states_buy = []  # buying time

        self.profit_rate_account = []  # account earnings
        self.profit_rate_stock = []  # stock performance
        self.market_value = 0


    def reset(self):
        """
        reset account
        :return: observation at time 0
        """
        self.hold_money = self.init_money  # initial asset
        self.buy_num = 0  # quantity bought
        self.hold_num = 0  # number of shares held
        self.stock_value = 0  # total market capitalization of stocks held
        self.market_value = 0  # total market value (including cash)
        self.last_value = self.init_money  # market value in the previous day
        self.total_profit = 0  # total profit
        self.t = self.window_size // 2  # time stamp
        self.reward = 0  # reward

        self.states_sell = []  # selling time
        self.states_buy = []  # buying time

        self.profit_rate_account = []  # account earnings
        self.profit_rate_stock = []  # stock performance
        return self.get_state(self.t)


    def get_state(self, t):  # state for time t
        """
        Return the next observation in order to update state in Agent object
        :param t: time stamp
        :return: the next observation
        """

        window_size = self.window_size + 1
        d = t - window_size + 1
        # At the beginning of the simulation, history record was not enough, fill window with time 0 data

        block = []
        if d < 0:
            for i in range(-d):
                block.append(self.trend[0])
            for i in range(t + 1):
                block.append(self.trend[i])
        else:
            block = self.trend[d: t + 1]

        res = []
        for i in range(window_size - 1):
            res.append((block[i + 1] - block[i]) / (block[i] + 0.0001))  # 每步收益

        return np.array(res)  # state

    def buy_stock(self):
        """
        Define trading volume, calculate fee, update cash, asset, etc, record buying signal
        :return: no return
        """
        # buy stock, trading volume
        self.buy_num = ((self.hold_money - self.weekly_takeout) // self.trend[self.t]) // 100
        self.buy_num = self.buy_num * 100


        tmp_money = self.trend[self.t] * self.buy_num
        service_change = tmp_money * self.buy_rate
        if service_change < self.buy_min:
            service_change = self.buy_min

        if service_change + tmp_money > self.hold_money:
            self.buy_num = self.buy_num - 100
        tmp_money = self.trend[self.t] * self.buy_num
        service_change = tmp_money * self.buy_rate
        if service_change < self.buy_min:
            service_change = self.buy_min

        self.hold_num += self.buy_num
        self.stock_value += self.trend[self.t] * self.buy_num
        self.hold_money = self.hold_money - self.trend[self.t] * \
                          self.buy_num - service_change
        self.states_buy.append(self.t)

    def sell_stock(self, sell_num):
        tmp_money = sell_num * self.trend[self.t]
        service_change = tmp_money * self.sell_rate
        if service_change < self.sell_min:
            service_change = self.sell_min
        stamp_duty = self.stamp_duty * tmp_money
        self.hold_money = self.hold_money + tmp_money - service_change - stamp_duty
        self.hold_num = 0
        self.stock_value = 0
        self.states_sell.append(self.t)

    def trick(self):
        if self.t > 0 \
                and (self.df['MA5'][self.t-1] < self.df2['MA20'][self.t-1]) \
                and (self.df['MA5'][self.t] > self.df2['MA20'][self.t]):
            return True
        else:
            return False

    def step(self, action, show_log=False, my_trick=False):

        if action == 1 and self.hold_money >= (self.trend[self.t] * 100 +
                                               max(self.buy_min, self.trend[self.t] * 100 * self.buy_rate)) \
                        and self.t < (len(self.trend) - self.half_window):
            buy_ = True
            if my_trick and not self.trick():
                # If using own triggers does not trigger a buy condition, do not buy
                buy_ = False
            if buy_:
                self.buy_stock()
                if show_log:
                    print('day:%d, buy price:%f, buy num:%d, hold num:%d, hold money:%.3f' %
                          (self.t, self.trend[self.t], self.buy_num, self.hold_num, self.hold_money))

        elif action == 2 and self.hold_num > 0:
            # sell
            self.sell_stock(self.hold_num)
            if show_log:
                print(
                    'day:%d, sell price:%f, total balance %f,'
                    % (self.t, self.trend[self.t], self.hold_money)
                )
        else:
            if my_trick and self.hold_num > 0 and not self.trick():
                self.sell_stock(self.hold_num)
                if show_log:
                    print(
                        'day:%d, sell price:%f, total balance %f,'
                        % (self.t, self.trend[self.t], self.hold_money)
                    )

        self.stock_value = self.trend[self.t] * self.hold_num

        if self.t > 0 and self.t % 5 == 0:
            self.hold_money = self.hold_money - self.weekly_takeout

        self.market_value = self.stock_value + self.hold_money
        self.total_profit = self.market_value - self.init_money
        self.reward = (self.market_value - self.last_value) / self.last_value


        reward = (self.trend[self.t + 1] - self.trend[self.t]) - self.risk_prefer * np.sqrt(np.std(self.trend[:self.t]))

        if len(self.profit_rate_account) > 0:
            if np.mean(self.profit_rate_account[max(0, self.t-10): self.t]) >= 0.001:
                reward += 0.5
            else:
                reward -= 5


        if self.hold_num > 0 or action == 2:
            self.reward = reward
            if action == 2:
                self.reward = -self.reward
        else:
            self.reward = -self.reward * 0.1

        self.last_value = self.market_value

        self.profit_rate_account.append((self.market_value - self.init_money) / self.init_money)
        self.profit_rate_stock.append((self.trend[self.t] - self.trend[0]) / self.trend[0])
        done = False
        self.t = self.t + 1
        if self.t == len(self.trend) - 2:
            done = True
        s_ = self.get_state(self.t)
        reward = self.reward

        return s_, reward, done

    def get_info(self):
        return self.states_sell, self.states_buy, self.profit_rate_account, self.profit_rate_stock

    def draw(self, save_name1, save_name2):
        # plot the performance
        states_sell, states_buy, profit_rate_account, profit_rate_stock = self.get_info()
        invest = profit_rate_account[-1]
        total_gains = self.total_profit
        close = self.trend
        fig = plt.figure(figsize=(15, 5))
        plt.plot(close, color='dimgrey', lw=2.)
        plt.plot(close, 'v', markersize=8, color='darkseagreen', label='selling signal', markevery=states_sell)
        plt.plot(close, '^', markersize=8, color='indianred', label='buying signal', markevery=states_buy)
        plt.title('Total profit %f, percentage earning %f' % (total_gains, invest))
        plt.legend()
        plt.savefig(save_name1)
        plt.close()

        fig = plt.figure(figsize=(15, 5))
        plt.plot(profit_rate_account, color='indianred', label='portfolio')
        plt.plot(profit_rate_stock, color='darkseagreen', label='baseline')
        plt.legend()
        plt.savefig(save_name2)
        plt.close()