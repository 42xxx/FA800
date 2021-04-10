import pandas as pd
import numpy as np
from pandas_datareader import data


def Get_data(Stocks_data):
	df = pd.DataFrame()
	for i in range(len(Stocks_data.columns)):
		df[i] = Stocks_data.iloc[:, i]

		# RSI
		delta = df[i].diff()
		up, down = delta.copy(), delta.copy()
		up[up < 0] = 0
		down[down > 0] = 0
		roll_up1 = up.ewm(span=14).mean()
		roll_down1 = down.abs().ewm(span=14).mean()

		RS = roll_up1 / roll_down1
		RSI = 100.0 - (100.0 / (1.0 + RS))
		index_data = pd.DataFrame(RSI[31:])
		index_data.columns = ['RSI']

		# SR
		SR = []
		data1 = df[i]
		for j in range(31, len(data1)):
			dat = data1[j - 30:j]
			returns = np.log(dat / dat.shift(1))
			volatility = returns.std() * np.sqrt(252)
			SR1 = (returns.mean() - 0.05) / volatility
			SR.append(SR1)
		SR = np.array(SR)
		index_data['SR'] = SR

		# EWM
		EWM = data1.ewm(span=20, adjust=False).mean()
		index_data['EWM'] = EWM[31:]

		# MA,window=20
		index_data['MA20'] = data1.rolling(window=20).mean()

		return index_data


print(Get_data('AAPL', "2011-01-01", "2020-12-31"))
a = Get_data('AAPL', "2011-01-01", "2020-12-31")
a = pd.DataFrame(a)
