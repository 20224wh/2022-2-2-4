#모듈 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import function as f

op = pd.read_csv('데이터/고령인구수.csv')

p = pd.read_csv('데이터/인구수.csv')


#인구 수
p_x, p_y = f.XY(p, 0)
#고령인구 수
op_x, op_y = f.XY(op, 0)
op_x = op_x[:-1]
op_y = op_y[:-1]

np_y = op_y/p_y
np_model = f.NR(op_x, np_y, 4)

x = np.arange(2001, 2091, 1)
y = np_model.predict(x[:, np.newaxis])
plt.plot(x, y)
plt.plot(p_x, np_y)
#88년