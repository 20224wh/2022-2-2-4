'''
파일 이름:신생아수와사망자수.py
과정
(1):출생아 수 파일과 사망자 수 파일을 불러온다.
(2):각각 x와 y로 분류한다.
(3):그래프를 그린다.
'''
import pandas as pd   #데이터를 불러올때 사용하는 모듈
import matplotlib.pyplot as plt   #그래프를 그릴때 사용하는 모듈
import function as f   #회귀분석을 할때 사용하는 함수를 만들어 놓은 파일

#파일 불러오기
bp = pd.read_csv('데이터/출생아수.csv')   #출생아 수
dp = pd.read_csv('데이터/사망자수.csv')   #사망자 수

#x와 y로 분류하기
bp_x, bp_y = f.XY(bp)   #출생아 수
dp_x, dp_y = f.XY(dp)   #사망자 수

#그래프 그리기
plt.title('test')
plt.plot(bp_x, bp_y, label='number of births', c='blue')   #출생아 수
plt.plot(dp_x, dp_y, label='number of deaths', c='red')   #사망자 수
plt.legend()   #레이블 보이기
