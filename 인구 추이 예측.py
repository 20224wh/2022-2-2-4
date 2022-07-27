'''
파일 이름:인구 추이 예측.py
과정
(1):총 인구 수 파일을 불러온다.
(2):(1)에서 불러온 파일을 사용하기 편하게 바꾼다.
(3):(2)의 파일로 회귀분석을 하여 그래프로 그린다.
'''
#모듈 불러오기
import numpy as np   #데이터를 처리할때 사용하는 모듈
import pandas as pd   #데이터를 불러올때 사용하는 모듈
import matplotlib.pyplot as plt   #그래프를 그릴때 사용하는 모듈
import function as f   #회귀분석을 할때 사용하는 함수를 만들어 놓은 파일

#인구 수
p = pd.read_csv('데이터/인구수.csv')   #불러오기
p_x, p_y, p_model = f.XYNR(p, 2)   #회귀분석하기

#그래프 그리기
x = np.arange(1951, 2136, 1)   #그래프에 그릴 점들의 x좌표 만들기
y = p_model.predict(x[:, np.newaxis])   #함수로 y좌표 구하기
plt.plot(x, y, label='NR', c='blue')   #회귀분석으로 만든 함수를 그래프로 그리기
plt.plot(p_x, p_y, label='data', c='red')   #원래 데이터를 그래프로 그리기
plt.legend()   #레이블 보이기