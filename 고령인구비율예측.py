'''
파일이름:고령인구비율예측.py
과정
(1):고령인구 수 데이터와 인구 수 데이터를 불러온다.
(2):고령인구 수 데이터와 인구 수 데이터를 이용하기 편한 형태로 바꾼다.
(3):(고령인구 수)/(인구 수)를 해서 고령인구의 비율을 구한다.
(4):회귀분석을 하고 그 결과를 그래프로 그린다.
'''
#모듈 불러오기
import numpy as np   #데이터를 처리할때 사용하는 모듈
import pandas as pd   #csv파일을 불러올때 사용하는 모듈
import matplotlib.pyplot as plt   #그래프를 그릴때 사용하는 모듈
import function as f    #회귀분석을 할때 사용하는 함수를 만들어 놓은 파일

#인구 수
p = pd.read_csv('데이터/인구수.csv')   #인구 수 파일 불러오기
p_x, p_y = f.XY(p)   #불러온 인구 수 데이터를 사용할 수 있게 x와 y로 분류하기

#고령인구 수
op = pd.read_csv('데이터/고령인구수.csv')   #고령 인구 수 파일 불러오기
op_x, op_y = f.XY(op)   #불러온 고령 인구 수 데이터를 사용할 수 있게 x와 y로 분류하기
op_x = op_x[:-1]   #인구 수 데이터는 2019년까지 있지만 고령 인구 수 데이터는 2020년까지 있어서 2020년 것 제거하기
op_y = op_y[:-1]

#회귀분석
np_y = op_y/p_y   #연도별 고령 인구의 비율 구하기
np_model = f.NR(op_x, np_y, 3)   #구한 비율로 회귀분석하기

#그래프로 그리기
x = np.arange(1960, 2100, 1)   #y좌표를 구할 x좌표 저장
y = np_model.predict(x[:, np.newaxis])   #y좌표 구하기
plt.plot(x, y, label='NR', c='blue')   #비선형 회귀분석한 그래프 그리기
plt.plot(p_x, np_y, label='data', c='red')   #실제 데이터의 그래프 그리기
plt.legend()   #레이블 보이기