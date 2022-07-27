'''
파일이름:function.py
과정
(1):비선형 형태인 데이터를 변형해서 선형으로 만든다.
(2):(1)의 데이터로 선형 회귀분석을 한다.
(3):회귀분석한 함수에 (1)에서 데이터를 변형한 것을 함수에 반대로 적용해서 비선형 회귀분석과 같은 효과를 내도록 한다.
'''
#모듈 불러오기
from sklearn.linear_model import LinearRegression   #선형회귀분석을 하는 함수
from sklearn.preprocessing import PolynomialFeatures   #비선형 회귀분석을 하는 함수
from sklearn.pipeline import make_pipeline   #위의 둘을 합쳐주는 함수
import numpy as np   #데이터를 처리할때 사용하는 모듈
import matplotlib.pyplot as plt   #그래프를 그릴때 사용하는 모듈

#비선형 회퀴분석 함수
def NR(x_data, y_data, d):
    x = np.array(x_data)[:, np.newaxis]   #x와 y를 sklearn에서 사용할 수 있는 형태로 바꿈
    y = np.array(y_data)
    
    model = make_pipeline(PolynomialFeatures(degree=d, include_bias=True), LinearRegression())   #모델 생성
    model.fit(x, y)  #학습
    
    return model

def XY(p):  #데이터를 x와 y로 분류한다. p:데이터
    x = np.array(list(map(int, list(p))))
    y = np.array(p.values[0])
    
    return x, y

def XYNR(p, d):   #데이터를 분류해서 회귀분석을 한다. p:데이터, d:만들 함수의 차수
    x, y = XY(p)   
    
    return x, y, NR(x, y, d)

def mk_plot(x, y, model):   #그래프를 그린다
    return plt.plot(x, y, x, model.predict(x[:, np.newaxis]))