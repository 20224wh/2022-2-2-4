#모듈 불러오기
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt

#비선형 회퀴분석 함수
def NR(x_data, y_data, d):
    x = np.array(x_data)[:, np.newaxis]   #x와 y를 sklearn에서 사용할 수 있는 형태로 바꿈
    y = np.array(y_data)
    
    model = make_pipeline(PolynomialFeatures(degree=d, include_bias=True), LinearRegression())   #모델 생성
    model.fit(x, y)  #학습
    
    return model

def XY(p, n):
    x = np.array(list(map(int, list(p))))
    y = np.array(p.values[n])
    
    return x, y

def XYNR(p, n, d):
    x, y = XY(p, n)
    
    return x, y, NR(x, y, d)

def mk_plot(x, y, model):
    return plt.plot(x, y, x, model.predict(x[:, np.newaxis]))