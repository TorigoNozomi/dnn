import numpy as np

def ReLU(x):
    y = np.maximum(0, x)
    return y
#uniform distribution
#data = 2*np.random.rand(10000,10000)-1
#normal distribution
data = np.random.randn(10000,10000)
data_2 = np.power(data, 2)
relu_data = ReLU(data)
relu_data_2 = np.power(relu_data, 2)

m = np.mean(data)
m_2 = np.mean(data_2)/2
m_ReLU = np.mean(relu_data)
m_ReLU_2 = np.mean(relu_data_2)
#median = np.median(data)
#variance = np.variance(data)
#stdev = np.stdev(data)
print('平均: {0:.2f}'.format(m))
print('平均_2: {0:.2f}'.format(m_2))
print('ReLU平均: {0:.2f}'.format(m_ReLU))
print('ReLU平均_2: {0:.2f}'.format(m_ReLU_2))
#print('中央値: {0:.2f}'.format(median))
#print('分散: {0:.2f}'.format(variance))
#print('標準偏差: {0:.2f}'.format(stdev))