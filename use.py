import numpy as np
import pickle
import matplotlib.pyplot as plt

with open('res/X1.pickle','rb') as f1:
	X = pickle.load(f1)
with open('res/XT1.pickle','rb') as f2:
	XT = pickle.load(f2)
with open('res/Y1.pickle','rb') as f3:
	Y = pickle.load(f3)
with open('res/YT1.pickle','rb') as f4:
	YT = pickle.load(f4)



# -------------------------以上代码获取数据，不必管-------------------------
import oneHiddenLayer as oh

parameters,costs = oh.nn_model(X,Y,42,200,0.01,True)

YP,accuracy = oh.predict(parameters,X,Y)

# 最终测试精度
print("测试精度为："+str(accuracy)+"%")

# 画出costs下降曲线
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.show()
