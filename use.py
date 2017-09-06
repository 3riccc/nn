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


parameters,costs,accuracies = oh.nn_model(X,Y,XT,YT,42,10000,0.005,True)




# 测试集效果曲线
plt.plot(accuracies)
plt.ylabel('accuracies')
plt.xlabel('iterations / 10')
plt.show()
