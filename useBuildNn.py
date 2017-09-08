import numpy as np
import buildNn as bn
import pickle
with open('res/X1.pickle','rb') as f1:
	X = pickle.load(f1)
with open('res/XT1.pickle','rb') as f2:
	XT = pickle.load(f2)
with open('res/Y1.pickle','rb') as f3:
	Y = pickle.load(f3)
with open('res/YT1.pickle','rb') as f4:
	YT = pickle.load(f4)


# -------------------------以上代码获取数据，不必管-------------------------

parameters = bn.initialize_parameters_deep(X,[2,3,1])

# 向前传播
AS,ZS = bn.L_model_forward(X,parameters)
# 计算cost
cost = bn.compute_cost(AS[len(AS)-1],Y)
# 反向传播
dA,dW,db = bn.L_model_backward(AS,Y)

