import numpy as np
import buildNn as bn
import pickle
import matplotlib.pyplot as plt
# X.shape 必须是(nx,m) ，其中nx是特征数量，m是样本数量，XT相同
with open('res/X1.pickle','rb') as f1:
	X = pickle.load(f1)
with open('res/XT1.pickle','rb') as f2:
	XT = pickle.load(f2)
# Y.shape 必须是(1,m) ，其中m是样本数量，YT相同
with open('res/Y1.pickle','rb') as f3:
	Y = pickle.load(f3)
with open('res/YT1.pickle','rb') as f4:
	YT = pickle.load(f4)


# -------------------------以上代码获取数据，不必管-------------------------


# # 向前传播
# AS,ZS = bn.L_model_forward(X,parameters)
# # 计算cost
# cost = bn.compute_cost(AS[len(AS)-1],Y)
# # 反向传播
# grads = bn.L_model_backward(AS,ZS,Y,parameters)
# # 更新参数
# parameters = bn.update_parameters(parameters,grads,0.001)
layers = [
	{
		"layer_num":5,
		"layer_activation":"tanh"
	},{
		"layer_num":5,
		"layer_activation":"tanh"
	},{
		"layer_num":5,
		"layer_activation":"tanh"
	},
	{
		"layer_num":1,
		"layer_activation":"sigmoid"
	}
]

# X = np.array([[1,2,3],[4,5,6]])
# XT = np.array([[2,4],[7,8]])
# Y = np.array([[1,1,0]])
# YT = np.array([[0,1]])


# 归一化
X,us,sigma2 = bn.normalizing_train(X)
XT = bn.normalizing_test(XT,us,sigma2)


parameters,costs,accuracies = bn.nn_model(X,Y,XT,YT,layers,1000,0.04,True,True)



plt.plot(accuracies)
plt.ylabel('accuracies')
plt.xlabel('iterations / 10')
plt.show()

# 测试集效果曲线
plt.plot(costs)
plt.ylabel('accuracies')
plt.xlabel('iterations / 10')
plt.show()