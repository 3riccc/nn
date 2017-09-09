import numpy as np
import buildNn as bn
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
		"layer_activation":"relu"
	},
	{
		"layer_num":8,
		"layer_activation":"tanh"
	},{
		"layer_num":4,
		"layer_activation":"relu"
	},{
		"layer_num":4,
		"layer_activation":"relu"
	},
	{
		"layer_num":1,
		"layer_activation":"sigmoid"
	}
]
# parameters = bn.initialize_parameters_deep(X,layers)
# AS,ZS = bn.L_model_forward(X,parameters,layers)
# cost = bn.compute_cost(AS[len(AS)-1],Y,layers[len(layers)-1]["layer_activation"])
# # 反向传播
# grads = bn.L_model_backward(AS,ZS,Y,parameters,layers)
# parameters = bn.update_parameters(parameters,grads,0.001)


parameters,costs,accuracies = bn.nn_model(X,Y,XT,YT,layers,500,0.4,True,True)



plt.plot(accuracies)
plt.ylabel('accuracies')
plt.xlabel('iterations / 10')
plt.show()

# 测试集效果曲线
plt.plot(costs)
plt.ylabel('accuracies')
plt.xlabel('iterations / 10')
plt.show()