import numpy as np

# sigmoid
def sigmoid(z):
	s = 1./(1. + np.exp(-z))
	return s

# 输入层，隐藏层和输出层的size
def layer_sizes(X,h,Y):

	n_x = X.shape[0]
	n_h = h
	n_y = Y.shape[0]

	return (n_x,n_h,n_y)

# 参数初始化
def initialize_parameters(n_x,n_h,n_y):

	# 初始化
	W1 = np.random.randn(n_h,n_x) * 0.01
	b1 = np.zeros((n_h,1))
	W2 = np.random.randn(n_y,n_h) * 0.01
	b2 = np.zeros((n_y,1))

	# 确认参数外形
	assert(W1.shape == (n_h,n_x))
	assert(b1.shape == (n_h,1))
	assert(W2.shape == (n_y,n_h))
	assert(b2.shape == (n_y,1))

	parameters = {
		"W1":W1,
		"b1":b1,
		"W2":W2,
		"b2":b2
	}
	return parameters

# 向前传播
def forward_propagation(X,parameters):
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	# 传播
	Z1 = np.dot(W1,X) + b1
	A1 = np.tanh(Z1)
	Z2 = np.dot(W2,A1) + b2
	A2 = sigmoid(Z2)

	# 确认形状
	assert(A2.shape == (1,X.shape[1]))

	cache = {
		"Z1":Z1,
		"A1":A1,
		"Z2":Z2,
		"A2":A2
	}

	return A2,cache

# 计算cost
def compute_cost(A2,Y,parameters):
	# 样本数量
	m = Y.shape[1]
	
	# cost
	cost = -(np.sum(Y * np.log(A2) + (1 - Y) * np.log(1-A2))) / m

	cost = np.squeeze(cost)
	return cost

# 反向传播
def backword_propagation(parameters,cache,X,Y):
	m = Y.shape[1]

	W1 = parameters["W1"]
	W2 = parameters["W2"]

	A1 = cache["A1"]
	A2 = cache["A2"]

	dZ2 = A2 - Y
	dW2 = np.dot(dZ2,A1.T) / m
	db2 = np.sum(dZ2,axis = 1 ,keepdims = True) / m
	dZ1 = np.dot(W2.T,dZ2) * (1 - A1 ** 2)
	dW1 = np.dot(dZ1,X.T) / m
	db1 = np.sum(dZ1,axis = 1,keepdims = True) / m

	grads = {
		"dW1":dW1,
		"dW2":dW2,
		"db1":db1,
		"db2":db2
	}
	return grads


# 更新参数
def update_parameters(parameters,grads,learning_rate = 1):
	W1 = parameters["W1"]
	W2 = parameters["W2"]
	b1 = parameters["b1"]
	b2 = parameters["b2"]

	dW1 = grads["dW1"]
	db1 = grads["db1"]
	dW2 = grads["dW2"]
	db2 = grads["db2"]


	# 更新
	W1 = W1 - learning_rate * dW1
	W2 = W2 - learning_rate * dW2
	b1 = b1 - learning_rate * db1
	b2 = b2 - learning_rate * db2

	parameters = {
		"W1":W1,
		"b1":b1,
		"W2":W2,
		"b2":b2
	}

	return parameters

# 整合
def nn_model(X,Y,n_h,num_interations = 10,learning_rate = 0.00001,print_cost = False):
	n_x = layer_sizes(X,n_h,Y)[0]
	n_y = layer_sizes(X,n_h,Y)[2]

	parameters = initialize_parameters(n_x,n_h,n_y)
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	# 迭代
	costs = []
	for i in range(num_interations):
		# 向前传播
		A2,cache = forward_propagation(X,parameters)
		# 计算cost
		cost = compute_cost(A2,Y,parameters)
		# 计算偏导
		grads = backword_propagation(parameters,cache,X,Y)
		# 更新参数
		parameters = update_parameters(parameters,grads,learning_rate)
		# 打印cost
		if print_cost and i % 1 == 0:
			print("cost:"+str(cost)+"   interation times:"+str(i))
		# 记录cost
		costs.append(cost)
	# 返回最终参数
	return parameters,costs


# 预测
def predict(parameters,XT,YT):
	# 向前传播
	A2,cache = forward_propagation(XT,parameters)
	# Y的预测值
	YP = np.array([0 if i <= 0.5 else 1 for i in np.squeeze(A2)])
	# 和测试结果之间的精度差距
	accuracy = format(100 - np.mean(np.abs(YP - YT)) * 100)
	return YP,accuracy

