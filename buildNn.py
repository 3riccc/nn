import numpy as np


def sigmoid(Z):
	A = 1./(1.+np.exp(-Z))
	cache = Z
	return A,cache

def relu(Z):
	A = np.maximum(0,Z)
	assert(A.shape == Z.shape)
	cache = Z
	return A,cache
def relu_backward(dA, Z):
	"""
	Implement the backward propagation for a single RELU unit.
	Arguments:
	dA -- post-activation gradient, of any shape
	cache -- 'Z' where we store for computing backward propagation efficiently
	Returns:
	dZ -- Gradient of the cost with respect to Z
	"""
	dZ = np.array(dA, copy=True) # just converting dz to a correct object.

	# When z <= 0, you should set dz to 0 as well. 
	dZ[Z <= 0] = 0

	assert (dZ.shape == Z.shape)

	return dZ
def sigmoid_backward(dA, Z):
	"""
	Implement the backward propagation for a single SIGMOID unit.
	Arguments:
	dA -- post-activation gradient, of any shape
	cache -- 'Z' where we store for computing backward propagation efficiently
	Returns:
	dZ -- Gradient of the cost with respect to Z
	"""
	s = 1/(1.+np.exp(-Z))
	dZ = dA * s * (1-s)
	assert (dZ.shape == Z.shape)
	return dZ

# 参数初始化
def initialize_parameters_deep(X,layer_dims):
	# 第一层参数和训练样本中的特征数量相关，因此插入
	layer_dims.insert(0,X.shape[0])
	# 层数
	# 如layer_dims = [2,3,4,5,1]表示
	# 从第一个隐藏层开始，每个隐藏层分别有2,3，4，5，1个神经元
	L = len(layer_dims)
	# 参数
	parameters = {}
	for i in range(1,L):
		# 初始化向量w和b
		parameters["W"+str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1]) * 0.001
		parameters["b"+str(i)] = np.zeros((layer_dims[i],1))

		# 确认形状
		assert(parameters["W"+str(i)].shape == (layer_dims[i],layer_dims[i-1]))
		assert(parameters["b"+str(i)].shape == (layer_dims[i],1))

	return parameters

# 向前传播
def linear_forward(A,W,b):
	Z = np.dot(W,A) + b
	# 确认形状
	assert(Z.shape == (W.shape[0],1))
	cache = (A,W,b)
	return Z,cache

# 向前传播并激活
def linear_activation_forward(A_prev,W,b,activation):
	# 先算出Z
	Z = np.dot(W,A_prev)+b
	# 然后根据不同的激活函数计算A
	if activation == "sigmoid":
		A,activation_cache = sigmoid(Z)
	elif activation == "relu":
		A,activation_cache = relu(Z)
	
	return A,activation_cache

# 在整个模型中向前传播，需要调用前面的函数
def L_model_forward(X,parameters):
	# 准备缓存每一层的Z
	ZS = []
	AS = []
	# 最开始的A=X
	AS.append(X)
	# 层数 
	L = int(len(parameters) / 2)
	# 按层传播
	for i in range(1,L):
		A_prev = AS[len(AS)-1]
		A,Z = linear_activation_forward(A_prev,parameters["W"+str(i)],parameters["b"+str(i)],"relu");
		AS.append(A)
		ZS.append(Z)
	# 最后一层必须用sigmoid
	A_prev = AS[len(AS)-1]
	A,Z = linear_activation_forward(A_prev,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid");
	AS.append(A)
	ZS.append(Z)

	return AS,ZS

# 计算cost
def compute_cost(AL,Y):
	# 样本数量
	m = Y.shape[1]
	cost = -1./m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
	# 压缩维数
	cost = np.squeeze(cost)
	assert(cost.shape == ())

	return cost

# 一个层的反向传播
def backward(dA_after,Z,A,W,activation):
	m = Z.shape[1]
	if activation == "sigmoid":
		# dz，用于计算da_prev,dw,db
		dZ = sigmoid_backward(dA_after,Z)
	else:
		dZ = relu_backward(dA_after,Z)
	# 计算da，dw，db
	dW = 1. / m * np.dot(dZ,A.T)
	db = 1. / m * np.sum(dZ,axis=1,keepdims = True)
	dA_prev = np.dot(W.T,dZ)

	return dA_prev,dW,db

# 计算最后一层的dAL
def last_backward(AL,Y):
	dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
	return dAL



# 整体的反向传播
def L_model_backward(AS,ZS,Y,parameters):
	# 记录导数
	grads = {}
	# 层数,AS包含输入层，所以AS-1是所有神经元层数
	L = len(AS) - 1
	# 最后一层AL
	AL = AS[L]

	# 计算最后一层的导数
	dAL = last_backward(AL,Y)
	# 倒着计算每一层的偏导
	dA_after = dAL
	for i in reversed(range(1,L+1)):
		if(i == L):
			activation = "sigmoid"
		else:
			activation = "relu"
		# 传播
		dA_after,grads["dW"+str(i)],grads["db"+str(i)] = backward(dA_after,ZS[i-1],AS[i-1],parameters["W"+str(i)],activation)
	return grads

# 更新参数
def update_parameters(parameters,grads,learning_rate):
	# 有多少层
	L = int(len(grads) / 2)
	# 一层一层更新
	for i in range(1,L+1):
		parameters["W"+str(i)] = parameters["W"+str(i)] - learning_rate * grads["dW"+str(i)]
		parameters["b"+str(i)] = parameters["b"+str(i)] - learning_rate * grads["db"+str(i)]
	return parameters

# 预测结果是否准确
def predict(parameters,XT,YT):
	# 向前传播
	AS,ZS = L_model_forward(XT,parameters)
	# Y的预测值
	YP = AS[len(AS)-1]
	YP = np.array([0 if i <= 0.5 else 1 for i in np.squeeze(YP)])
	# 和测试结果之间的精度差距
	accuracy = format(100 - np.mean(np.abs(YP - YT)) * 100)
	return YP,accuracy

def nn_model(X,Y,XT,YT,structure,num_interations,learning_rate,print_cost = False,print_accu=False):
	# 初始化
	parameters = initialize_parameters_deep(X,structure)
	# 循环
	costs = []
	accuracies = []
	for i in range(num_interations):
		# 向前传播
		AS,ZS = L_model_forward(X,parameters)
		# 计算cost
		cost = compute_cost(AS[len(AS)-1],Y)
		costs.append(cost)
		# 反向传播
		grads = L_model_backward(AS,ZS,Y,parameters)
		print(grads["dW2"])
		print(grads["db2"])
		# 更新参数
		parameters = update_parameters(parameters,grads,learning_rate)
		# 打印输出 
		if print_cost and i % 10 == 0:
			print("cost:"+str(cost)+"  interation times:"+str(i))
		# 打印测试结果
		if i % 10 == 0:
			YP,accuracy = predict(parameters,XT,YT)
			print("accuracy:"+str(accuracy)+"%")
			accuracies.append(accuracy)
	# 返回最终参数，cost和精度
	return parameters,costs,accuracies