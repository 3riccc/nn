import numpy as np
import matplotlib as plt

import pickle

with open('res/x_origin.pickle','rb') as f1:
	X = pickle.load(f1)
with open('res/xt_origin.pickle','rb') as f2:
	XT = pickle.load(f2)
with open('res/Y.pickle','rb') as f3:
	Y = pickle.load(f3)
with open('res/YT.pickle','rb') as f4:
	YT = pickle.load(f4)

tempX = np.array(X).T
tempXT = np.array(XT).T
tempY = np.array(Y)
tempYT = np.array(YT)

Y = []
for i in range(len(tempY)):
	Y.append([tempY[i]])
Y = np.array(Y)
Y = Y.T

YT = []
for i in range(len(tempYT)):
	YT.append([tempYT[i]])
YT = np.array(YT)
YT = YT.T

X = []
for i in range(len(tempX)):
	line = []
	for j in range(len(tempX[i])):
		line.append(float(tempX[i][j]))
	X.append(line)
X = np.array(X)

XT = []
for i in range(len(tempXT)):
	line = []
	for j in range(len(tempXT[i])):
		line.append(float(tempXT[i][j]))
	XT.append(line)
XT = np.array(XT)


# ------------------------------以上获取并修正数据---------------------------


# sigmoid
def sigmoid(z):
	s = 1./(1. + np.exp(-z))
	return s

def ReLU(z):
	s = []
	for i in len(z):
		if z[i] > 0:
			s.append(z[i])
		else:
			z.append(0)
	return np.array(s)

def initialize_with_zeros(dim):
	w = np.zeros((dim,1))
	b = 0.

	return w,b

# 向前传播:根据w，b和x的运算，计算a(activation)，然后再用a和y计算cost并且计算各个参数对应的偏导
def propagate(w,b,X,Y):
	# 样本数量
	m = X.shape[1]

	# 激活函数
	A = sigmoid(np.dot(w.T,X) + b)

	# 计算cost
	cost = -np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))/m

	# 计算w和b的导数
	dw = np.dot(X,(A-Y).T) / m
	db = np.sum(A-Y) / m

	# 检测数据格式是否正确
	assert(dw.shape == w.shape)
	assert(db.dtype == float)
	# 确认cost是实数
	cost = np.squeeze(cost)
	assert(cost.shape == ())

	grads = {
		"dw":dw,
		"db":db
	}
	return grads,cost

# 训练
def optimeze(w,b,X,Y,n_iterations,learning_rate,print_cost = False):
	costs = []
	# 迭代
	for i in range(n_iterations):
		# 向前传播
		grads,cost = propagate(w,b,X,Y)
		
		# cost
		costs.append(cost)
		if print_cost and i % 100 == 0:
			print("cost:"+str(cost)+"  iteration times:"+str(i))
		
		# 反相传播
		dw = grads["dw"]
		db = grads['db']

		w = w - learning_rate * dw
		b = b - learning_rate * db

	# 记录最终参数
	params = {
		"w":w,
		"b":b
	}

	return params,costs

# 预测函数
def predict(w,b,X):
	# 样本数量
	m = X.shape[1]
	A = sigmoid(np.dot(w.T,X) + b)
	# 每一个样本一个确定的值
	y_prediction = np.zeros((1,m))
	for i in range(m):
		if A[0][i] > 0.5:
			y_prediction[0][i] = 1
		else:
			y_prediction[0][i] = 0
	return y_prediction


# 
w,b = initialize_with_zeros(X.shape[0])
params,cost = optimeze(w,b,X,Y,20000,0.0000000000005,True)

YP = predict(params['w'],params['b'],XT)

print("最终计算精度:"+str(format(100 - np.mean(np.abs(YP - YT)) * 100)))
