
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# import sklearn
# import sklearn.datasets
# import sklearn.linear_model


# with open('res/x_origin.pickle','rb') as f1:
# 	X = pickle.load(f1)
# with open('res/xt_origin.pickle','rb') as f2:
# 	XT = pickle.load(f2)
# with open('res/Y.pickle','rb') as f3:
# 	Y = pickle.load(f3)
# with open('res/YT.pickle','rb') as f4:
# 	YT = pickle.load(f4)

# tempX = np.array(X).T
# tempXT = np.array(XT).T
# tempY = np.array(Y)
# tempYT = np.array(YT)

# Y = []
# for i in range(len(tempY)):
# 	Y.append([tempY[i]])
# Y = np.array(Y)
# Y = Y.T

# YT = []
# for i in range(len(tempYT)):
# 	YT.append([tempYT[i]])
# YT = np.array(YT)
# YT = YT.T

# X = []
# for i in range(len(tempX)):
# 	line = []
# 	for j in range(len(tempX[i])):
# 		line.append(float(tempX[i][j]))
# 	X.append(line)
# X = np.array(X)

# XT = []
# for i in range(len(tempXT)):
# 	line = []
# 	for j in range(len(tempXT[i])):
# 		line.append(float(tempXT[i][j]))
# 	XT.append(line)
# XT = np.array(XT)


# with open('res/X1.pickle','wb') as f0:
# 	pickle.dump(X,f0)
# with open('res/XT1.pickle','wb') as f1:
# 	pickle.dump(XT,f1)
# with open('res/Y1.pickle','wb') as f2:
# 	pickle.dump(Y,f2)
# with open('res/YT1.pickle','wb') as f3:
# 	pickle.dump(YT,f3)