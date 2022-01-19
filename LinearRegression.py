import numpy as np 
import matplotlib.pyplot as plt
import csv

#ReadDataFile
with open('C:\\Users\\admin\\Desktop\Hoctap\\TrainingData\\Fish.csv') as f:
    reader = csv.reader(f,delimiter='\t')
    l = [row for row in reader]
ListData = np.asarray(l)
#WeightData
Y =np.array([float(x) for x in ListData[1:35,0]]).T.reshape(34,1)
#LengthData
X1=np.array([float(x) for x in ListData[1:35,1]]).T.reshape(34,1)
#HeightData
X2=np.array([float(x) for x in ListData[1:35,2]]).T.reshape(34,1)
#TestData
Predict = np.array([float(x) for x in ListData[35,:]])

# Building Xbar 
one = np.ones((X1.shape[0],1))
Xbar = np.concatenate((one, X1, X2), axis = 1)

# Calculating weights of BreamFish
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, Y)
w = np.dot(np.linalg.pinv(A), b)
print ('w = ', w)

#DrawLinearRegressionPlot
def fun(x, y, w):
    w0, w1, w2 = w[0], w[1], w[2]
    return (x*w1+y*w2+w0)

#Drawing the 3D-Plot
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X1[:], X2[:], Y[:], color='k', zorder=15, marker='o', alpha=0.5)
ax.scatter(Predict[1], Predict[2], Predict[0], color='r', zorder=15, marker='s', alpha=0.5, label ='Weight_test')

x = np.linspace(30, 50, 30)      
y = np.linspace(10, 20, 30) 
X, Y = np.meshgrid(x, y)
Z_Predict = np.array(fun(np.ravel(X), np.ravel(Y),w))
Z_Predict = Z_Predict.reshape(X.shape)
ax.plot_surface(X, Y, Z_Predict, color ='c')
ax.scatter(Predict[1], Predict[2], (Predict[1]*w[1]+Predict[2]*w[2]+w[0]), color='k', zorder=15, marker='x', alpha=1, label ='Weight_Predict')
ax.legend()
ax.set_xlabel('Length')
ax.set_ylabel('Height')
ax.set_zlabel('Weight')
plt.title(f"Linear Regression predict weight of Bream Fish by height and length \n Length_test, Height_test = {Predict[1], Predict[2]} \n Weight_Predict = {(Predict[1]*w[1]+Predict[2]*w[2]+w[0])[0]}, Weight_test = {Predict[0]} \n Error = {abs((Predict[0]-(Predict[1]*w[1]+Predict[2]*w[2]+w[0]))*100/Predict[0])[0]}%")

Command = input("Rotate or not: (Y/N)\n")
if Command == "Y" or Command == "y":
    for angle in range(0, 360):
        ax.view_init(20, angle)
        plt.draw()
        plt.pause(.001)
else:
    plt.show()