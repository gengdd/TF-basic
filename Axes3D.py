import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 

x=np.arange(-2,2,0.1)
y=np.arange(-2,2,0.1)
print(x)
print(y)

x,y=np.meshgrid(x,y)
print('扩充x:',x)
print('扩充y:',y)

def fun1(x,y):
    return np.cos(x*np.sin(y))

def fun2(x,y):
    return np.power(x,2)+np.power(y,2)

fig1=plt.figure()
fig2=plt.figure()

ax1=Axes3D(fig1)
ax2=Axes3D(fig2)

z1=fun1(x,y)
z2=fun2(x,y)

ax1.plot_surface(x,y,z1,rstride=1,cstride=1,cmap=plt.cm.coolwarm)
ax1.set_xlabel('x lable',color='r')
ax1.set_ylabel('y lable',color='g')
ax1.set_zlabel('z1 lable',color='b')
ax1.set_xticks(())
ax1.set_yticks(())
ax1.set_zticks(())

ax2.plot_surface(x,y,z2,rstride=1,cstride=1,cmap=plt.cm.coolwarm)
ax2.set_xlabel('x lable',color='r')
ax2.set_ylabel('y lable',color='g')
ax2.set_zlabel('z2 lable',color='b')
ax2.set_xticks(())
ax2.set_yticks(())
ax2.set_zticks(())

plt.show()