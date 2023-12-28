# 一元回归

import numpy as np
import scipy.stats as stats

x=[150,160,170,180,190,200,210,220,230,240,250,260]
y=[5.58,5.72,6.04,6.34,6.68,6.99,7.27,7.59,7.86,8.10,8.47,8.80]

x=np.array(x)
y=np.array(y)

n=len(x)

x_t=np.sum(x)
y_t=np.sum(y)

# print(x_t,y_t)

x2_t=np.sum(x**2)
y2_t=np.sum(y**2)
xy=np.sum(x*y)

# print(x2_t,y2_t,xy)

x_m=x_t/n
y_m=y_t/n
# print(x_m,y_m)
b_hat=(xy-n*x_m*y_m)/(x2_t-n*x_m**2)
b_hat=round(b_hat,4)
a_hat=round(y_m-b_hat*x_m,4)
if b_hat>=0:
    print(f"经验回归方程为：y={a_hat}+{b_hat}x")
else:
    print(f"经验回归方程为：y={a_hat}{b_hat}x")


# sigma的估计

sigma_hat_2=(np.sum((y-y_m)**2)-(b_hat**2)*np.sum((x-x_m)**2))/n
# print(round(sigma_hat_2,4))

# 检验是否显著

alpha=0.05

sigma_star_hat_2=n/(n-2)*sigma_hat_2
sigma_star_hat_2=round(sigma_star_hat_2,4)
# print(sigma_star_hat_2)
lxx=np.sum((x-x_m)**2)

t=b_hat/np.sqrt(sigma_star_hat_2)*np.sqrt(lxx)
t=round(t,4)

print(f"统计量t={t}")
t_value=stats.t.ppf(1-alpha/2,n-2)
t_value=round(t_value,4)
print(f"查表得到的t={t_value}")

# 经验相关系数

lxy=np.sum((x-x_m)*(y-y_m))
lyy=np.sum((y-y_m)**2)


r=round(lxy/(np.sqrt(lxx*lyy)),4)
print(f'经验相关系数r={r}')


# 预测
x0=195
y0_hat=a_hat+b_hat*x0
# print(y0_hat)
delta=round(t_value*np.sqrt(sigma_star_hat_2)*np.sqrt(1+1/n+((x0-x_m)**2)/lxx),3)
# print(delta)

print(f'置信度为{(1-alpha)*100}%预测区间为({y0_hat-delta},{y0_hat+delta})')
