# 双因素方差分析
# 等重复实验，无重复实验

import numpy as np
import scipy.stats as stats

alpha=0.05

# 等重复实验,一共有rsl个实验结果
# alpha称为水平A的指标效应
# beta称为水平B的指标效应
# gama称为组合水平A和B对指标的交互效应

# datas=[
#     [[15,15,17],[19,19,16],[16,18,21]],
#     [[17,17,17],[15,15,15],[19,22,22]],
#     [[15,17,16],[18,17,16],[18,18,18]],
#     [[18,20,22],[15,16,17],[17,17,17]],
# ]
# 
# 无重复分析
# datas=[
#     [[258],[279],[242]],
#     [[302],[314],[336]],
#     [[321],[318],[327]]
# ]
# 4.6
datas=[
    [[32],[35],[35.5],[38.5]],
    [[33.5],[36.5],[38],[39.5]],
    [[36],[37.5],[39.5],[43]],
]
# 4.9
datas=[
    [[71,73],[72,73],[75,73],[77,75]],
    [[73,75],[76,74],[78,77],[74,74]],
    [[76,73],[79,77],[74,75],[74,73]],
    [[75,73],[73,72],[70,71],[69,69]]
]

datas=np.array(datas)
r=len(datas)
s=len(datas[0])
l=len(datas[0][0])

m_data=np.zeros((r,s))
x_r=[]
x_s=[]

da_list=np.reshape(datas,(1,-1))
x_m=round(np.mean(da_list[0]),3)
# print(x_m)

for i in range(r):
    for j in range(s):
        m_data[i][j]=round(np.mean(datas[i][j]),3)

# print(m_data)

for i in range(r):
    x_r.append(round(np.mean(m_data[i]),3))

# print(x_r)

m_data_T=m_data.T
for j in range(s):
    x_s.append(round(np.mean(m_data_T[j]),3))

# print(x_s)


QA=s*l*np.sum((x_r-x_m)**2)
# print(QA)
QA_m=QA/(r-1)

QB=r*l*np.sum((x_s-x_m)**2)
# print(QB)
QB_m=QB/(s-1)

x_r=np.reshape(x_r,(r,1))
x_s=np.reshape(x_s,(1,s))


QI=l*np.sum((m_data-x_r-x_s+x_m)**2)
QI_m=QI/((r-1)*(s-1))
# print(QI)

if l!=1:
    m_data=np.reshape(m_data,(r,s,1))
    QE=np.sum((datas-m_data)**2)
    # print(QE)
    QE_m=QE/((r*s*(l-1)))

if l!=1:
    FA=QA_m/QE_m
    FB=QB_m/QE_m
    FI=QI_m/QE_m
else:
    FA=QA_m/QI_m
    FB=QB_m/QI_m


if l!=1:
    QT=QA+QB+QI+QE
    msg=f"""
    方差来源 |  平方和  | 自由度 |  均方和  |  F值  |
    因素A   |   {QA:.2f}    |   {r-1}   |   {QA_m:.2f}  |   {FA:.2f}    |
    因素B   |   {QB:.2f}    |   {s-1}   |   {QB_m:.2f}  |   {FB:.2f}    |
    交互作用I|  {QI:.2f}    |   {(r-1)*(s-1)}   |  {QI_m:.2f}  |   {FI:.2f}    |
    误差    |   {QE:.2f}    |   {r*s*(l-1)}     |   {QE_m:.2f}  |
    总和    |   {QT:.2f}    |   {r*s*l-1}   |
    """
    fa_value=stats.f.ppf(1-alpha,r-1,r*s*(l-1))
    fb_value=stats.f.ppf(1-alpha,s-1,r*s*(l-1))
    fi_value=stats.f.ppf(1-alpha,(r-1)*(s-1),r*s*(l-1))
    print(f"fa={round(fa_value,2)},fb={round(fb_value,2)},fi={round(fi_value,2)}")

else:
    QT=QA+QB+QI
    msg=f"""
    方差来源 |  平方和  | 自由度 |  均方和  |  F值  |
    因素A   |   {QA:.2f}    |   {r-1}   |   {QA_m:.2f}  |   {FA:.2f}    |
    因素B   |   {QB:.2f}    |   {s-1}   |   {QB_m:.2f}  |   {FB:.2f}    |
    交互作用I|  {QI:.2f}    |   {(r-1)*(s-1)}   |  {QI_m:.2f}  |
    总和    |   {QT:.2f}    |   {r*s*l-1}   |
    """
    fa_value=stats.f.ppf(1-alpha,r-1,(r-1)*(s-1))
    fb_value=stats.f.ppf(1-alpha,s-1,(r-1)*(s-1))
    print(f"fa={round(fa_value,2)},fb={round(fb_value,2)}")

print(msg)
