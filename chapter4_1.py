import numpy as np
import scipy.stats as stats

# 单因素分析
alpha=0.05
# 例题
# datas=[
#     [2,4,3,2,4,7,7,2,5],
#     [5,6,8,5,10,7,12,6,6],
#     [7,11,6,6,7,9,5,10,6],
#     ]
# 4.2
# datas=[
#     [65,60,69,79,38,68,54,67,68,43],
#     [74,71,58,49,58,49,48,68,56,47],
#     [22,34,24,21,20,36,36,31,28,33]
# ]
# 4.4
datas=[
    [29.6,24.3,28.5,32.0],
    [27.3,32.6,30.8,34.8],
    [5.8,6.2,11.0,8.3],
    [21.6,17.4,18.3,19.0],
    [29.2,32.8,25.0,24.2]
]

r=len(datas)
ns=[]
for d in datas:
    ns.append(len(d))


m_ds=[]
for d in datas:
    da=np.array(d)
    m_d=np.mean(da)
    m_ds.append(m_d)
mean_d=np.mean(np.array(m_ds))
print(mean_d)

n=np.sum(ns)
print(f"r={r}")
print(f'n={n}')

QA=round(np.sum(ns*(m_ds-mean_d)**2),3)
QA_m=round(QA/(r-1),3)
print(f"误差QA={QA},自由度r-1={r-1},均方和={QA_m}")
print("-------------------------")

QE=0
for i in range(r):
    QE+=np.sum((datas[i]-m_ds[i])**2)
QE=round(QE,3)
QE_m=round(QE/(n-r),3)
print(f'误差QE={QE},自由度n-r={n-r},均方和={QE_m}')
print("--------------------------")
QT=QA+QE
print(f"总和QT={QT},自由度n-1={n-1}")
print('--------------------------')
F=round(QA_m/QE_m,3)
print(f'F={F}')
print('--------------------------')
print(f"""
方差来源 |  平方和  | 自由度 | 均方和 |  F值  |
因素A    | QA={QA} | {r-1} | {QA_m} |  {F}  |
因素B    | QE={QE} | {n-r} | {QE_m} |       |
总和T    | QT={QT} | {n-1} |        |       |
""")

f_value=stats.f.ppf(1-alpha,r-1,n-r)
print(round(f_value,2))

