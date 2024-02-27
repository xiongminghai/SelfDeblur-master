import os
import re
import matplotlib.pyplot as plt

import numpy as np
import openpyxl as op

def draw(x,y,title):
    plt.plot(x, y, 'y-')
    plt.title('UNSPERVISED '+title)
    plt.xlabel('Iteration')
    plt.ylabel(title)
    plt.show()
    print("The best "+title+" is "+str(y.max())+"  "+str(y.argmax()))
    return y.argmax()

avg_psnr=0
avg_psnr_sm=0
testData=[]

# for i in range(1,21):

with open('./results/polyu_blur/15_mean_blur1_logs.txt') as f:
    iteration =[]
    loss_mse = []
    loss_niqe = []
    loss_pi = []
    psnr = []
    SX = []
    SY = []
    Q1 = []
    SX_t = []
    loss_nrqm = []
    loss_clipiqa = []
    count=0
    for line in f.readlines():
        count=count+1
        if count>=50:
            a = []
            a = re.findall("\d+\.?\d*", line)
            a = list(map(float, a))
            loss_mse.append(a[2]*10000)
            iteration.append(a[0])
            loss_niqe.append(a[3])
            loss_pi.append(a[4])
            psnr.append(a[1])
            Q1.append(a[8])
            SX.append(a[9])
            SY.append(a[10])
            SX_t.append(a[13])
            loss_nrqm.append(a[5])
            loss_clipiqa.append(a[6])


f.close()
iteration = np.array(iteration)
loss_mse = np.array(loss_mse)
loss_niqe = np.array(loss_niqe)
loss_pi=np.array(loss_pi)
psnr=np.array(psnr)
SX=np.array(SX)
SY=np.array(SY)
Q1=np.array(Q1)
SX_t=np.array(SX_t)
loss_nrqm=np.array(loss_nrqm)
loss_clipiqa=np.array(loss_clipiqa)
# avg_psnr+=psnr.max()
# avg_psnr_sm+=psnr_sm.max()
# testData.append([psnr.max(),psnr_sm.max()])
draw(iteration,loss_mse,'loss_mse')
draw(iteration,loss_niqe,'loss_niqe')
draw(iteration,loss_pi,'loss_pi')
draw(iteration,psnr,'psnr')
draw(iteration,SX,'SX')
draw(iteration,SY,'SY')
draw(iteration,Q1,'Q1')
draw(iteration,SX_t,'SX_t')
draw(iteration,loss_nrqm,'loss_nrqm')
draw(iteration,loss_clipiqa,'loss_clipiqa')

# wb = op.Workbook()  # 创建工作簿对象
# ws = wb['Sheet']  # 创建子表
# ws.append(['PSNR', 'PSNR_SM'])
# for i in range(0,len(testData)):
#     ws.append(testData[i])
# wb.save('result.xlsx')
#
# avg_psnr=avg_psnr/20
# avg_psnr_sm=avg_psnr_sm/20

# print('The avg_psnr is %.2f,The avg_psnr_sm is %.2f'%(avg_psnr,avg_psnr_sm))
