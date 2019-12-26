'''
@Author: NoserQJH
@LastEditors  : NoserQJH
@Date: 2019-12-23 21:56:40
@LastEditTime : 2019-12-25 10:39:59
@Description:
'''
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

#加载日志数据

ea1 = event_accumulator.EventAccumulator(
    r'./logsBackup/events.out.tfevents.1577182394.c4130-007')
ea1.Reload()
print(ea1.scalars.Keys())
ea2 = event_accumulator.EventAccumulator(
    r'./logsBackup/events.out.tfevents.1577211522.qinjiahao')
ea2.Reload()
a = ea2.scalars
print(ea2.scalars.Keys())

val_acc_1_1 = ea1.scalars.Items('valid_acc_1')
val_acc_1_1_step = [i.step for i in val_acc_1_1]
val_acc_1_1_value = [i.value for i in val_acc_1_1]
val_acc_1_2 = ea1.scalars.Items('valid_acc_2')
val_acc_1_2_step = [i.step for i in val_acc_1_2]
val_acc_1_2_value = [i.value for i in val_acc_1_2]
print(max(val_acc_1_1_value), max(val_acc_1_2_value))

val_acc_2_1 = ea2.scalars.Items('valid_acc_1')
val_acc_2_1_step = [i.step for i in val_acc_2_1]
val_acc_2_1_value = [i.value for i in val_acc_2_1]
val_acc_2_2 = ea2.scalars.Items('valid_acc_2')
val_acc_2_2_step = [i.step for i in val_acc_2_2]
val_acc_2_2_value = [i.value for i in val_acc_2_2]
print(max(val_acc_2_1_value), max(val_acc_2_2_value))

import plotly.offline as py
import plotly.graph_objects as go
figs = [
     go.Scatter(
        x=val_acc_1_1_step,
        y=val_acc_1_1_value,
        mode="lines",
        name='val_acc of Net 1 trained by vDML with teacher'),

    go.Scatter(
        x=val_acc_1_2_step,
        y=val_acc_1_2_value,
        mode="lines",
        name='val_acc of Net 1 trained by vDML without teacher'),
    go.Scatter(
        x=val_acc_2_1_step,
        y=val_acc_2_1_value,
        mode="lines",
        name='val_acc of Net 1 trained by vDML with teacher'),

    go.Scatter(
        x=val_acc_2_2_step,
        y=val_acc_2_2_value,
        mode="lines",
        name='val_acc of Net 1 trained by vDML without teacher'),

]
py.plot(figs, filename='./temp.html', auto_open=True)

plt.show()

