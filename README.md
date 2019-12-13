<!--
 * @Author: NoserQJH
 * @LastEditors: NoserQJH
 * @Date: 2019-12-11 21:37:14
 * @LastEditTime: 2019-12-13 19:55:06
 * @Description:
 -->
# copied from https://github.com/chxy95/Deep-Mutual-Learning.git
# Dependence
Pytorch 1.0.0
tensorboard 1.14.0
# Overview
Overview of the algorithm:
<img src="https://raw.githubusercontent.com/chxy95/Deep-Mutual-Learning/master/images/Overview.png" width="700"/>
# Usage
The default network for DML is ResNet32.
Train 2 models using DML by main.py:
```
python train.py --model_num 2
```
Use tensorboard to monitor training process on choosing port:
```
tensorboard --logdir logs --port 6006
```
# Result
| Network | ind_avg_acc | Dml_avg_acc|
|---------|:-----------:|:----------:|
|ResNet32 |   69.83%    | **71.03%** |
