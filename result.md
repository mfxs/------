+ **Multi-rate Soft Sensor Experiment**
|      Method       | QV1 (R2 (RMSE) ) | QV2(R2 (RMSE) ) |
| :---------------: | :--------------: | :-------------: |
|        MLP        |        /         |        /        |
|       L-MLP       |        /         |        /        |
|    Vanilla RNN    |        /         |        /        |
| **CW-RNN (ours)** |      **/**       |      **/**      |

> + MLP——只使用少量包含完整变量的样本建立MLP
> + L-MLP——使用lifting技巧将过程变量堆叠后建立MLP
> + Vanilla RNN——将缺少的变量用0替代后建立RNN
> + CW-RNN——根据每一时刻所包含变量对隐层特征进行周期更新（所提方法）

