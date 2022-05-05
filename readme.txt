mlTSK toolbox: Optimize TSK fuzzy system for regression and classification on big data
% ml: matlab, or machine learning
% big data: large-scale and large-volume

文件夹包含：
    batchNormalizationBackward.m 计算BN的导数
    calculateDeltaLY.m 计算回归或分类损失相对于预测标签的导数
    calculateFiringLevel.m 使用隶属度函数计算规则激活度
    EIASC.m 区间二型模糊降阶算法
    FuzzyCMeans.m 模糊C均值聚类
    fis2mat.m 将sugfis对象转换为模糊系统矩阵
    mat2fis.m 将模糊系统矩阵转换为sugfis对象    
    mlTSK.m 主程序，TSK模糊系统优化算法（默认为CDR-FCM-RDpA，见第54行）
    mlTSK.mlapp 主程序图形化界面
    Musk1.mat 示例回归数据集
    Musk1_result.PNG 示例回归可视化结果
    Musk1C.mat 示例分类数据集    
    Musk1C_result.PNG 示例分类可视化结果    
    readme.txt 说明文档

验证环境：MATLAB 版本: 9.8.0.1323502 (R2020a)

验证步骤：打开Matlab，运行mlTSK.m，得到输出结果：
    载入Musk1.mat时，得到回归结果：
        train on 333 samples, tune on 71 samples, test on 72 samples, num. of features is 166.
        Regression Task.
        Iteration: 640, trainRMSE: 0.22, tuneRMSE: 0.38, testRMSE: 0.63.
    载入Musk1C.mat时，得到分类结果：
        train on 333 samples, tune on 71 samples, test on 72 samples, num. of features is 166.
        Classification Task, num. of class is 2.
        Iteration: 320, trainBCA: 0.97, tuneBCA: 0.98, testBCA: 0.90.

图形化界面：打开Matlab，打开mlTSKapp.mlapp并运行，得到图形化界面，点击Load Data and Run按钮载入数据，得到输出结果：                   
    载入Musk1.mat时，得到回归结果，如图Musk1_result.png所示。
    载入Musk1C.mat时，得到分类结果，如图Musk1C_result.png所示。

已实现TSK模糊系统优化算法包括但不限于（详见mlTSK.m第55行至第73行）：
    MBGD-RDA [1]
    FCM-RDpA [2]
    FCM-RDpA-UR-BNC [3]
    FCM-RDpA-LN-ReLU
    CDR(P)-FCM-RDpA
    CFS(P)-FCM-RDpA

参考文献：
[1] D. Wu, Y. Yuan, J. Huang, and Y. Tan, “Optimize TSK fuzzy systems for regression problems: Mini-batch gradient descent with regularization, DropRule, and AdaBound (MBGD-RDA),” IEEE Trans. on Fuzzy Systems, vol. 28, no. 5, pp. 1003–1015, 2020.
[2] Z. Shi, D. Wu, C. Guo, C. Zhao, Y. Cui, and F.-Y. Wang, “FCM-RDpA: TSK fuzzy regression model construction using fuzzy c-means clustering, regularization, DropRule, and Powerball AdaBelief,” Information Sciences, vol. 574, pp. 490–504, 2021.
[3] Y. Cui, D. Wu, and J. Huang, “Optimize TSK fuzzy systems for classification problems: Mini-batch gradient descent with uniform regularization and batch normalization,” IEEE Trans. on Fuzzy Systems, vol. 28, no. 12, pp. 3065–3075, 2020.






              