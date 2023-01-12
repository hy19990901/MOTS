import geatpy as ea  # import geatpy
import numpy as np
import os

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）

        M = 2  # 初始化M（目标维数）

        maxormins = [-1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）

        # f = open(r'./EAParameters/Dim.txt', 'r')
        Dim = 10000  # 决策变量维数 x_select.shape[0]  决策变量维度其实仍然是1000维  但挨着的几个维度是某个聚类中的样本的优先级表示
        # f.close()  # 初始化Dim（决策变量维数）

        varTypes = [0] * Dim  # 这是一个list,初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [i - i for i in range(Dim)]  # 决策变量下界
        ub = [i - i + 100 for i in range(Dim)]  # 决策变量上界
        lbin = [i - i + 1 for i in range(Dim)]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [i - i + 1 for i in range(Dim)]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        # 50 * 10000 的矩阵
        f = open('./EAParameters/A.txt', 'w')
        for i in range(len(Vars)):
            temp = list(Vars[i])
            temp = [str(val) for val in temp]
            writeline = '\t'.join(temp)
            f.write(writeline + "\n")
        f.close()

        os.system("python ./objectives.py")

        f = open(r'./EAParameters/EAobjective.txt', 'r')
        lines = f.readlines()
        vector = []

        for line in lines:
            objectives = line[:-1].split("\t")
            # print(objectives)
            objectives = [float(val) for val in objectives]
            vector.append(objectives)

        f.close()

        pop.ObjV = np.array(vector)



def RunEA():
    # f = open(r'./EAParameters/selectnums.txt', 'w')
    # f.write(str(select_nums))
    # f.close()

    """================================实例化问题对象============================="""
    problem = MyProblem()  # 生成问题对象
    """==================================种群设置================================"""
    Encoding = 'RI'  # 编码方式
    NIND = 50  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置==============================="""

    myAlgorithm = ea.moea_NSGA2_templet(problem, population)  # 实例化一个算法模板对象`
    myAlgorithm.MAXGEN = 50  # 最大进化代数
    myAlgorithm.logTras = 1  # 设置每多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = True  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """==========================调用算法模板进行种群进化=========================
    调用run执行算法模板，得到帕累托最优解集NDSet以及最后一代种群。NDSet是一个种群类Population的对象。
    NDSet.ObjV为最优解个体的目标函数值；NDSet.Phen为对应的决策变量值。
    """
    [NDSet, population] = myAlgorithm.run()  # 执行算法模板，得到非支配种群以及最后一代种群
    NDSet.save()  # 把非支配种群的信息保存到文件中
    """==================================输出结果=============================="""
    print('用时：%f 秒' % myAlgorithm.passTime)
    print('评价次数：%d 次' % myAlgorithm.evalsNum)
    print('非支配个体数：%d 个' % NDSet.sizes) if NDSet.sizes != 0 else print('没有找到可行解！')
    # if myAlgorithm.log is not None and NDSet.sizes != 0:
    #     print('GD', myAlgorithm.log['gd'][-1])
    #     print('IGD', myAlgorithm.log['igd'][-1])
    #     print('HV', myAlgorithm.log['hv'][-1])
    #     print('Spacing', myAlgorithm.log['spacing'][-1])
    #     """=========================进化过程指标追踪分析========================="""
    #     metricName = [['igd'], ['hv']]
    #     Metrics = np.array([myAlgorithm.log[metricName[i][0]] for i in range(len(metricName))]).T
    #     # 绘制指标追踪分析图
    #     ea.trcplot(Metrics, labels=metricName, titles=metricName)

    f = open(r'./EAParameters/selectsize.txt', 'r')
    line = f.readlines()[0]
    f.close()
    selectsize = int(line)
    writebasepath_name = r'./EA_Result_50'
    os.rename(r'./Result', r'./' + str(selectsize))
    import shutil
    # shutil.move(r'./' + str(selectsize), writebasepath_name + "/" + "mnist_fgsm_" + str(selectsize))
    shutil.move(r'./' + str(selectsize), writebasepath_name + "/" + "cifar_bim-a" + "_" + str(selectsize))
