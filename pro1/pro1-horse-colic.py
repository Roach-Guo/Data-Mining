# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 16:41:45 2017

@author: Roach
"""
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
import statsmodels.api as sm

#==============================================================================
# 准备数据
#==============================================================================
# 属性
attribute = ["surgery","Age","Hospital Number","rectal temperature"," pulse",
"respiratory rate","temperature of extremities","peripheral pulse","mucous membranes",
"capillary refill time","pain","peristalsis","abdominal distension","nasogastric tube",
"nasogastric reflux","nasogastric reflux PH","rectal examination","abdomen",
"packed cell volume","total protein","abdominocentesis appearance",
"abdomcentesis total protein","outcome","surgical lesion","t1","t2","t3","cp_data"]
# 数值属性
name_value = ["rectal temperature"," pulse","respiratory rate","nasogastric reflux PH",
"packed cell volume","total protein","abdomcentesis total protein"]
# 标称属性
name_category = ["surgery","Age","Hospital Number","temperature of extremities",
"peripheral pulse","mucous membranes","capillary refill time","pain","peristalsis",
"abdominal distension","nasogastric tube","nasogastric reflux","rectal examination",
"abdomen","abdominocentesis appearance","outcome","surgical lesion","cp_data"]
# 读取数据
data_origin = pd.read_csv("horse-colic.csv",names = attribute,na_values = "?")
# 将字符数据转换为category
for item in name_category:
    data_origin[item] = data_origin[item].astype('category')
#==============================================================================    

    
#==============================================================================
#                               1 数据摘要
#==============================================================================
# ----------------1.1 对标称属性，给出每个可能取值的频数--------------------------

# 使用value_counts函数统计每个标称属性的取值频数
for item in name_category:
    print (item, '的频数为：\n', pd.value_counts(data_origin[item].values), '\n')

# -------1.2 对数值属性，给出最大、最小、均值、中位数、四分位数及缺失值的个数--------

# 最大值
data_show = pd.DataFrame(data = data_origin[name_value].max(), columns = ['max'])
# 最小值
data_show['min'] = data_origin[name_value].min()
# 均值
data_show['mean'] = data_origin[name_value].mean()
# 中位数
data_show['median'] = data_origin[name_value].median()
# 四分位数
data_show['quartile'] = data_origin[name_value].describe().loc['25%']
# 缺失值个数
data_show['missing'] = data_origin[name_value].describe().loc['count'].apply(lambda x : 368-x)

print(data_show)
#==============================================================================


#==============================================================================
#                                  2 数据可视化
#==============================================================================
# --------------------------------2.1 绘制直方图--------------------------------
fig = plt.figure(figsize = (20,20))
i = 1
for item in name_value:
    ax = fig.add_subplot(2, 4, i)
    data_origin[item].plot(kind = 'hist', title = item, ax = ax)
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
fig.savefig('histogram.jpg')

# ---------------------------------2.2 绘制qq图---------------------------------
fig = plt.figure(figsize = (20,20))
i = 1
for item in name_value:
    ax = fig.add_subplot(2, 4, i)
    sm.qqplot(data_origin[item], ax = ax)
    ax.set_title(item)
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
fig.savefig('qqplot.jpg')

# ---------------------------------2.3 绘制盒图---------------------------------
fig = plt.figure(figsize = (20,20))
i = 1
for item in name_value:
    ax = fig.add_subplot(2, 4, i)
    data_origin[item].plot(kind = 'box')
    i += 1
fig.savefig('boxplot.jpg')
#==============================================================================


#==============================================================================
#               3 处理缺失数据并且可视化的与旧数据集进行比较
#==============================================================================
nan_list = pd.isnull(data_origin).any(1).nonzero()[0]
# ---------------------------3.1 将缺失部分剔除---------------------------------
# 将缺失值对应的数据整条剔除，生成新数据集
data_filtrated = data_origin.dropna()
# 绘制可视化图
fig = plt.figure(figsize = (20,20))
i = 1
# 对标称属性，绘制折线图
for item in name_category:
    ax = fig.add_subplot(5, 5, i)
    ax.set_title(item)
    pd.value_counts(data_origin[item].values).plot(ax = ax, marker = '^', label = 'origin', legend = True)
    pd.value_counts(data_filtrated[item].values).plot(ax = ax, marker = 'o', label = 'filtrated', legend = True)
    i += 1
    
i = 19
# 对数值属性，绘制直方图
for item in name_value:
    ax = fig.add_subplot(5, 5, i)
    ax.set_title(item)
    data_origin[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'origin', legend = True)
    data_filtrated[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'filtrated', legend = True)
    ax.axvline(data_origin[item].mean(), color = 'r')
    ax.axvline(data_filtrated[item].mean(), color = 'b')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
# 保存图像和处理后数据
fig.savefig('missing_data_delete.jpg')
data_filtrated.to_csv('missing_data_delete.csv', mode = 'w', encoding='utf-8', index = False,header = False)

# ----------------------3.2 用最高频率值来填补缺失值-----------------------------
# 建立原始数据的拷贝
data_filtrated = data_origin.copy()
# 对每一列数据，分别进行处理
for item in attribute:
    # 计算最高频率的值
    most_frequent_value = data_filtrated[item].value_counts().idxmax()
    # 替换缺失值
    data_filtrated[item].fillna(value = most_frequent_value, inplace = True)

    # 绘制可视化图
fig = plt.figure(figsize = (20,20))

i = 1
# 对标称属性，绘制折线图
for item in name_category:
    ax = fig.add_subplot(5, 5, i)
    ax.set_title(item)
    pd.value_counts(data_origin[item].values).plot(ax = ax, marker = '^', label = 'origin', legend = True)
    pd.value_counts(data_filtrated[item].values).plot(ax = ax, marker = 'o', label = 'filtrated', legend = True)
    i += 1    

i = 19
# 对数值属性，绘制直方图
for item in name_value:
    ax = fig.add_subplot(5, 5, i)
    ax.set_title(item)
    data_origin[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'origin', legend = True)
    data_filtrated[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'droped', legend = True)
    ax.axvline(data_origin[item].mean(), color = 'r')
    ax.axvline(data_filtrated[item].mean(), color = 'b')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

# 保存图像和处理后数据
fig.savefig('missing_data_most.jpg')
data_filtrated.to_csv('missing_data_most.csv', mode = 'w', encoding='utf-8', index = False,header = False)
# ---------------------3.3 通过属性的相关关系来填补缺失值-------------------------
# 使用pandas中Series的***interpolate()***函数，对数值属性进行插值计算，并替换缺失值。

# 建立原始数据的拷贝
data_filtrated = data_origin.copy()
# 对数值型属性的每一列，进行插值运算
for item in name_value:
    data_filtrated[item].interpolate(inplace = True)

    # 绘制可视化图
fig = plt.figure(figsize = (20,20))

i = 1
# 对标称属性，绘制折线图
for item in name_category:
    ax = fig.add_subplot(5, 5, i)
    ax.set_title(item)
    pd.value_counts(data_origin[item].values).plot(ax = ax, marker = '^', label = 'origin', legend = True)
    pd.value_counts(data_filtrated[item].values).plot(ax = ax, marker = 'o', label = 'filtrated', legend = True)
    i += 1   
    
i = 19
# 对数值属性，绘制直方图
for item in name_value:
    ax = fig.add_subplot(5, 5, i)
    ax.set_title(item)
    data_origin[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'origin', legend = True)
    data_filtrated[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'droped', legend = True)
    ax.axvline(data_origin[item].mean(), color = 'r')
    ax.axvline(data_filtrated[item].mean(), color = 'b')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

# 保存图像和处理后数据
fig.savefig('missing_data_corelation.jpg')
data_filtrated.to_csv('missing_data_corelation.csv', mode = 'w', encoding='utf-8', index = False,header = False)

# ---------------------3.4 通过对象之间的相似性填补缺失值-------------------------

# 首先将缺失值设为0，对数据集进行正则化。然后对每两条数据进行差异性计算（分值越高差异性越大）。计算标准为：标称数据不相同记为1分，数值数据差异性分数为数据之间的差值。在处理缺失值时，找到和该条数据对象差异性最小（分数最低）的对象，将最相似的数据条目中对应属性的值替换缺失值。
# 建立原始数据的拷贝，用于正则化处理
data_norm = data_origin.copy()
# 将数值属性的缺失值替换为0
data_norm[name_value] = data_norm[name_value].fillna(0)
# 对数据进行正则化
data_norm[name_value] = data_norm[name_value].apply(lambda x : (x - np.mean(x)) / (np.max(x) - np.min(x)))

# 构造分数表
score = {}
range_length = len(data_origin)
for i in range(0, range_length):
    score[i] = {}
    for j in range(0, range_length):
        score[i][j] = 0    
test = 0;
# 在处理后的数据中，对每两条数据条目计算差异性得分，分值越高差异性越大
for i in range(0, range_length):
    for j in range(i, range_length):
        for item in name_category:
            if data_norm.iloc[i][item] != data_norm.iloc[j][item]:
                score[i][j] += 1
        for item in name_value:
            temp = abs(data_norm.iloc[i][item] - data_norm.iloc[j][item])
            score[i][j] += temp
        score[j][i] = score[i][j]
    test = i

# 建立原始数据的拷贝
data_filtrated = data_origin.copy()


# 对有缺失值的条目，用和它相似度最高（得分最低）的数据条目中对应属性的值替换
for index in nan_list:
    best_friend = sorted(score[index].items(), key=operator.itemgetter(1), reverse = False)[1][0]
    for item in name_value:
        if pd.isnull(data_filtrated.iloc[index][item]):
            if pd.isnull(data_origin.iloc[best_friend][item]):
                data_filtrated.ix[index, item] = data_origin[item].value_counts().idxmax()
            else:
                data_filtrated.ix[index, item] = data_origin.iloc[best_friend][item]

# 绘制可视化图
fig = plt.figure(figsize = (20,20))

i = 1
# 对标称属性，绘制折线图
for item in name_category:
    ax = fig.add_subplot(5, 5, i)
    ax.set_title(item)
    pd.value_counts(data_origin[item].values).plot(ax = ax, marker = '^', label = 'origin', legend = True)
    pd.value_counts(data_filtrated[item].values).plot(ax = ax, marker = 'o', label = 'filtrated', legend = True)
    i += 1   
    
i = 19
# 对数值属性，绘制直方图
for item in name_value:
    ax = fig.add_subplot(5, 5, i)
    ax.set_title(item)
    data_origin[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'origin', legend = True)
    data_filtrated[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'droped', legend = True)
    ax.axvline(data_origin[item].mean(), color = 'r')
    ax.axvline(data_filtrated[item].mean(), color = 'b')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

# 保存图像和处理后数据
fig.savefig('missing_data_similarity.jpg')
data_filtrated.to_csv('missing_data_similarity.csv', mode = 'w', encoding='utf-8', index = False,header = False)


