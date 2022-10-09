import os

import xlrd
import numpy as np
import json
if not os.path.exists('stats.json'):
    xlsx = xlrd.open_workbook('MDA_HT.xlsx')
    table = xlsx.sheet_by_index(0)
    print('cols & rows:', table.ncols, table.nrows)
    names_set = list()
    len_stats = list()
    stats_years = {}
    for i in range(1, table.nrows):
        company_name = table.cell_value(i, 2)
        company_statistic = table.cell_value(i, 6)
        company_statistic_len = len(company_statistic)
        len_stats.append(company_statistic_len)
        names_set.append(company_name)
        released_year = table.cell_value(i, 3)
        if released_year in stats_years:
            stats_years[released_year] += 1
        else:
            stats_years[released_year] = 0
    names_set = list(set(names_set))
    with open('stats.json', mode='wt', encoding='utf-8') as outp:
        json.dump([names_set, len_stats,stats_years], outp, ensure_ascii=False)
else:
    with open('stats.json', mode='rt', encoding='utf-8') as inp:
        (names_set, len_stats,stats_years) = json.load(inp)
print('length of name_set:', len(names_set))
print('reports released time statistics:',stats_years) # 公告发布时间统计
print(np.average(len_stats), np.var(len_stats))
max = max(len_stats)
min = min(len_stats)
print('max/min length:', max, '/', min)
len_stats_ = [0] * 10
j = 0
for idx,i in enumerate(range(min, max, int((max-min)/10)+1)):
    j = i + (max-min)/10
    for x in len_stats:
        if i < x < j:
            len_stats_ [idx] +=1
print('statistic of the annual reports (ten folds)', len_stats_)
# 公告长度的10等分统计:



