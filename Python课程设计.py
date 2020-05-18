# -*- coding: utf-8 -*-
"""
Created on Wed May 29 21:28:39 2019

@author: ZLT
"""

import requests
import re
import time
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.misc.pilutil import imread
import matplotlib as mpl
import pandas as pd
import csv
import numpy as np
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import jieba
import os

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = [u'SimHei']

from requests.exceptions import RequestException
currentFont = 'SimHei'


def get_page(offset):
    '''获取单页源码'''
    try:
        url = "https://www.douban.com/game/27005453/{}".format(offset)
        headers = {
            "User-Agent":"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36"
        }
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            return res.text
        return None
    except:
        return None

def parse_one_page(html):
    '''解析单页源码 '''
    pattern = re.compile('<div class="user-info">[\s]*?<a.*?>(.*?)</a>[\s]*?<span class="pubtime">(.*?)'+
                         '</span>[\s]*?<span class=".*?([0-9]\d*)" title=.*?</span>[\s\S]*?'+
                         '</div>[\s]*?<p>[\s]*?<span class.*?>(.*?)</span>(.*?)[\s]*?</p>')
    items = re.findall(pattern,html)
    #print(items)
    
    for item in  items:
        yield {
            'name' : item[0],
            'time' : item[1],
            'score' : item[2],
            'comment' : item[3]
        }
        
def write_to_csvfile(content,pages):
    '''写入到csv文件中'''
    with open("Comments.csv",'a',encoding='gb18030',newline='') as f:
        fieldnames = ["name", "time", "score", "comment"]
        writer = csv.DictWriter(f,fieldnames=fieldnames)
        if pages ==0:
            writer.writeheader()
        writer.writerows(content)
        f.close()


def make_pie(dictname):
    name = []
    number = []
    color = ['red', 'orange', 'yellow', 'green', 'blue',  'goldenrod']  
    for key,value in dictname.items():
        name.append(key)
        number.append(value)
    #print(name,number)
    plt.figure(figsize=(6,6))
    pie = plt.pie(number, colors=color, labels=name,autopct='%1.1f%%')
    for font in pie[1]:       #pie[1]:l_text,pie图外的文本
        font.set_fontproperties(currentFont)
        font.set_size(12)  #设置标签字体大小
    for digit in pie[2]:     #pie[2]:p_text,pie图内的文本
        digit.set_size(12)  
    plt.axis('equal')
    plt.title(u'评分', fontproperties=currentFont, fontsize=20)
    plt.legend(loc=0, bbox_to_anchor=(1, 1.2))  # 图例
    #设置legend的字体大小
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontproperties=currentFont, fontsize=6)
 
    # 显示图
    plt.show()


def datesort(dict):
    sort = {}
    years = ['2017', '2018', '2019']
    months = ['1','2','3','4','5','6','7','8','9','10','11','12']
    search1 = re.compile('(.*?)年.*?月')
    search2 = re.compile('.*?年(.*?)月')
    for year in years:
        for month in months:
            for key,value in dict.items():
                syear = re.findall(search1, key)
                smonth = re.findall(search2, key)
                if syear[0] == year and smonth[0] == month:
                    sort[key] = value      
    #print(sort)
    return sort

def make_brokenline(dict,s,a,b):
    plt.figure(figsize=(25,6))
    x = []
    y = []
    for key,value in dict.items():
        x.append(key)
        y.append(value)
    plt.ylim(0,max(y)+max(y)/10)
    plt.plot(x, y, marker='o', mec='r', mfc='w',label=u'y=x^2曲线图')
    plt.title(s)
    plt.xlabel(a)
    plt.ylabel(b)
    plt.show()        
    
def make_wordcloud(txt,file):
    ls = jieba.cut(txt)
    text = " ".join(ls)
    #print(text)
    wc = WordCloud(
        background_color="white", #背景颜色
        max_words=200, #显示最大词数
        #font_path="./font/msyh.tcc",  #使用字体
        font_path='simfang.ttf',
        width=1000,  #图幅宽度
        height=700
        )
    wc.generate(text)
    wc.to_file(file+".png")

    
def analyze_Comments(csvFile):
    #fitems = csv.reader(f)
    fitems = pd.read_csv(csvFile,encoding = 'gb18030')
    #print(fitems['score'])
    #filmHead = next(fitems)
    print('正在分析评论')
    #评分饼状图
    scoreCount = {}
    for comment in fitems['score']:
        scoreCount[comment] = scoreCount.get(comment,0) + 1
    #print(scoreCount)
    ret = sorted(scoreCount.items(),key = lambda x:x[1],reverse = True)
    make_pie(dict(ret))
    
    #受关注度和好评率的折线图
    search = re.compile('(.*?月).*?日')
    score_month_high = {}
    score_month = {}
    
    low_comments = ''
    high_comments = ''
    for stime,score,comment in zip(fitems['time'],fitems['score'],fitems['comment']):
        if score == 50:
            high_comments += comment
        if score <= 20:
            low_comments += comment
        year = re.findall(search,stime)
        Year = year[0]
        if len(Year) >=7 :
            score_month[Year] = score_month.get(Year,0) + 1
            if (score == 50 or score == 40):
                score_month_high[Year] = score_month_high.get(Year,0) + 1
        else:
            #print(year,score)
            month = '2019年' + Year
            score_month[month] = score_month.get(month,0) + 1
            if (score == 50 or score == 40):
                score_month_high[month] = score_month_high.get(month,0) + 1
    #print(score_month_high)
    sort_high = datesort(score_month_high)
    sort_month = datesort(score_month)
    high_rate = {}
    for high,month in zip(sort_high.items(),sort_month.items()):
        #print(high[1],month[1])
        rate = high[1]/month[1]
        high_rate[high[0]] = rate 
    make_brokenline(sort_month,'关注度随时间的变化','时间','关注数')
    #print(high_rate)
    make_brokenline(high_rate,'好评率随时间的变化','时间','好评率')
    #make_brokenline(sort_high)
    #print(score_month)
    time.sleep(1)
    print('正在生成词云')
    #词云的制作
    make_wordcloud(high_comments,'high')
    make_wordcloud(low_comments,'low')
    time.sleep(2)
    print('词云生成完毕')
    
    


def kill_match_stats():
        # 先把玩家被击杀的数据导入
        
    if not os.path.isdir(r'deaths'):
        print('文件不存在')
        return 0
    
    death_0 = pd.read_csv(r'deaths\kill_match_stats_final_0.csv')
    for file in range(0,1):
        death_1 = pd.read_csv(r'deaths\kill_match_stats_final_{}.csv'.format(file), nrows=5000000)
        death = death_0.merge(death_1, how='outer')
        
    
    print(death.shape)
    # (18426348, 12)
   
    last_seconds_erg = death.loc[(death['map'] == 'ERANGEL')&(death['killer_placement']==1), :].dropna()
    last_seconds_erg['killed_by'].value_counts()[1:10].sort_values().plot.barh(figsize=(10,5))
    plt.yticks(fontsize=12)
    plt.xlabel('击杀数')
    plt.ylabel('武器')
    plt.title('ERANGEL武器击杀数排行榜')
    plt.show()
    plt.savefig('ERANGEL武器排行.png', dpi=100)
    
    last_seconds_erg = death.loc[(death['map'] == 'MIRAMAR')&(death['killer_placement']==1), :].dropna()
    last_seconds_erg['killed_by'].value_counts()[1:10].sort_values().plot.barh(figsize=(10,5))
    plt.yticks(fontsize=12)
    plt.xlabel('击杀数')
    plt.ylabel('武器')
    plt.title('MIRAMARL武器击杀数排行榜')
    plt.show()
    plt.savefig('MIRAMAR武器排行榜.png', dpi=100)
    
    # 筛选落地成盒的玩家(选取开局4分钟之内死亡的玩家)
    in_240_seconds_erg = death.loc[(death['map'] == 'ERANGEL') & (death['time'] < 240), :].dropna()
    in_240_seconds_mrm = death.loc[(death['map'] == 'MIRAMAR') & (death['time'] < 240), :].dropna()
    
    data_erg = in_240_seconds_erg[['victim_position_x', 'victim_position_y']].values
    data_mrm = in_240_seconds_mrm[['victim_position_x', 'victim_position_y']].values
    data_erg = data_erg * 4096 / 800000
    data_mrm = data_mrm * 1000 / 800000
    
    def heatmap(x, y, s, bins=100):
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
        heatmap = gaussian_filter(heatmap, sigma=s)
    
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        return heatmap.T, extent
    
    
    bg = imread(r'erangel.jpg')
    hmap, extent = heatmap(data_erg[:, 0], data_erg[:, 1], 4.5)
    alphas = np.clip(Normalize(0, hmap.max(), clip=True)(hmap) * 4.5, 0.0, 1.)
    colors = Normalize(0, hmap.max(), clip=True)(hmap)
    colors = cm.Reds(colors)
    colors[..., -1] = alphas
    
    fig, ax = plt.subplots(figsize=(24, 24))
    ax.set_xlim(0, 4096)
    ax.set_ylim(0, 4096)
    ax.imshow(bg)
    ax.imshow(colors, extent=extent, origin='lower', cmap=cm.Reds, alpha=0.9)
    plt.gca().invert_yaxis()
    plt.savefig('out1.png', dpi=100)
    
    
    bg = imread(r'miramar.jpg')
    hmap, extent = heatmap(data_mrm[:, 0], data_mrm[:, 1], 4.5)
    alphas = np.clip(Normalize(0, hmap.max(), clip=True)(hmap) * 4.5, 0.0, 1.)
    colors = Normalize(0, hmap.max(), clip=True)(hmap)
    colors = cm.Reds(colors)
    colors[..., -1] = alphas
    
    fig, ax = plt.subplots(figsize=(24, 24))
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.imshow(bg)
    ax.imshow(colors, extent=extent, origin='lower', cmap=cm.Reds, alpha=0.9)
    plt.gca().invert_yaxis()
    plt.savefig('out2.png', dpi=100)
    
    
    
def sortV(adict):
    keys = list(adict.keys())
    keys.sort()
    dict = {}
    for key in keys:
        dict[key] = adict[key]
    return dict

def makebar(dict,s,s1,s2):
    x = []
    y = []
    plt.figure(figsize=(18,6))
    for key,value in dict.items():
        x.append(key)
        y.append(value)
    plt.bar(x,y)
    plt.xticks(x)
    plt.title(s)
    plt.xlabel(s1)
    plt.ylabel(s2)
    plt.show()

    
    
def agg_match_stats():
    assists_dict1 = {}
    assists_dict2 = {}
    kill_dict1 = {}
    kill_dict2 = {}
    if not os.path.isdir(r'aggregate'):
        print('文件不存在')
        return 0
    print('正在分析数据')
    for file in range(2):
        path = r'aggregate\agg_match_stats_{}.csv'.format(file)
        fitems = pd.read_csv(path,encoding = 'gb18030')
        #print(fitems)
        for a,b,c in zip(fitems['player_assists'],fitems['team_placement'],fitems['player_kills']):
            assists_dict1[a]=assists_dict1.get(a,0)+1
            if int(c)<45:
                kill_dict1[c] = kill_dict1.get(c,0)+1
            if int(b)==1:
                assists_dict2[a] = assists_dict2.get(a,0)+1
            if int(b)==1 and int(c)<45:
                kill_dict2[c] = kill_dict2.get(c,0)+1
        #print(assists_dict2)
        #print(assists_dict1)
        dict2 = sortV(assists_dict2)
        dict1 = sortV(assists_dict1)
        kdict1 = sortV(kill_dict1)
        kdict2 = sortV(kill_dict2)
        dict_ass = {}
        dict_k = {}
        for a,b in zip(dict1.items(),dict2.items()):
            dict_ass[a[0]] = b[1]/a[1]
        for a,b in zip(kdict1.items(),kdict2.items()):
            dict_k[a[0]] = b[1]/a[1]

    makebar(dict_ass,'吃鸡率与助攻数的关系','助攻数','吃鸡率')
    makebar(dict_k,'吃鸡率与击杀数的关系','击杀数','吃鸡率')
    
def main():
    i = 1
    print("------------执行开始------------")
    for page in range(85):
        time.sleep(1)
        p_comment = get_page("comments?start={}&sort=score".format(page*20)) 
        rows = []
        for item in parse_one_page(p_comment):
            rows.append(item)
#        print(rows)
        tt = int(i*20/85)
        a = '**'*tt
        b = '..'*(20-tt)
        c = (i/85)*100
        i += 1
        print("\r已爬取进度%{:^3.0f}[{}->{}]".format(c,a,b),end='')
        write_to_csvfile(rows, page)
    print("\n数据写入完毕")


if __name__ == '__main__':
    main()
    analyze_Comments('Comments.csv')
    kill_match_stats()
    agg_match_stats()








