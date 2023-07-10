from collections import defaultdict  
import collections
import pandas as pd
import json, time, datetime, math, csv, copy, sys
from dateutil.parser import parse
'''

worker_quality
    回答者的质量： 质量越高则发布者能获得更高的兴趣。

project_list_lines
    所有的项目

project_info
    分类
    发布时间
    结束时间

entry_info
    每个项目中的每个回答者的到达时间： 可以通过到达时间 - 任务的发布时间模拟兴趣。


industry_list

'''

# % read worker attribute: worker_quality
worker_quality = {}
csvfile = open("data/worker_quality.csv", "r")
csvreader = csv.reader(csvfile)
heaher = next(csvreader)
for line in csvreader:
    if float(line[1]) > 0.0:
        worker_quality[int(line[0])] = float(line[1]) / 100.0
csvfile.close()


# % read project id
file = open("data/project_list.csv", "r")
project_list_lines = file.readlines()
file.close()
project_dir = "project/"
entry_dir = "entry/"

all_begin_time = parse("2018-01-01T0:0:0Z")

worker_domain=defaultdict(list)

project_info = {}
entry_info = {}
limit = 24
industry_list = {}
proj_frame = []
for line in project_list_lines:
    line = line.strip('\n').split(',')

    project_id = int(line[0])
    entry_count = int(line[1])

    file = open('data/'+project_dir + "project_" + str(project_id) + ".txt", "r")
    htmlcode = file.read()
    file.close()
    text = json.loads(htmlcode)

    c = int(text["category"]) 
    project_info[project_id] = {}
    project_info[project_id]["sub_category"] = int(text["sub_category"])# % project sub_category
    project_info[project_id]["category"] = int(text["category"]) #% project category
    project_info[project_id]["entry_count"] = int(text["entry_count"]) #% project answers in total
    project_info[project_id]["start_date"] = parse(text["start_date"]) #% project start date
    project_info[project_id]["deadline"] = parse(text["deadline"]) #% project end date
    proj_frame.append([project_id, text['total_awards'], c])

    if text["industry"] not in industry_list:
        industry_list[text["industry"]] = len(industry_list)
    project_info[project_id]["industry"] = industry_list[text["industry"]] #% project domain

    entry_info[project_id] = {}
    k = 0
    while (k < entry_count):
        file = open('data/'+entry_dir + "entry_" + str(project_id) + "_" + str(k) + ".txt", "r")
        htmlcode = file.read()
        file.close()
        text = json.loads(htmlcode)

        for item in text["results"]:
            entry_number = int(item["entry_number"])
            entry_id = int(item['author'])
            entry_info[project_id][entry_number] = {}
            entry_info[project_id][entry_number]["entry_created_at"] = parse(item["entry_created_at"]) #% worker answer_time

            worker_domain[entry_id].append(c)
            # print(item.get('worker',None))
            # entry_info[project_id][entry_number]["worker"] = int(item["worker"]) #% work_id
        k += limit

print("finish read_data")


frame = []
for worker, quality in worker_quality.items():
    # print(worker)
    v = worker_domain.get(worker, None)

    if v != None:
        counter = collections.Counter(v)
        top_three = counter.most_common(1)
        top = top_three[0][0]
    else:
        top = -1
    
    frame.append([worker, top, quality])
    # if top == -1:
        # print(worker, -1)
frame = pd.DataFrame(frame, columns=['worker','domain','quality'])
frame.to_csv('worker_info.csv')
proj_frame = pd.DataFrame(proj_frame,columns=['proj_id','awards','domain'])
proj_frame.to_csv('projects_info.csv')
# print(frame)



# for k, v in worker_domain.items():
    # counter = collections.Counter(v)
    # top_three = counter.most_common(1)
    # print(k, top_three)
    # worker_domain[k] = 
    # return [item[0] for item in top_three]


# for i in range(10):
    # print(i)










'''

worker
    state : 0000 (有没有被推荐任务)
    reward:  
            [任务是不是第一次被拿] * [project的奖金]  *  [woker的兴趣 与 proj的相关性] 
    action....

proj
    state: 0000 (是不是第一次推荐给别人)
    reward:
            [woker的兴趣 与 proj的相关性] * worker_quality
    action....


pi(0000,000) -----》 (proj to worker) [action]

pi(0000,000) -----》 (proj0 -> woker1) [action]

# (0000,000) s0
# (0100,100) s1 a1 r1
# (0110,110) s2 a2 r2
# (0111,111) s3 a3 r3
# (1111,111) s4 a4 r4
# END

Q((p_state, w_state), action) = reward_W + reward_P



Data

worker： 兴趣, qualit   >>>>>  k个
proj:    domain, award=100.  >>>>q个



p1(domain=1,)        w1(domain=1, qualiyt=1)
p2(domain=2)         w2(domain=2, quality=1)

(p1->w1, p2->w2) reward
(p2->w1, p1->w2)



'''