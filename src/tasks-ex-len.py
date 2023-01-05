import json
import os
import tqdm
import glob

task_names = open('data/splits/default/train_tasks.txt','r')
raw_lines = task_names.readlines()



for task in raw_lines:
    
    task = task.strip('\n')
    with open(os.path.join('data/tasks/', f'{task}.json')) as fin:
        task_data = json.load(fin)
        if len(task_data['Positive Examples']) >=4 and len(task_data['Negative Examples']) >=2 and len(task_data['Positive Examples'][0]['output']) <= 30:
            print(task," ",  len(task_data['Positive Examples']), ' ', len(task_data['Negative Examples']))
                