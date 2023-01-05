import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("modelname", default=None, type=str)
parser.add_argument("--num_instances", default=2, type=int)


args = parser.parse_args()

data_dir="data/splits/default"
task_dir="data/tasks"
max_num_instances_per_eval_task=args.num_instances

modelname=args.modelname
modelfolder=modelname.split('/')[-1]
b1_corruption="empty-baseline1"
b2_corruption="empty-baseline2"

b1_output_dir="output/default/{}/{}".format(modelfolder,b1_corruption)
b2_output_dir="output/default/{}/{}".format(modelfolder,b2_corruption)

# #baseline1
# print("####### Running Baseline1 ##############")
# os.system('python src/run-opt.py \
#     --data_dir {} \
#     --task_dir {} \
#     --modelname {} \
#     --corruption {} \
#     --overwrite_cache \
#     --max_num_instances_per_task 2 \
#     --max_num_instances_per_eval_task {} \
#     --add_task_definition False \
#     --num_pos_examples 0 \
#     --num_neg_examples 0 \
#     --add_explanation False \
#     --max_source_length 1024 \
#     --max_target_length 5 \
#     --output_dir {}'.format(data_dir, task_dir, modelname, b1_corruption, max_num_instances_per_eval_task, b1_output_dir))

# # baseline2
# print("####### Running Baseline2 ##############")
# os.system('python src/run-opt.py \
#     --data_dir {} \
#     --task_dir {} \
#     --modelname {} \
#     --corruption {} \
#     --overwrite_cache \
#     --max_num_instances_per_task 2 \
#     --max_num_instances_per_eval_task {} \
#     --add_task_definition False \
#     --num_pos_examples 4 \
#     --num_neg_examples 0 \
#     --add_explanation False \
#     --max_source_length 1024 \
#     --max_target_length 5 \
#     --output_dir {}'.format(data_dir, task_dir, modelname, b2_corruption, max_num_instances_per_eval_task, b2_output_dir))



corruptions_list = ['label-empty',
                    #'input-empty']
                    # ['instr-placement-before-ex']
                    'instr-randomwords',
                    'instr-frequentwords',
                    'label-random-labelspace',
                    'label-random-labelspace-half',
                    'label-randomwords',
                    'input-oodrandom',
                    'instr-placement-after-ex']

for corrup in corruptions_list:
    output_dir="output/default/{}/{}".format(modelfolder,corrup)    
    print("####### Running {} ##############".format(corrup))
    os.system('python src/run-opt.py \
        --data_dir {} \
        --task_dir {} \
        --modelname {} \
        --corruption {} \
        --overwrite_cache \
        --max_num_instances_per_task 2 \
        --max_num_instances_per_eval_task {} \
        --add_task_definition True \
        --num_pos_examples 4 \
        --num_neg_examples 0 \
        --add_explanation False \
        --max_source_length 1024 \
        --max_target_length 5\
        --output_dir {}'.format(data_dir, task_dir, modelname, corrup, max_num_instances_per_eval_task, output_dir))