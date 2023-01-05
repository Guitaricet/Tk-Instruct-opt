import os

data_dir="data/splits/default"
task_dir="data/tasks"
max_num_instances_per_eval_task=2

modelname="facebook/opt-125m"
modelfolder=modelname.split('/')[-1]
b1_corruption="empty-baseline1"
b2_corruption="empty-baseline2"

b1_output_dir="output/default/{}/{}".format(modelfolder,b1_corruption)
b2_output_dir="output/default/{}/{}".format(modelfolder,b2_corruption)

#baseline1
os.system('python src/run-opt.py \
    --data_dir {} \
    --task_dir {} \
    --modelname {} \
    --corruption {} \
    --overwrite_cache \
    --max_num_instances_per_task 2 \
    --max_num_instances_per_eval_task {} \
    --add_task_definition False \
    --num_pos_examples 0 \
    --num_neg_examples 0 \
    --add_explanation False \
    --max_source_length 1024 \
    --max_target_length 8 \
    --output_dir {}'.format(data_dir, task_dir, modelname, b1_corruption, max_num_instances_per_eval_task, b1_output_dir))

#baseline2
os.system('python src/run-opt.py \
    --data_dir {} \
    --task_dir {} \
    --modelname {} \
    --corruption {} \
    --overwrite_cache \
    --max_num_instances_per_task 2 \
    --max_num_instances_per_eval_task {} \
    --add_task_definition False \
    --num_pos_examples 4 \
    --num_neg_examples 0 \
    --add_explanation False \
    --max_source_length 1024 \
    --max_target_length 8 \
    --output_dir {}'.format(data_dir, task_dir, modelname, b2_corruption, max_num_instances_per_eval_task, b2_output_dir))



corruptions_list = ['instr-placement-before-ex', 
                    'instr-placement-after-ex',
                    'instr-randomwords',
                    'instr-frequentwords',
                    'label-random-labelspace',
                    'label-random-labelspace-half',
                    'label-randomwords',
                    'label-empty',
                    'input-empty',
                    'input-oodrandom']

for corrup in corruptions_list:
#instr-placement-before-ex 
    output_dir="output/default/{}/{}".format(modelfolder,corrup)    
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
        --max_target_length 8 \
        --output_dir {}'.format(data_dir, task_dir, modelname, corrup, max_num_instances_per_eval_task, output_dir))