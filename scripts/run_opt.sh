data_dir=data/splits/default
task_dir=data/tasks
output_dir=output/opt-125m/definition-before
max_num_instances_per_eval_task=100
modelname=facebook/opt-125m

echo "instruction + 2 positive examples"

python src/run-opt.py \
    --data_dir $data_dir \
    --task_dir $task_dir \
    --overwrite_cache \
    --max_num_instances_per_task 100 \
    --max_num_instances_per_eval_task ${max_num_instances_per_eval_task} \
    --add_task_definition True \
    --num_pos_examples 2 \
    --num_neg_examples 0 \
    --add_explanation False \
    --max_source_length 1024 \
    --max_target_length 128 \
    --output_dir ${output_dir}/default/opt
    
python src/compute_metrics.py --predictions ${output_dir}/default/opt/predicted_examples.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics
