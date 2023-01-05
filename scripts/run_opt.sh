data_dir=data/splits/default
task_dir=data/tasks
max_num_instances_per_eval_task=2

modelname=facebook/opt-125m
modelfolder=opt-125m
corruption=label-randomwords

output_dir=output/default/$modelfolder/$corruption

echo "instruction + 4 positive examples"

python src/run-opt.py \
    --data_dir $data_dir \
    --task_dir $task_dir \
    --modelname $modelname \
    --corruption $corruption \
    --overwrite_cache \
    --max_num_instances_per_task 2 \
    --max_num_instances_per_eval_task ${max_num_instances_per_eval_task} \
    --add_task_definition True \
    --num_pos_examples 4 \
    --num_neg_examples 0 \
    --add_explanation False \
    --max_source_length 1024 \
    --max_target_length 8 \
    --output_dir ${output_dir}
    
python src/compute_metrics.py --predictions ${output_dir}/predicted_examples.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics

# rm -rf /home/hf_cache/datasets_cache/natural_instructions/
# https://github.com/Guitaricet/llm_vis
# python src/compute_metrics.py --predictions output/default/opt-125m/empty-baseline1/predicted_examples.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics
