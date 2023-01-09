task_dir=data-generator/processed-tasks
corruption=baseline
output_dir=evaluation/OUTPUT/

echo "###############  Running evaluation  #########################"

python evaluation/evaluate_model.py \
    --taskdirectory $task_dir \
    --corruption $corruption \
    --output_dir $output_dir
    --max_num_instances_per_eval_task 3 \
    --add_task_definition True \
    --num_pos_examples 0 \
    --max_source_length 2000 \
    --max_target_length 5 \
    --batch_size 2
    
# python src/compute_metrics.py --predictions ${output_dir}/predicted_examples.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics

# rm -rf /home/hf_cache/datasets_cache/natural_instructions/
# https://github.com/Guitaricet/llm_vis
# python src/compute_metrics.py --predictions output/default/opt-30b/empty-baseline1/predicted_examples.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics
