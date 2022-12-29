import json
import tqdm
import os
import random
import pickle
import torch
from transformers import HfArgumentParser
from run_s2s import DataTrainingArguments
from datasets import load_dataset
from ni_collator import DataCollatorForNI
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class OPTArguments(DataTrainingArguments):
    data_dir: str = field(
        default="data/splits", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    output_dir: str = field(
        default="output/opt/", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    # modelname: float = field(
    #     default=0, metadata={"help": "model name for opt"}
    # )
    


if __name__ == "__main__":
    
    random.seed(123)
    parser = HfArgumentParser((OPTArguments,))
    args, = parser.parse_args_into_dataclasses()
    raw_datasets = load_dataset(
        "src/ni_dataset.py", 
        data_dir=args.data_dir, 
        task_dir=args.task_dir, 
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=args.max_num_instances_per_eval_task
    )
    
    device_map = {
    'model.decoder.embed_tokens': 0,
    'model.decoder.embed_positions': 0,
    'model.decoder.final_layer_norm': 0,
    'model.decoder.layers.0': 0,
    'model.decoder.layers.1': 0,
    'model.decoder.layers.2': 0,
    'model.decoder.layers.3': 0,
    'model.decoder.layers.4': 0,
    'model.decoder.layers.5': 0,
    'model.decoder.layers.6': 0,
    'model.decoder.layers.7': 0,
    'model.decoder.layers.8': 0,
    'model.decoder.layers.9': 0,
    'model.decoder.layers.10': 0,
    'model.decoder.layers.11': 0,
    'model.decoder.layers.12': 0,
    'model.decoder.layers.13': 0,
    'model.decoder.layers.14': 0,
    'model.decoder.layers.15': 0,
    'model.decoder.layers.16': 0,
    'model.decoder.layers.17': 0,
    'model.decoder.layers.18': 0,
    'model.decoder.layers.19': 0,
    'model.decoder.layers.20': 0,
    'model.decoder.layers.21': 0,
    'model.decoder.layers.22': 0,
    'model.decoder.layers.23': 0,
    'model.decoder.layers.24': 1,
    'model.decoder.layers.25': 1,
    'model.decoder.layers.26': 1,
    'model.decoder.layers.27': 1,
    'model.decoder.layers.28': 1,
    'model.decoder.layers.29': 1,
    'model.decoder.layers.30': 1,
    'model.decoder.layers.31': 1,
    'model.decoder.layers.32': 1,
    'model.decoder.layers.33': 1,
    'model.decoder.layers.34': 1,
    'model.decoder.layers.35': 1,
    'model.decoder.layers.36': 1,
    'model.decoder.layers.37': 1,
    'model.decoder.layers.38': 1,
    'model.decoder.layers.39': 1,
    'model.decoder.layers.40': 1,
    'model.decoder.layers.41': 1,
    'model.decoder.layers.42': 1,
    'model.decoder.layers.43': 1,
    'model.decoder.layers.44': 1,
    'model.decoder.layers.45': 1,
    'model.decoder.layers.46': 1,
    'model.decoder.layers.47': 1,
    'lm_head': 1,
    }


    model = AutoModelForCausalLM.from_pretrained('facebook/opt-30b', device_map=device_map, load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-30b', return_tensors="pt")
    data_collator = DataCollatorForNI(
        tokenizer,
        model=None,
        padding="max_length" if args.pad_to_max_length else "longest",
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        add_task_definition=args.add_task_definition,
        num_pos_examples=args.num_pos_examples,
        num_neg_examples=args.num_neg_examples,
        add_explanation=args.add_explanation,
        text_only=True
    )
    print('hi')
    os.makedirs(args.output_dir, exist_ok=True)

    # create attention folder
    attn_path = 'attentions/definition-before/'
    isExist = os.path.exists(attn_path)
    if not isExist:
        os.makedirs(attn_path)


    with open(os.path.join(args.output_dir, "opt_run_config.json"), "w") as fout:
        json.dump(args.__dict__, fout)

    existing_requests = {}
    if os.path.exists(os.path.join(args.output_dir, "predicted_examples.jsonl")):
        with open(os.path.join(args.output_dir, "predicted_examples.jsonl")) as fin:
            for line in fin:
                request_info = json.loads(line)
                existing_requests[request_info["opt_input"]] = request_info["opt_response"]


    with open(os.path.join(args.output_dir, "predicted_examples.jsonl"), "w") as fout:
        print(raw_datasets['test']['Task'])
        for example in tqdm.tqdm(raw_datasets["test"]):
            encoded_example = data_collator([example])
            
            example["opt_input"] = encoded_example["inputs"][0].strip()
            example["opt_target"] = encoded_example["labels"][0].strip()

            if example["opt_input"] in existing_requests:
                response = existing_requests[example["opt_input"]]
            else:
                tok_input = tokenizer(example["opt_input"], return_tensors="pt")
                tok_input = tok_input.to("cuda")
                output = model.generate(**tok_input, max_length=len(tok_input.input_ids[0])+args.max_target_length, return_dict_in_generate=True)
                generate_ids = output[0]
                attentions = output[1]
                response = tokenizer.decode(generate_ids[0][len(tok_input.input_ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                example["opt_response"] = response

                # save attentions
                task_attn_path = 'attentions/definition-before/{}'.format(example['Task']) 
                if not os.path.exists(task_attn_path):
                    os.makedirs(task_attn_path)

                with open('attentions/definition-before/{}/{}.pickle'.format(example['Task'], example['Instance']['id']), 'wb') as handle:
                        pickle.dump(attentions, handle)
                
            # Note: we cut the generated text at the first period, since the GPT3 language model sometimes generates more than one sentences.
            # Our results show that this won't affect the instruct-GPT3 model very much, but will significantly improve the original GPT3 LM.
            example["prediction"] = response.strip().split(".")[0]
            print(example["prediction"])
            fout.write(json.dumps(example) + "\n")

        