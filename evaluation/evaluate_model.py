import argparse
import os
import json
from typing import Optional
from dataclasses import dataclass, field

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    OPTForCausalLM,
)
from safetensors.numpy import save_file

try:
    from data_collator import DataCollatorForNI
except:
    from evaluation.data_collator import DataCollatorForNI

torch.set_grad_enabled(False)


OPT_30B_DEVICE_MAP = {
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


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--taskdirectory", type=str, default="data-generator/processed-tasks")
    # parser.add_argument("taskname", type=str, default="task1344_glue_entailment_classification")
    parser.add_argument("--corruption", type=str, default="baseline")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--modelname", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_source_length", type=int, default=2500)
    parser.add_argument("--max_target_length", type=int, default=5)
    parser.add_argument("--add_task_definition", type=bool, default=False)
    parser.add_argument("--num_pos_examples", type=int, default=0)
    parser.add_argument("--max_num_instances_per_eval_task", type=int, default=10)
    args = parser.parse_args()
    return args

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    task_dir: str = field(
        default=None, metadata={"help": "The directory for saving the NaturalInstructions tasks json files."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_num_instances_per_task: int = field(
        default=None, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=500, metadata={"help": "The maximum number of instances we will consider for each validation/test task."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
            "Useful for multilingual models like mBART where the first generated token"
            "needs to be the target language token (Usually it is the target language token)"
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_task_definition: Optional[bool] = field(
        default=True,
        metadata={"help": "whether to preappend task definition before the task input."}
    )
    num_pos_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    num_neg_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context negative examples."}
    )
    add_explanation: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to add explanation for both the postive examples and negtive examples."}
    )
    def __post_init__(self):
        pass


@dataclass
class OPTArguments(DataTrainingArguments):
    taskdirectory: str = field(
        default="data-generator/processed-task", metadata={"help": "task files"}
    )
    output_dir: str = field(
        default="output/", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    modelname: str = field(
        default="facebook/opt-125m", metadata={"help": "model name"}
     )
    corruption: str = field(
        default="baseline", metadata={"help": "corruption name"}
    )
    batch_size: int = field(
        default=4, metadata={"help": "batch size for evaluation"}
    )
    taskname: str = field(
        default=None, metadata={"help": "task name"}
    )
    

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = HfArgumentParser((OPTArguments,))
    args, = parser.parse_args_into_dataclasses()

    # load the dataset (loads all files one after the other sequencially)
    raw_datasets = load_dataset(
        "evaluation/loaddataset.py", 
        data_dir="evaluation/",  # dir for test_tasks.txt
        task_dir=args.taskdirectory, 
        max_num_instances_per_eval_task=args.max_num_instances_per_eval_task,
        corruption = args.corruption
    )

    # load the tokenizer and model
    modelname = args.modelname
    tokenizer = AutoTokenizer.from_pretrained(modelname, return_tensors="pt")
    tokenizer.padding_side = "left"  # for batch inference
    tokenizer.pad_token = tokenizer.eos_token # to avoid an error

    if device == "cpu": # to test the code on local system
        model = AutoModelForCausalLM.from_pretrained(modelname)
    elif modelname == "facebook/opt-30b":
        model = AutoModelForCausalLM.from_pretrained(modelname, device_map=OPT_30B_DEVICE_MAP, load_in_8bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(modelname, device_map="auto", load_in_8bit=True)

    model: OPTForCausalLM  # for type hints
    model.eval()

    # data collator
    data_collator = DataCollatorForNI(
        tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        add_task_definition=args.add_task_definition,
        num_pos_examples=args.num_pos_examples,
        num_neg_examples=args.num_neg_examples,
        add_explanation=args.add_explanation,
        corruption=args.corruption,     
    )

    eval_dataloader = torch.utils.data.DataLoader(
        raw_datasets['test'],
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )

    # create output directory
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    prefix = f"{n_params:.2f}B_"
    model_name_for_save = prefix + modelname.split('/')[-1]

    output_dir = args.output_dir + '/' + model_name_for_save + '/'
    os.makedirs(output_dir, exist_ok=True)

    # Save the configuration of evaluation
    with open(os.path.join(output_dir, "opt_run_config.json"), "w") as fout:
        json.dump(args.__dict__, fout, indent=4)

    from collections import defaultdict
    saved_correct_prediction_heads = defaultdict(lambda: False)
    saved_incorrect_prediction_heads = defaultdict(lambda: False)

    # batch evaluation loop
    with torch.no_grad():
        with open(os.path.join(output_dir, "predicted_examples.jsonl"), "w") as fout:
            for i, batch in enumerate(tqdm(eval_dataloader)):
                
                for j in range(len(batch['labels'])):
                    # strip the whitespace in input and target
                    batch['inputs'][j] = batch['inputs'][j].strip()
                    batch['labels'][j] = batch['labels'][j].strip()
                    print("################### " + batch['Task'][j] + " #####################")
                    print("################### " + args.corruption + " #####################")
                    # print(batch['inputs'][j])

                tok_input = tokenizer(batch['inputs'], return_tensors="pt", padding=True)

                # outputs: generated ids
                # can't add attentions for some reason, output_attentions=True doesn't do anything
                outputs = model.generate(
                    input_ids=tok_input['input_ids'].to(device), # https://discuss.huggingface.co/t/batch-generation-with-gpt2/1517/6
                    attention_mask=tok_input['attention_mask'].to(device),
                    max_new_tokens=args.max_target_length,
                )

                # save predictions
                for k in range(len(batch['labels'])):
                    complete_dict = {}

                    n_input_tokens = len(tok_input['input_ids'][k])
                    response_ids = outputs[k][n_input_tokens:] # remove input tokens from generated tokens
                    response = tokenizer.decode(response_ids,
                                                skip_special_tokens=True, 
                                                clean_up_tokenization_spaces=False) # decode target tokens only. input not considered.
                                                #TODO what does clean_up_tokenization_spaces do?

                    full_generation = tokenizer.decode(
                        outputs[k],
                        skip_special_tokens=True,
                    )

                    # Note: Following original paper, we cut the generated text at the first period, since the language model sometimes generates more than one sentences.
                    response = response.strip().split(".")[0]
                    complete_dict = {
                        'id': batch['id'][k], 
                        'Task': batch['Task'][k],
                        'Corruption' : args.corruption,
                        'Categories': batch['Categories'][k],
                        'Reasoning': batch['Reasoning'][k],
                        'Instance': {
                            "input" : batch['inputs'][k], 
                            "output" : [batch['labels'][k]]
                        },
                        'Target': batch['labels'][k],
                        'Prediction': response,
                        'Full generation': full_generation,  # usefull if we want to feed this into a visualization
                    }
                    fout.write(json.dumps(complete_dict) + "\n")

                    # sve attention heads for two examples (guessed and didn't guess) for each task
                    # so, this is very hacky, but I don't see any other way to extract attention patterns
                    attention_patterns_output = model(
                        input_ids=outputs[k].unsqueeze(0),
                        output_attentions=True,
                    )

                    attention_heads = torch.stack(attention_patterns_output.attentions)
                    attention_heads = attention_heads.squeeze(1)
                    attention_heads = attention_heads.detach().cpu().numpy()

                    attention_heads_artifact = {
                        "attention_heads": attention_heads,  # (layers, heads, seq, seq)
                    }
                    safetensors_metadata = {
                        "model_name": modelname,
                        "text": full_generation,
                        "id": batch["id"][k],
                        "task": batch["Task"][k],
                        "corruption": args.corruption,
                        "output": response,
                        "target": batch["labels"][k],
                    }

                    _id = batch['id'][k]
                    _task = batch['Task'][j]
                    attention_heads_path = f"{model_name_for_save}_{args.corruption}_{_task}_{_id}.safetensors"

                    if not saved_correct_prediction_heads[_task] and response == batch['labels'][k]:
                        print("Saving correct prediction attention patterns")
                        saved_correct_prediction_heads[_task] = True
                        save_file(
                            tensor_dict=attention_heads_artifact,
                            filename=output_dir + "/" + attention_heads_path,
                            metadata=safetensors_metadata,
                        )

                    if not saved_incorrect_prediction_heads[_task] and response != batch['labels'][k]:
                        print("Saving *in*correct prediction attention patterns")
                        saved_incorrect_prediction_heads[_task] = True
                        save_file(
                            tensor_dict=attention_heads_artifact,
                            filename=output_dir + "/" + attention_heads_path,
                            metadata=safetensors_metadata,
                        )
