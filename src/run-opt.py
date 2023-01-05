import json
from tqdm import tqdm
import os
import random
import torch
from transformers import HfArgumentParser
# from run_s2s import DataTrainingArguments
from datasets import load_dataset
from ni_collator_opt import DataCollatorForNI
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from safetensors.numpy import save_file


# TODO add a script to add more in-context examples to tasks
# TODO if prediction already exists then not compute it again

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
    tk_instruct: Optional[bool] = field(
        default=False,
        metadata={"help": "tk_instruct will train a model combining all valid instruction encodings. This will overwrite the other settings about instruction encoding."} 
    )
    def __post_init__(self):
        pass


@dataclass
class OPTArguments(DataTrainingArguments):
    data_dir: str = field(
        default="data/splits", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    output_dir: str = field(
        default="output/", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    modelname: str = field(
        default="facebook/opt-125m", metadata={"help": "model name"}
     )
    corruption: str = field(
        default="inst-placement-before-ex", metadata={"help": "model name"}
    )
    
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # random seed is from SuperNatInstrcut paper
    # Rethinking paper has seeds 100,13,21,42,87
    random.seed(123)
    parser = HfArgumentParser((OPTArguments,))
    args, = parser.parse_args_into_dataclasses()

    # load the dataset
    raw_datasets = load_dataset(
        "src/ni_dataset.py", 
        data_dir=args.data_dir, 
        task_dir=args.task_dir, 
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=args.max_num_instances_per_eval_task
    )
    
    # device map for OPT-30b
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
    

    # load the tokenizer and model
    modelname = args.modelname
    tokenizer = AutoTokenizer.from_pretrained(modelname, return_tensors="pt")
    tokenizer.padding_side = "left"  # for batch inference
    tokenizer.pad_token = tokenizer.eos_token # to avoid an error

    if device=="cpu": # to test the code on local system
        model = AutoModelForCausalLM.from_pretrained(modelname)
    elif modelname == "facebook/opt-30b":
        model = AutoModelForCausalLM.from_pretrained(modelname, device_map=device_map, load_in_8bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(modelname, device_map="auto", load_in_8bit=True)

    model.eval()
    
    randomwords = None
    frequentwords = None
    # create a random word sentence for this corruption
    if args.corruption == 'instr-randomwords' or args.corruption == 'label-randomwords':
        randomwords = []
        random_words_file = open('src/dataforcorruptions/randomwords.txt','r')
        raw_lines = random_words_file.readlines()
        for i in range(200):    # create a sentence with 200 random words 
            randomwords.append(raw_lines[i].strip('\n'))
        randomwords = ' '.join(randomwords) 

    # create a frequent word sentence for this corruption
    if args.corruption == 'instr-frequentwords':
        frequentwords = []
        frequent_words_file = open('src/dataforcorruptions/frequentwords.txt','r')
        raw_lines = frequent_words_file.readlines()
        for i in range(200):    # create a sentence with 200 random words 
            frequentwords.append(raw_lines[i].strip('\n'))
        frequentwords = ' '.join(frequentwords) 
        
    # data collator
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
        text_only=True,
        corruption=args.corruption,
        randomwords=randomwords,
        frequentwords=frequentwords
    )
    
    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir+"/"+ "attentions", exist_ok=True)

    # Save the configuration of evaluation
    with open(os.path.join(args.output_dir, "opt_run_config.json"), "w") as fout:
        json.dump(args.__dict__, fout)

    eval_dataloader = torch.utils.data.DataLoader(raw_datasets["test"], batch_size=4, collate_fn=data_collator)
    # batch evaluation loop
    with torch.no_grad():
        with open(os.path.join(args.output_dir, "predicted_examples.jsonl"), "w") as fout:
            for i, batch in enumerate(tqdm(eval_dataloader)):
                for j in range(len(batch['labels'])):
                    # strip the whitespace in input and target
                    batch['inputs'][j] = batch['inputs'][j].strip()
                    batch['labels'][j] = batch['labels'][j].strip()
                
                tok_input = tokenizer(batch['inputs'], return_tensors="pt", padding=True)

                # outputs [generated_ids, attentions]
                outputs = model.generate(input_ids=tok_input['input_ids'].to(device), # https://discuss.huggingface.co/t/batch-generation-with-gpt2/1517/6
                                        attention_mask=tok_input['attention_mask'].to(device),
                                        max_new_tokens=args.max_target_length, 
                                        return_dict_in_generate=True,
                                        output_attentions=True)
                
                # save predictions
                for k in range(len(batch['labels'])):
                    complete_dict = {}
        
                    response = tokenizer.decode(outputs[0][k][-args.max_target_length:], 
                                                skip_special_tokens=True, 
                                                clean_up_tokenization_spaces=False) # decode target tokens only. input not considered.
                                                #TODO what does clean_up_tokenization_spaces do?

                    # Note: Following original paper, we cut the generated text at the first period, since the language model sometimes generates more than one sentences.
                    response = response.strip().split(".")[0]
                    complete_dict = {'id': batch['id'][k], 
                                      'Task': batch['Task'][k],
                                      'Categories': batch['Categories'][k],
                                      'Reasoning': batch['Reasoning'][k],
                                      'Instance': {"input" : batch['inputs'][k], 
                                                    "output" : [batch['labels'][k]]},
                                      'target': batch['labels'][k],
                                      'prediction': response
                                    }
                    fout.write(json.dumps(complete_dict) + "\n")
                
                    # # save attentions
                    # # outputs[1] is a tuple of lenght of max_new_tokens, outputs[1][0] which the attention heads for generating first new token
                    # attention_heads = torch.stack(outputs[1][0])  # shape (num_layers, batch_size, heads, seq_len, seq_len)
                    # attention_heads = attention_heads[:,k,:,:,:] # to make batch_size is 1, shape (num_layers, heads, seq_len, seq_len)
                    # attention_heads = attention_heads.detach().cpu().numpy()
                    # # attention_heads = attention_heads.squeeze(1)  #  (num_layers, heads, seq_len, seq_len)

                    # # how to save
                    # attention_heads_artifact = {
                    #     "attention_heads": attention_heads,  # (layers, heads, seq, seq)
                    # }
                    # safetensors_metadata = {
                    #     "model_name": args.modelname,
                    #     "corruption": args.corruption,
                    #     "task_id" : batch['Task'][k],
                    #     "input_id" : batch['id'][k],
                    #     "input_text": batch['inputs'][k],
                    # }
                    # n_params = sum(p.numel() for p in model.parameters()) / 1e9
                    # prefix = f"{n_params:.2f}B_"
                    # model_name_for_save = prefix + args.modelname.split('/')[-1]
                    # # task_name_for_save = batch['Task'][k].split('_')[0]
                    # # model_name_for_save = prefix + args.modelname.replace("/", "_")
                    # attention_heads_path = "AH_" + model_name_for_save + "_" + args.corruption+ "_" + batch['id'][k] + ".safetensors" # AH stands for attention heads
                
                    # save_file(
                    #     tensor_dict=attention_heads_artifact,
                    #     filename=args.output_dir + "/" + "attentions" + "/" + attention_heads_path,
                    #     metadata=safetensors_metadata,
                    # )
                    
                    

        
