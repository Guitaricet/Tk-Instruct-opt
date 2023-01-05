 ## Requirements

 ```bash
pip install -r requirements.txt
```
 
 ## Data

The models are evaluated on subset of [Super-NaturalInstructions](https://github.com/allenai/natural-instructions). The original set of datasets can be cloned by running:

```bash
git clone git@github.com:allenai/natural-instructions.git data
```
create `dev_tasks.txt` in the `data/splits/default` folder. \TODO remove this need
 
## Metric
Exact match and Rouge-L

## Corruptions
 
 1) empty-baseline1 -  x . 
                   # set --add_task_definition False \
                   #     --num_pos_examples 0 \
                   #     --num_neg_examples 0 \
                   #     --add_explanation False \
                   #     --corruption baseline1
                   Keep the rest of the arguments unchanged
 2) examples-baseline2 -  xy xy xy xy | x
                   # set --add_task_definition False \
                   #     --num_pos_examples 4 \
                   #     --num_neg_examples 0 \
                   #     --add_explanation False \
                   #     --corruption baseline2
                   
 3) instr-placement-before-ex -   I | xy xy xy xy | x
                   # set --add_task_definition True \
                   #     --num_pos_examples 4 \
                   #     --num_neg_examples 0 \
                   #     --add_explanation False \
                   #     --corruption instr-placement-before-ex
 4) instr-placement-after-ex  -       xy xy xy xy | I | x
 5) instr-randomwords        -   I(random) | xy xy xy xy | x 
 6) instr-frequentwords      -   I(freq) | xy xy xy xy | x 
 Rethinking corruptions 
 7) label-random-labelspace        -   I | xy(ran) xy(ran) xy(ran) xy(ran) | x  # works only for datasets where there is label options and all are incorect labels
 8) label-random-labelspace-half   -   I | xy(ran) xy(ran) xy xy | x  # works only for datasets where there is label options, every other label is incorrect
 9) label-randomwords              -   I | xy(ran) xy(ran) xy(ran) xy(ran) | x # label space is created with random words and y i s randomly chosen from this label space
 10) label-empty                    -   I | x x x x | x   # Input:x
 11) input-empty                   -   I | y y y y (number of classes) | x  # Output:y
 12)  input-oodrandom                -   I | x(oodran)y x(oodran)y x(oodran)y x(oodran)y | x \TODO get the common crawl corpus
   
NOTE:  for all these corruptions, no explanation and negative examples are included. \TODO use explanations and negative examples too

## How to run evaluation

run `./scripts/run_opt.sh` to run evaluation

change modelname, modelfolder name and corruption name for the desired output
```
modelname=facebook/opt-125m
modelfolder=opt-125m
corruption=label-randomwords
```

The output is stored in `output/default/`
jsonl contains the prediction and `attention` folder stores all attention heads.

Other files and folders
- `dataforcorruptions` includes
    -  list of random english words in `randomwords.txt`
    -  list of wikipedia frequent words in `frequentwords.txt`
    -  corpus for ood inputs in `corpus.txt`
 - `compute_metrics.py` has all metric computation methods
 - `dataset_labels.py` has label space for datasets
 - `ni_collator_opt.py` is the modified version of `ni_collator.py` from Super-NaturalInstructions and includes all mentioned corruptions  