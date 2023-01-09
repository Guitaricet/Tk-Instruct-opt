import os
import json
import random
import dataset_labels
from transformers import AutoTokenizer


# creates the baseline data. baseline data is changes added to original file. Corruptions will be added on the baseline data

def process_task1344_glue_entailment_classification(data, add_inline_instruction=False):
    # Add inline instrcutions if exists
    data["inline-instruction"] = "Does Sentence 1 entail Sentence 2?"
    for example in data["Positive Examples"]:
        input_str = example["input"]
        output_str = "yes" if example["output"] == "1" else "no"
        if add_inline_instruction:
            example['input'] = "{} {}".format(input_str, data['inline-instruction'])
        example['output'] = output_str
        example['explanation'] = example['explanation'].replace(" 0", " no").replace(" 1 ", " yes ")
    
    for example in data["Negative Examples"]:
        input_str = example["input"]
        output_str = "yes" if example["output"] == "1" else "no"
        if add_inline_instruction:
            example['input'] = "{} {}".format(input_str, data['inline-instruction'])
        example['output'] = output_str
        example['explanation'] = example['explanation'].replace(" 0", " no").replace(" 1 ", " yes ")
    
    for example in data["Instances"]:
        input_str = example["input"]
        output_str = "yes" if example["output"] == "1" else "no"
        if add_inline_instruction:
            example['input'] = "{} {}".format(input_str, data['inline-instruction'])
        example['output'] = output_str
    
    return data


def replace_wth_randomwords(text, tokenizer, start_idx, end_idx):
    print(text)
    text = text.split()
    text_tok_len = len(tokenizer(text)['input_ids']) # number of tokens

    randomwords = []
    random_words_file = open('data-generator/data-for-corruptions/randomwords.txt','r')

    raw_lines = random_words_file.readlines()
    for i in range(start_idx, end_idx):    # create a sentence with start_idx to end_idx random words 
        randomwords.append(raw_lines[i].strip('\n'))
    randomwords = ' '.join(randomwords) 
    
    if text_tok_len == 1 : text_tok_len += 1
    randomwords_text_ids = tokenizer(randomwords)['input_ids'][:text_tok_len]
    randomwords_text = tokenizer.decode(randomwords_text_ids, skip_special_tokens=True)
    
    return randomwords_text


def replace_wth_frequqentwords(text, tokenizer, start_idx, end_idx):
    '''replace a text with randomly selected words from top 500 high frequent words (from wikipedia) list
    '''
    text = text.split()
    text_tok_len = len(tokenizer(text)['input_ids']) # number of tokens

    frequentwords = []
    all_frequentwords = open('data-generator/data-for-corruptions/frequentwords.txt','r')
    raw_lines = all_frequentwords.readlines()
   
    for i in range(500):    # create a sentence with 500 high frequent words 
        frequentwords.append(raw_lines[i].strip('\n'))
    random.seed(1)
    random.shuffle(frequentwords)
    frequentwords = frequentwords[start_idx: end_idx] # can be 0 to 500
    frequent_text = ' '.join(frequentwords) 

    frequentwords_text_ids = tokenizer(frequent_text)['input_ids'][:text_tok_len]
    frequentwords_text = tokenizer.decode(frequentwords_text_ids, skip_special_tokens=True)
    return frequentwords_text
    

def main():
    # Directory containing the JSON files
    directory = "data-generator/processed-tasks"
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", return_tensors="pt")

    # Modify each JSON file in the directory
    for filename in os.listdir(directory):
        if filename.endswith("baseline.json"):

            ############### OPEN BASELINE FILE AND ADD CORRUPTIONS AND DUMP INTO JSON FILE FOR EACH CORRUPTION AND FOR EACH TASK #############
            print(filename)
            
            ############### instr_randomwords ##################
            # Open the baseline JSON file
            with open("data-generator/processed-tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                baseline_data = json.load(f)

            baseline_data["corruption_id"] = "instr_randomwords"  # with inline instruction
    
            baseline_data['Definition'][0] =  replace_wth_randomwords(baseline_data['Definition'][0], tokenizer, 0, 200) # replace definition with random words
            inline_intr = baseline_data['inline-instruction']
            inline_intr_randomwords = replace_wth_randomwords(baseline_data['inline-instruction'], tokenizer, 200, 300)

            for example in baseline_data["Positive Examples"]:
                example['input'] = example['input'].replace(inline_intr, inline_intr_randomwords)
            
            for example in baseline_data["Negative Examples"]:
                example['input'] = example['input'].replace(inline_intr, inline_intr_randomwords)
            
            for example in baseline_data["Instances"]:
                example['input'] = example['input'].replace(inline_intr, inline_intr_randomwords)
                
            filename_revised = filename.split('.')[0] # without .json
            filename_revised = filename_revised.replace('_baseline', '') # without baseline
            # Write the data back to the file
            with open("data-generator/processed-tasks/{}_{}.json".format(filename_revised, "instr_randomwords"), "w") as f:
                json.dump(baseline_data, f)


             ############### instr_frequentwords ##################

            # Open the baseline JSON file
            with open("data-generator/processed-tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                baseline_data = json.load(f)

            baseline_data["corruption_id"] = "instr_frequentwords"  # with inline instruction
            baseline_data['Definition'][0] =  replace_wth_frequqentwords(baseline_data['Definition'][0], tokenizer, 0, 350) # replace definition with frequent words
            inline_intr = baseline_data['inline-instruction']
            inline_intr_frequentwords = replace_wth_frequqentwords(baseline_data['inline-instruction'], tokenizer, 350, 499)

            for example in baseline_data["Positive Examples"]:
                example['input'] = example['input'].replace(inline_intr, inline_intr_frequentwords)
            
            for example in baseline_data["Negative Examples"]:
                example['input'] = example['input'].replace(inline_intr, inline_intr_frequentwords)
            
            for example in baseline_data["Instances"]:
                example['input'] = example['input'].replace(inline_intr, inline_intr_frequentwords)
                
            filename_revised = filename.split('.')[0] # without .json
            filename_revised = filename_revised.replace('_baseline', '') # without baseline
            # Write the data back to the file
            with open("data-generator/processed-tasks/{}_{}.json".format(filename_revised, "instr_frequentwords"), "w") as f:
                json.dump(baseline_data, f)

            
            ############### wrong_labels_labelspace ################## works only if the dataset has a labelspace

            # Open the baseline JSON file
            with open("data-generator/processed-tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                baseline_data = json.load(f)
                filename_revised = filename.split('.')[0] # without .json
                filename_revised = filename_revised.replace('_baseline', '') # without baseline

                if filename_revised in dataset_labels.LABELS: # if labelspace exists for a dataset

                    baseline_data["corruption_id"] = "wrong_labels_labelspace"  # with inline instruction
                    
                    for example in baseline_data["Positive Examples"]:
                        labelspace = dataset_labels.LABELS[filename_revised].copy()
                        true_label = example['output']
                        labelspace.remove(true_label)
                        example['output'] = random.choice(labelspace)
                    
                    for example in baseline_data["Negative Examples"]:
                        labelspace = dataset_labels.LABELS[filename_revised].copy()
                        true_label = example['output']
                        labelspace.remove(true_label)
                        example['output'] = random.choice(labelspace)
                    
                    # Write the data back to the file
                    with open("data-generator/processed-tasks/{}_{}.json".format(filename_revised, "wrong_labels_labelspace"), "w") as f:
                        json.dump(baseline_data, f)

            ############### wrong_labels_labelspace_halfcases ################## works only if the dataset has a labelspace

            # Open the baseline JSON file
            with open("data-generator/processed-tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                baseline_data = json.load(f)
                filename_revised = filename.split('.')[0] # without .json
                filename_revised = filename_revised.replace('_baseline', '') # without baseline
            
                if filename_revised in dataset_labels.LABELS: # if labelspace exists for a dataset

                    baseline_data["corruption_id"] = "wrong_labels_labelspace_halfcases"  # with inline instruction
                    
                    count = 0
                    for example in baseline_data["Positive Examples"]:
                        if count%2==0:
                            labelspace = dataset_labels.LABELS[filename_revised].copy()
                            true_label = example['output']
                            labelspace.remove(true_label)
                            example['output'] = random.choice(labelspace)
                        count += 1

                    count = 0
                    for example in baseline_data["Negative Examples"]:
                        if count%2==0:
                            labelspace = dataset_labels.LABELS[filename_revised].copy()
                            true_label = example['output']
                            labelspace.remove(true_label)
                            example['output'] = random.choice(labelspace)
                        count += 1
                        
                    # Write the data back to the file
                    with open("data-generator/processed-tasks/{}_{}.json".format(filename_revised, "wrong_labels_labelspace_halfcases"), "w") as f:
                        json.dump(baseline_data, f)

            
            ############### labels_randomwords ################## 

            # Open the baseline JSON file
            with open("data-generator/processed-tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                baseline_data = json.load(f)
                filename_revised = filename.split('.')[0] # without .json
                filename_revised = filename_revised.replace('_baseline', '') # without baseline

                baseline_data["corruption_id"] = "labels_randomwords"
                start_idx, end_idx = 0, 20 # assuming target len wont be greater than 20

                for example in baseline_data["Positive Examples"]:
                    example['output'] = replace_wth_randomwords(example['output'], tokenizer, start_idx, end_idx)
                    start_idx = end_idx
                    end_idx += 10
                
                for example in baseline_data["Negative Examples"]:
                    example['output'] = replace_wth_randomwords(example['output'], tokenizer, start_idx, end_idx)
                    start_idx = end_idx
                    end_idx += 10

                # Write the data back to the file
                with open("data-generator/processed-tasks/{}_{}.json".format(filename_revised, "labels_randomwords"), "w") as f:
                    json.dump(baseline_data, f)


            ############### input_ood ################## 

            # Open the baseline JSON file
            with open("data-generator/processed-tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                baseline_data = json.load(f)
                filename_revised = filename.split('.')[0] # without .json
                filename_revised = filename_revised.replace('_baseline', '') # without baseline

                baseline_data["corruption_id"] = "input_ood"
               
                ood_corpus = open('src/dataforcorruptions/corpus.txt','r')
                raw_lines = ood_corpus.readlines()
                ood_sent = []
                for i in range(len(raw_lines)):
                    ood_input = raw_lines[i].strip("\n")
                    ood_sent.append(ood_input)

                index = 0
                for example in baseline_data["Positive Examples"]:
                    example['input'] = ood_sent[index]
                    index += 1
                  
                for example in baseline_data["Negative Examples"]:
                    example['input'] = ood_sent[index]
                    index += 1
                    
                # Write the data back to the file
                with open("data-generator/processed-tasks/{}_{}.json".format(filename_revised, "input_ood"), "w") as f:
                    json.dump(baseline_data, f)

            ############### no_inline_instr ################## 

            # Open the baseline JSON file
            with open("data-generator/processed-tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                baseline_data = json.load(f)
                filename_revised = filename.split('.')[0] # without .json
                filename_revised = filename_revised.replace('_baseline', '') # without baseline

                baseline_data["corruption_id"] = "no_inline_instr"
               
                for example in baseline_data["Positive Examples"]:
                    example['input'] = example['input'].replace(baseline_data['inline-instruction'],'').strip()

                for example in baseline_data["Negative Examples"]:
                    example['input'] = example['input'].replace(baseline_data['inline-instruction'],'').strip()
                
                for example in baseline_data["Instances"]:
                    example['input'] = example['input'].replace(baseline_data['inline-instruction'],'').strip()


                    
                # Write the data back to the file
                with open("data-generator/processed-tasks/{}_{}.json".format(filename_revised, "no_inline_instr"), "w") as f:
                    json.dump(baseline_data, f)
                 
                 

if __name__ == "__main__":
    main()