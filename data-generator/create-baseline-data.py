import os
import json
from transformers import AutoTokenizer


# creates the baseline data. baseline data is changes added to original file. Corruptions will be added on the baseline data

def process_task1344_glue_entailment_classification(data):
    # Add inline instrcutions if exists
    data["inline-instruction"] = "Does Sentence 1 entail Sentence 2?"
    data["Definition"] = ["In this task, you're given two sentences. Indicate if the first sentence clearly entails the second sentence (i.e., one can conclude the 2nd sentence by reading the 1st one). Indicate your answer with 'yes' if the first sentence entails the second sentence, otherwise answer with 'no'."]
    for example in data["Positive Examples"]:
        input_str = example["input"]
        output_str = "yes" if example["output"] == "1" else "no"
        
        example['input'] = "{} {}".format(input_str, data['inline-instruction'])
        example['output'] = output_str
        example['explanation'] = example['explanation'].replace(" 0", " no").replace(" 1 ", " yes ")
    
    for example in data["Negative Examples"]:
        input_str = example["input"]
        output_str = "yes" if example["output"] == "1" else "no"
        
        example['input'] = "{} {}".format(input_str, data['inline-instruction'])
        example['output'] = output_str
        example['explanation'] = example['explanation'].replace(" 0", " no").replace(" 1 ", " yes ")
    
    for example in data["Instances"]:
        input_str = example["input"]
        output_str = "yes" if example["output"] == "1" else "no"
        example['input'] = "{} {}".format(input_str, data['inline-instruction'])
        example['output'] = output_str  
    return data


def process_task843_financial_phrasebank_classification(data):
    # Add inline instrcutions if exists
    data["inline-instruction"] = "Is the sentiment of the sentence 'negative', 'neutral', or 'positive'?"

    for example in data["Positive Examples"]:
        input_str = example["input"].replace(" .",".")
        example['input'] = "{} {}".format(input_str, data['inline-instruction'])
      
    for example in data["Negative Examples"]:
        input_str = example["input"].replace(" .",".")
        example['input'] = "{} {}".format(input_str, data['inline-instruction'])

    for example in data["Instances"]:
        input_str = example["input"].replace(" .",".")
        example['input'] = "{} {}".format(input_str, data['inline-instruction'])

    return data


def main():
    # Directory containing the JSON files
    directory = "data-generator/original-tasks"

    # Modify each JSON file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            # Open the JSON file
            with open("data-generator/original-tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                original_data = json.load(f)

            ############### BASELINE DATA ##################
            original_data["corruption_id"] = "baseline".split()  # with inline instruction
            print(original_data["corruption_id"])
            
            ############## creating task1344_glue_entailment_classification_baseline.json  ###########
            if filename == "task1344_glue_entailment_classification.json":
                processed_task = process_task1344_glue_entailment_classification(original_data)
                # Write the data back to the file
                filename = filename.split('.')[0]
                with open("data-generator/processed-tasks/{}_{}.json".format(filename,"baseline"), "w") as f:
                    json.dump(processed_task, f)

            ############# creating task843_financial_phrasebank_classification_baseline.json  ###########
            if filename == "task843_financial_phrasebank_classification.json":
                processed_task = process_task843_financial_phrasebank_classification(original_data)

                # Write the data back to the file
                filename = filename.split('.')[0]
                with open("data-generator/processed-tasks/{}_{}.json".format(filename,"baseline"), "w") as f:
                    json.dump(processed_task, f)

            

if __name__ == "__main__":
    main()