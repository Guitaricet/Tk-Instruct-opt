import logging
import random
import string
from transformers.data.data_collator import *
import dataset_labels

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForNI:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_task_definition: bool = True
    num_pos_examples: int = 0
    num_neg_examples: int = 0
    add_explanation: bool = False
    tk_instruct: bool = False
    text_only: bool=False
    corruption: str='inst-placement-before-ex'
    randomwords: str=None
    frequentwords: str=None
    
    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []
        id = []
        task = []
        categories = []
        reasoning = []
        for instance in batch:
            # id
            id.append(instance['id'])

            # Task
            task.append(instance['Task'])

            # Categories
            categories.append(instance['Categories'])

            # Categories
            reasoning.append(instance['Reasoning'])

            # input
            if self.tk_instruct:
                all_valid_encodings = [
                    # instruction only
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 0, "num_neg_examples": 0, "add_explanation": False}, 
                    # example only
                    {"add_task_name": False, "add_task_definition": False, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False}, 
                    # instruction + pos examples
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False}, 
                    # instruction + pos examples + neg examples 
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 2, "add_explanation": False},
                    # instruction + pos (w. explanation) 
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": True}, 
                ]
                encoding_schema = random.choice(all_valid_encodings)
                add_task_name = encoding_schema["add_task_name"]
                add_task_definition = encoding_schema["add_task_definition"]
                num_pos_examples = encoding_schema["num_pos_examples"]
                num_neg_examples = encoding_schema["num_neg_examples"]
                add_explanation = encoding_schema["add_explanation"]
            else:
                add_task_name = self.add_task_name
                add_task_definition = self.add_task_definition
                num_pos_examples = self.num_pos_examples
                num_neg_examples = self.num_neg_examples
                add_explanation = self.add_explanation 

            task_input = ""
            # add the input first.
            task_input += "Now complete the following example -\n"
            task_input += f"Input: {instance['Instance']['input'].strip()}"
            if not task_input[-1] in string.punctuation:
                task_input += "."
            task_input += "\n"
            task_input += "Output: "
            
            task_name = ""
            if add_task_name:
                task_name += instance["Task"] + ". "

            definition = ""
            if add_task_definition:
                if isinstance(instance["Definition"], list):
                    definition = "Definition: " + instance["Definition"][0].strip() # TODO: should we use <Definition>?
                else:
                    definition = "Definition: " + instance["Definition"].strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                definition += "\n\n"
            
            if self.corruption=='instr-randomwords':
                len_def = len(self.tokenizer(definition)['input_ids'])
                randomwords_def_ids = self.tokenizer(self.randomwords)['input_ids'][:len_def]
                randomwords_def = self.tokenizer.decode(randomwords_def_ids)
                definition = randomwords_def
            
            if self.corruption=='instr-frequentwords':
                len_def = len(self.tokenizer(definition)['input_ids'])
                frequentwords_def_ids = self.tokenizer(self.randomwords)['input_ids'][:len_def]
                frequentwords_def = self.tokenizer.decode(frequentwords_def_ids)
                definition = frequentwords_def
                
            
            # try to add positive examples.
            pos_examples = []
            for idx, pos_example in enumerate(instance["Positive Examples"][:num_pos_examples]):
                pos_example_str = ''
                if self.corruption != 'input-empty':
                    pos_example_str = f" Positive Example {idx+1} -\n"
                    if self.corruption != 'input-oodrandom':
                        pos_example_str += f"Input: {pos_example['input'].strip()}"
                    else:
                        ood_corpus = open('src/dataforcorruptions/corpus.txt','r')
                        raw_lines = ood_corpus.readlines()
                        random.seed(1)
                        ood_input = random.choice(raw_lines)
                        pos_example_str += f"Input: {ood_input}"

                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."
                    pos_example_str += "\n"

                if self.corruption != 'label-empty':
                    pos_example_output = f" Output: {pos_example['output'].strip()}"
                    
                    if self.corruption.startswith("label-random-labelspace") and instance['Task'] in dataset_labels.LABELS:
                        correct_output = pos_example['output']
                        list_of_labels = dataset_labels.LABELS[instance['Task']].copy()
                        list_of_labels.remove(correct_output)
                        if len(list_of_labels)==1:
                            pos_example_output = f" Output: {list_of_labels[0]}"
                        else:
                            random.seed(1)
                            pos_example_output = f" Output: {random.choice(list_of_labels)}"

                        if self.corruption == 'label-random-labelspace-half' and idx%2==0:
                            pos_example_output = f" Output: {pos_example['output'].strip()}"
                    
                    if self.corruption == 'label-randomwords' and instance['Task'] in dataset_labels.LABELS:
                        len_label_space = len(dataset_labels.LABELS[instance['Task']])
                        randomwords_label_space = self.randomwords.split(' ')[:len_label_space]
                        random.seed(1)
                        pos_example_output = f" Output: {random.choice(randomwords_label_space)}"

                    pos_example_str += pos_example_output
                
                
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n" 
                if add_explanation and "explanation" in pos_example:
                    pos_example_str += f" Explanation: {pos_example['explanation'].strip()}"
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."
                    pos_example_str += "\n"
                pos_example_str += "\n"
                if len(self.tokenizer(definition + " ".join(pos_examples) + pos_example_str + task_input)["input_ids"]) <= self.max_source_length:
                    pos_examples.append(pos_example_str)
                else:
                    break
            
            # try to add negative examples.
            neg_examples = []
            for idx, neg_example in enumerate(instance["Negative Examples"][:num_neg_examples]):
                neg_example_str = f" Negative Example {idx+1} -\n"
                neg_example_str += f"Input: {neg_example['input'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                neg_example_str += f" Output: {neg_example['output'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                if add_explanation and "explanation" in neg_example:
                    neg_example_str += f" Explanation: {neg_example['explanation'].strip()}"
                    if not neg_example_str[-1] in string.punctuation:
                        neg_example_str += "."
                    neg_example_str += "\n"
                neg_example_str += "\n"
                if len(self.tokenizer(definition + " ".join(pos_examples) + " ".join(neg_examples) + neg_example_str + task_input)["input_ids"]) <= self.max_source_length:
                    neg_examples.append(neg_example_str)
                else:
                    break 

           
            source = task_name + definition + "".join(pos_examples) + "".join(neg_examples) + task_input # corruption  'inst-placement-before-ex'

            if self.corruption=='inst-placement-after-ex': # corruption  'inst-placement-after-ex'
                source = task_name  + "".join(pos_examples) + "".join(neg_examples) + definition + task_input  # Instrcution is after the in-context examples.
            
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))

        if self.text_only:
            model_inputs = {"id":id, "Task":task, "Categories":categories ,"Reasoning":reasoning, "inputs": sources} # edited by Namrata Shivagunde
            # model_inputs = {"inputs": sources}
        else:
            model_inputs = self.tokenizer(
                sources, 
                max_length=self.max_source_length, 
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)

        if "output" in batch[0]["Instance"] and batch[0]["Instance"]["output"]:
            # Randomly select one reference if multiple are provided.
            labels = [random.choice(ex["Instance"]["output"]) for ex in batch]
            if self.text_only:
                model_inputs["labels"] = labels
            else:
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        labels,
                        max_length=self.max_target_length,
                        padding=self.padding,
                        return_tensors=self.return_tensors,
                        truncation=True,
                        pad_to_multiple_of=self.pad_to_multiple_of
                    )
                label_mask = labels["attention_mask"].bool()
                model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)
        else:
            model_inputs["labels"] = None

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels") and not self.text_only:
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
            model_inputs["decoder_input_ids"] = decoder_input_ids
            
        return model_inputs