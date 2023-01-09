import logging
import random
import string
from transformers.data.data_collator import *


logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForNI:

    tokenizer: PreTrainedTokenizerBase
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
    text_only: bool=False
    corruption: str='baseline'
    corruption_type: str='data' # can be 'data' or 'placement'

    
    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []
        id = []
        task = []
        categories = []
        reasoning = []
        for instance in batch:
        
            id.append(instance['id'])
            task.append(instance['Task'])
            categories.append(instance['Categories'])
            reasoning.append(instance['Reasoning'])

            add_task_name = self.add_task_name
            add_task_definition = self.add_task_definition
            num_pos_examples = self.num_pos_examples
            num_neg_examples = self.num_neg_examples
            add_explanation = self.add_explanation 

            # Creating the prompt
            task_input = ""

            # test input
            task_input += f"{instance['Instance']['input'].strip()}"
            if not task_input[-1] in string.punctuation:
                task_input += "."
            
            # Instruction
            definition = ""
            if add_task_definition:
                definition = instance["Definition"][0].strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                definition += "\n\n"
            
            # Add positive examples.
            pos_examples = []
            for _ , pos_example in enumerate(instance["Positive Examples"][:num_pos_examples]):
                pos_example_str = ''
                if self.corruption != 'input_empty':
                    pos_example_str += pos_example['input'].strip()
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."
        
                if self.corruption != 'label-empty':
                    pos_example_output = pos_example['output'].strip()
                  
                    pos_example_str += pos_example_output
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."

                pos_example_str += "\n\n" 

                if len(self.tokenizer(definition + " ".join(pos_examples) + pos_example_str + task_input)["input_ids"]) <= self.max_source_length:
                    pos_examples.append(pos_example_str)
                else:
                    break
            
            # Combine all elements
            if self.corruption == 'instr-placement-after-ex':
                source =  "".join(pos_examples) + definition + task_input 
            else:    
                source = definition + "".join(pos_examples) + task_input 

            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))

            labels = [random.choice(ex["Instance"]["output"]) for ex in batch]
       
            model_inputs = {"id":id, "Task":task, "Categories":categories ,"Reasoning":reasoning, "inputs": sources, "labels":labels}

        return model_inputs