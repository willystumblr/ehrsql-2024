from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
from accelerate import Accelerator
from peft import PeftModel
import gc

class Model():
    def __init__(
        self,
        classifier:str, # path-like, or hf repo
        generator:str,  # path-like, or hf repo
        cls_config=None,
        gen_config=None,
        args=None,
        **kwargs
    ):
        if not cls_config:
            cls_config = dict(
                id2label={0: "unanswerable", 1: "answerable"},
                num_labels=2,
                label2id={"unanswerable":0, "answerable":1}
            )
        if not gen_config:
            gen_config = dict(
            device_map={"":Accelerator().local_process_index},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if args.bf16 else "auto",
            use_cache=False,
        )
        model_cls = AutoModelForSequenceClassification.from_pretrained(args.base_model_name,                                                                             
                                                                            **cls_config)
        self.model_cls = PeftModel.from_pretrained(model_cls, classifier)
        self.model_gen = AutoModelForCausalLM.from_pretrained(generator, **gen_config)
        self.args = args
        # print(self.model_cls)
        
    def to(self, device):
        self.model_cls.to(device)
        self.model_gen.to(device)
        
    def eval(self):
        self.model_cls.eval()
        self.model_gen.eval()
        
    def generate(
        self,
        cls_input_ids,
        gen_input_ids,
        gen_config,
        output_scores=True,
        return_dict_in_generate=True,
        # **kwargs
    ):
        # print(cls_input_ids)
        input_ids, attention_mask = cls_input_ids['input_ids'].to(self.args.device), cls_input_ids['attention_mask'].to(self.args.device)
        cls_outputs = self.model_cls(
                                     input_ids=input_ids,
                                     attention_mask=attention_mask
                                     )
        # print(cls_outputs)
        # Get the predicted class probabilities
        probabilities = cls_outputs.logits.softmax(dim=-1)

        # Get the predicted label
        predicted_label_id = probabilities.argmax(dim=-1)
        # print(predicted_label_id)
        # predicted_labels = [self.model_cls.config.id2label[label_id.item()] for label_id in predicted_label_id]

        outputs = []
        for is_answerable in predicted_label_id:
            if is_answerable:
                with torch.inference_mode():
                    generated_outputs = self.model_gen.generate(
                        input_ids=gen_input_ids,
                        output_scores=output_scores,
                        return_dict_in_generate=return_dict_in_generate,
                        **gen_config
                    )
                generated_outputs['type']='text2sql'
                # print(generated_outputs['type'])
                outputs.append(generated_outputs)
            else:
                # print(cls_outputs)
                cls_outputs['type'] = 'answerability'
                outputs.append(cls_outputs)
        # print(outputs)        
        gc.collect()
        torch.cuda.empty_cache()
        return outputs