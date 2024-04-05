from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
from accelerate import Accelerator

class Model():
    def __init__(
        self,
        classifier:str, # path-like, or hf repo
        generator:str,  # path-like, or hf repo
        cls_config,
        gen_config,
        *args,
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
        self.model_cls = AutoModelForSequenceClassification.from_pretrained(classifier,                                                                             
                                                                            **cls_config)
        self.model_gen = AutoModelForCausalLM.from_pretrained(generator, **gen_config)
        
    def to(self, device):
        self.model_cls.to(device)
        self.model_gen.to(device)
        
    def eval(self):
        self.model_cls.eval()
        self.model_gen.eval()
        
    def generate(
        self,
        input_ids,
        gen_config,
        output_scores=True,
        return_dict_in_generate=True,
    ):
        cls_input_ids, gen_input_ids = input_ids
        cls_outputs = self.model_cls(cls_input_ids)
        
        labels = cls_outputs.logits.argmax(-1).tolist() # batch_size MUST BE 1
        
        outputs = []
        for is_answerable in labels:
            if is_answerable:
                with torch.inference_mode():
                    generated_outputs = self.model_gen.generate(
                        input_ids=gen_input_ids,
                        output_scores=output_scores,
                        return_dict_in_generate=return_dict_in_generate,
                        **gen_config
                    )
                    generated_outputs["type"] = 'text2sql'
                    outputs.append(generated_outputs)
            else:
                cls_outputs['type'] = 'answerbility'
                outputs.append(cls_outputs)
            
        return outputs