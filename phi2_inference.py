# Empty VRAM and clear model, trainer variables
# try: 
#     del model
#     del tokenizer
#     del trainer
#     import gc
#     gc.collect()
# except:
#     pass

# load libraries for inference
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


if __name__ == "__main__":
    # memory cleared so recreate parameters
    baseModelName:str="microsoft/phi-2"
    project:str="Finetune"
    max_steps:int=1000

    modelName:str=baseModelName.replace("\\", "_").replace("/", "_")
    run_name:str=f"{project}-{modelName}"
    output_dir:str="./" + run_name # this will be the dir to store run infomation and model weights

    # reload our base model and tokeniser
    modelInference:object=AutoModelForCausalLM.from_pretrained(
        baseModelName,  # Phi2, same as before
        torch_dtype=torch.float32, # fixes issue in inference related to float16 values producing "!!!!" rather than output.
        device_map="auto",                                      
        trust_remote_code=True,
        load_in_8bit=True,
    )
    tokenizerInference:object=AutoTokenizer.from_pretrained(baseModelName,
                                                add_bos_token=True,
                                                trust_remote_code=True,
                                                use_fast=False)
    tokenizerInference.pad_token = tokenizerInference.eos_token

    # load finetuned QLoRA adapters which were saved during training
    finetunedFolder:str=f"{output_dir}/checkpoint-{max_steps}" # get latest model by default (can change if you see better performance on other models)
    FTmodel:object=PeftModel.from_pretrained(modelInference, finetunedFolder) # load FT model

    # model hyperparameters
    repetition_penalty:float=1.0
    max_tokens:int=200

    # test a prompt
    testPrompt:dict={
          "Title": "Pay rent/mortgage on time.",
          "Details": {
            "Date": "2024/06/04",
            "Time": "22:53",
            "Location": "",
            "Priority": "Medium"
          }
       }

    formattedPrompt:str=f"Data: {testPrompt}\nQuestion: " # format like training set formatting, see above.
    tokenisedPrompt:dict=tokenizerInference(formattedPrompt, return_tensors="pt").to("cuda") # tokenise prompt
    FTmodel.eval() # set in inference mode
    with torch.no_grad():
        response:str=tokenizerInference.decode(FTmodel.generate(**tokenisedPrompt, max_new_tokens=max_tokens, repetition_penalty=repetition_penalty)[0], skip_special_tokens=True)
        print(response)