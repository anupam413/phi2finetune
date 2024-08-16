from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

import warnings, os
warnings.filterwarnings("ignore")

def formattingFunc(textExample:str) -> str:
    """
    This function formats our text to be continuous rather than in json format. The output of this function is submitted directly to phi-2 for finetuning.
    """
    # for question generation based on the context
    text:str=f"Data: {textExample['Data']}\nQuestion: {textExample['Question']}"
    
    # text:str=f"{example['note']}" # if continuous text
    return text

def loadModel(baseModelName:str):
    """_summary_

    Args:
        baseModelName (str): Name of the base model from where the model is to be extracted.

    Returns:
        model (object): base model data after combining all the shards.
        tokenizer (object): tokenizer present with the base model for tokenization.
    """

    # Load our base model
    model:object=AutoModelForCausalLM.from_pretrained(baseModelName,
                                                torch_dtype="auto", # fixes issue in inference related to float16 values producing "!!!!" rather than output.
                                                device_map="cpu",
                                                trust_remote_code=True)

    # Load our tokenizer
    tokenizer:object=AutoTokenizer.from_pretrained(
        baseModelName,
        padding_side="left", # add padding so that our input sequences are all the same length. Left means that pad token is repeated until we reach our input text.
        add_eos_token=True, # end of sequence token
        add_bos_token=True, # beginning of sequence token
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token # set out pad token to be the same as eos token

    return model, tokenizer

def tokenizePrompt(prompt:object) -> dict:
    """
    Tokenizes prompt based on prompt and tokenizer.
    """
    tokenizedPrompt:dict=tokenizer(formattingFunc(prompt))
    return tokenizedPrompt

# this function will set all tokens to the same length using left hand padding and the eos token (setup above)
def tokenizePromptAdjustedLengths(prompt:object):
    """
    Tokenizes prompt with adjusted lengths with left handed padding. All sequences will be of the same length which will assist training.
    """
    tokenizedResponse = tokenizer(
        formattingFunc(prompt),
        truncation=True,
        max_length=maxLengthTokens,
        padding="max_length",
    )
    return tokenizedResponse

def train(model, tokenizedTrain, tokenizedVal):
    """Training Pipeline

    Args:
        model (object): Phi-2 base model.
        tokenizedTrain (dict): Tokenized training dataset
        tokenizedVal (dict): Tokenized testing dataset

    Returns:
        trainer (object): This is used for training the phi2 on basis of the training arguments
    """
    # Setup train run parameters
    project:str="Finetune"
    modelName:str=baseModelName.replace("\\", "_").replace("/", "_")
    run_name:str=f"{project}-{modelName}"
    output_dir:str="./" + run_name # this will be the dir to store run infomation and model weights

    # get GPU count for CUDA.
    print(f"GPU COUNT: {torch.cuda.device_count()}")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to_empty(device)
    
    if torch.cuda.device_count() > 1: # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True
    
    stepsSaveEvalLoss:int=50
    numberStepPartitions:int=20 # stepsSaveEvalLoss muliplied by numberStepPartitions gets max_steps - done so that the last step is always a multiple of stepsSaveEvalLoss and it saves.
    max_steps:int=stepsSaveEvalLoss*numberStepPartitions
    trainer:object=Trainer(
        model=model,
        train_dataset=tokenizedTrain,
        eval_dataset=tokenizedVal,
        args=TrainingArguments(
            output_dir=output_dir, # output dir defined above
            warmup_steps=1, # number of steps for the warmup phase where the learning rate is gradually increased from a low value to the maximum value where normal schedule begins - can improve the stability and performance.
            per_device_train_batch_size=2, # specifies the batch size per device for training. It should be an integer that is greater than zero.
            gradient_accumulation_steps=1, # specifies the number of steps to accumulate gradients before performing a backward and an optimizer step. It should be an integer that is greater than zero. The effective batch size is the product of this argument and the per_device_train_batch_size
            max_steps=max_steps, # max number of training steps
            learning_rate=2.5e-5, # aim for small LR for finetuning scenarios
            optim="adamw_torch", # optimiser type to adjust LR during training
            logging_dir=f"{output_dir}/logs", # Where logs are stored for training
            logging_steps=stepsSaveEvalLoss, # train loss cadence
            do_eval=True, # perform eval on eval set
            evaluation_strategy="steps", # eval model loss set to steps
            eval_steps=stepsSaveEvalLoss, # eval loss cadence
            save_strategy="steps", # checkpoint model progress strategy set to steps
            save_steps=stepsSaveEvalLoss, # save every x steps cadence
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False), # mlm - masked language modeling
    )
    return trainer
 
if __name__ == "__main__":
    # Load directory paths for persisting model

    MODEL_DIR = os.environ["MODEL_DIR"]
    MODEL_FILE_LDA = os.environ["MODEL_FILE_LDA"]
    MODEL_FILE_NN = os.environ["MODEL_FILE_NN"]
    MODEL_PATH_LDA = os.path.join(MODEL_DIR, MODEL_FILE_LDA)
    MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)
    
    
    # Load the dataset
    dataName:str = "train_data.jsonl"
    valName:str = "test_data.jsonl"
    trainDataset, evalDataset = load_dataset ('json', data_files=dataName, split='train'), load_dataset('json', data_files=valName, split='train')
    
    ## baseModel name
    baseModelName:str="microsoft/phi-2"
    model, tokenizer = loadModel(baseModelName)
    
    # Format and Tokenize Dataset
    tokenizedTrain: dict = trainDataset.map(tokenizePrompt)
    tokenizedVal:dict=evalDataset.map(tokenizePrompt)

    
    # counting lengths of both dataset so we can adjust max length
    lengthTokens:list=[len(x['input_ids']) for x in tokenizedTrain] # count lengths of tokenizedTrain
    if tokenizedVal != None:
        lengthTokens += [len(x['input_ids']) for x in tokenizedVal] # count lengths of tokenizedVal
    maxLengthTokens:int=max(lengthTokens) + 2 #  we could also visualise lengthTokens using matplotlib if we wish to see the distribution
    tokenDiffOriginal:int=maxLengthTokens-min(lengthTokens) # create metric original
    
    del tokenizedTrain; del tokenizedVal # clean up old variables
    tokenizedTrain:dict=trainDataset.map(tokenizePromptAdjustedLengths) # apply adjusted size tokenization
    tokenizedVal:dict=evalDataset.map(tokenizePromptAdjustedLengths)

    # count adjusted size difference
    lengthTokens:list=[len(x['input_ids']) for x in tokenizedTrain] # count lengths of tokenizedTrain
    if tokenizedVal != None:
        lengthTokens += [len(x['input_ids']) for x in tokenizedVal] # count lengths of tokenizedVal
    tokenDiffAdjusted:int=max(lengthTokens)-min(lengthTokens) # create metric adjusted

    print(f"| Diff Token Size |\nOriginal Lengths: {tokenDiffOriginal}\nAdjusted Lengths: {tokenDiffAdjusted}") # compare size differences using metrics from original and adjusted lengths.
    
    # Get model information and Set Up LoRA for finetune
    loraConfig:object=LoraConfig(
        r=64, # Rank of low-rank matrix, controls the number of parameters trained - a higher rank allowing more parameters to be trained and larger update matrices (and more compute cost). Play with this and see how it effects number of trainable params.
        lora_alpha=16, # LoRA scaing factor of learned weights: alpha/r
        target_modules=[ # modules (eg attention blocks) to apply LoRA matrices.
            "Wqkv",
            "fc1",
            "fc2",
        ],
        bias="none", # should bias parameters also be trained: none, all, lora_only
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    model:object=get_peft_model(model, loraConfig) # parameter-efficient fine tune - freeze pretrained model parameters and add small number of tunable adapters on top.
    print(f"Model Architecture:\n{model}")
    model.print_trainable_parameters() # print trainable parameters
    
    trainer = train(model, tokenizedTrain, tokenizedVal)
    trainer.train()