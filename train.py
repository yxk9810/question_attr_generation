



import argparse

parser = argparse.ArgumentParser(description='命令行中传入一个数字')
#type是要传入的参数的数据类型  help是该参数的提示信息
parser.add_argument('--model_name', type=str, help='训练后的模型文件')
parser.add_argument('--seed',type=int,default=42, 
                    help='input seed')
parser.add_argument('--batch_size',type=int,default=2, 
                    help='input seed')
parser.add_argument('--epochs',type=int,default=3, 
                    help='input seed')
parser.add_argument("--learning_rate", default=1e-5, type=float)
parser.add_argument("--save_filename", type=str,help='请输入存储文件名')

args = parser.parse_args()
print(args)

import numpy as np
import random
import torch
from transformers import set_seed
import json
set_seed(args.seed)
seed = args.seed 
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
import pandas as pd
import datasets
import json
# import random
from rouge import Rouge
rouge_score = Rouge()
train_data = json.load(open('train_qg_attr_0913.json','r',encoding='utf-8'))
dev_data = json.load(open('dev_qg_attr_0913.json','r',encoding='utf-8'))
train_data =[d for d in train_data if '#' in d['target']]
dev_data =[d for d in dev_data if '#' in d['target']]


print('train size %d'.format(len(train_data)))
print('dev size %d'.format(len(dev_data)))

train_dataset =datasets.Dataset.from_pandas(pd.DataFrame(train_data))
dev_dataset = datasets.Dataset.from_pandas(pd.DataFrame(dev_data))


import os
os.environ["WANDB_DISABLED"] = "true"
checkpoint=args.model_name
from transformers import AutoTokenizer,T5Tokenizer
# from modeling_cpt import CPTForConditionalGeneration
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def preprocess_function(example):
  model_inputs = tokenizer(example['source']+'[SEP]'+example['attr'], truncation=True, padding="max_length", max_length=512)
  labels = tokenizer(text_target=example["target"].replace(' ','').split('#')[1], max_length=128, truncation=True)
  model_inputs['labels'] = labels['input_ids']
  return model_inputs

ds={'train':train_dataset,'validation':dev_dataset}
for split in ds:
  ds[split] = ds[split].map(preprocess_function)

from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

def get_model():
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    return model
import numpy as np
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    print(decoded_preds[0]+'\t'+decoded_labels[0])
    decoded_labels=['no' if t.strip()=='' else t for t in decoded_labels]
    decoded_preds=['no' if t.strip()=='' else t for t in decoded_preds]

    scores = rouge_score.get_scores(decoded_preds,decoded_labels,avg=True)
    for key in scores:
      scores[key] = scores[key]['f']*100
    result = scores
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

training_args = Seq2SeqTrainingArguments(
    output_dir="bart_seq2seq_task9",
    evaluation_strategy="epoch",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=args.epochs,
    predict_with_generate=True,
    fp16=True,
    report_to='tensorboard',
    load_best_model_at_end=True,
    save_strategy='epoch',
    seed=args.seed
)

trainer = Seq2SeqTrainer(
    model_init=get_model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model('./bart_seq2seq_task9')
from transformers import AutoTokenizer
folder='./bart_seq2seq_task9'
tokenizer = AutoTokenizer.from_pretrained(folder)
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(folder)

import torch
device = torch.device('cuda')
model.to(device)

from tqdm import tqdm
best_precision = 0
best_beam = 0
preds = []
def inference(model,inputs,max_s_length=512,max_target_length=128):
    padding=True
    model_inputs = tokenizer(inputs,max_length=max_s_length, padding=padding, truncation=True,return_tensors='pt').input_ids
    outputs= model.generate(model_inputs.to(device),max_length=max_target_length)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def inference_with_file(dataset,batch_size=8):
    import json
    results =[]
    batch_data = []
    for idx,data in enumerate(tqdm(dataset)):
      batch_data.append(data['source']+'[SEP]'+data['attr'])
      if len(batch_data)==batch_size:
          batch_preds = inference(model,batch_data)
          batch_data = []
          results.extend(batch_preds)
    if len(batch_data)>0:
      batch_preds = inference(model,batch_data)
      results.extend(batch_preds)
    return results

preds = inference_with_file(dev_data)
total = 0.0
found = 0.0
right = 0.0
for pred,d in zip(preds,dev_data):
  if '#' not in d['target']:continue
  pred_attr = pred.replace(' ','') #+'\t'+d['target'])
  target = d['target']
  if pred_attr:
    found+=1
    if '#' in target:
      target_attrs = [attr for attr in target.split('#')[1].split(';')]
      pred_attrs = [attr for attr in pred_attr.split(';')]
      total+=len(target_attrs)
      found+=len(pred_attrs)
      for p in pred_attrs:
        if p in target_attrs:
          right+=1
p = float(right)/found
r = float(right)/total
print('p='+str(p)+'\t'+str(r)+'\t'+str(2*p*r/(p+r)))

multi_queries = set(json.load(open('test_multi_questions.json','r',encoding='utf-8')))
test_data = [json.loads(line.strip()) for line in open('test_data_qg_attr_0914.jsonl','r',encoding='utf-8').readlines() if json.loads(line.strip())['source'] in multi_queries]
test_preds = inference_with_file(test_data)
id_to_entity = {}
with open('test_data_qg_attr_0914.jsonl','r',encoding='utf-8') as lines:
  for line in lines:
    data = json.loads(line.strip())
    id_to_entity[data['id']] = data['entity']
def save_preds(preds,test_data):
  dataset = []
  for pred,d in zip(preds,test_data):
    pred_attr = pred.replace(' ','') #+'\t'+d['target'])
    if  pred_attr and d['entity'].strip()!='':
      entity = d['entity'].strip()
      for q in pred_attr.split(';'):
        sub_q = entity+'的'+q+'是什么?'
        dataset.append({'id':d['id'],'question':sub_q})
  return dataset
multi_questions = save_preds(test_preds,test_data)
json.dump(multi_questions,open(args.save_filename,'w',encoding='utf-8'),ensure_ascii=False,indent=4)
