<<<<<<< HEAD
# Training and Testing

## bert-base-uncased:
Training bert model：
```bash
python main.py --train_file dataset/train.json --predict_file dataset/valid.json --model_type bert --model_name_or_path bert-base-uncased --output_dir output/ --version_2_with_negative --do_train --do_eval --do_lower_case --overwrite_output --save_steps 0
```

Testing bert model：
```bash
python main.py --train_file dataset/train.json --predict_file dataset/test.json --model_type bert --model_name_or_path output/ --output_dir output/eval/ --version_2_with_negative --do_eval --do_lower_case
```

## Roberta-base:
Training roberta model：
```bash
python main2.py --train_file dataset/train.json --predict_file dataset/valid.json --model_type roberta --model_name_or_path roberta-base --output_dir output/ --version_2_with_negative --do_train --do_eval --overwrite_output --save_steps 0
```

Testing roberta model：
```bash
python main2.py --train_file dataset/train.json --predict_file dataset/test.json --model_type roberta --model_name_or_path output/ --output_dir output/eval/ --version_2_with_negative --do_eval 
```

# Modified / Added python files
## Modified files: utils_squad.py
Adds the attribute **type_ids** to InputFeatures

In function **convert_examples_to_features**, adds new list type_ids and Assign a value to it during word segmentation. The detailed implementation can be seen in the [report](./report_A2.pdf).

## Modified files: main.py
In function **load_and_cache_examples**, extract the new parameter **type_ids** from the feature. 
```code
all_type_ids = torch.tensor([f.type_ids for f in features], dtype=torch.long)
```
And pass it to the train method and evaluate method.

In function train , New parameters are added to the optimizer group to enable a separate learning of the weights of the input_type_embedding layer.
```code
{'params': model.bert.embeddings.input_type_embeddings.parameters(), 'lr': args.input_type_lr},
```
The equivalent is to add a parameter: **input_type_lr** to the parser.

And pass the newly added parameter input_type_ids to input. (Same in evaluate method)
```code
'input_type_ids': batch[3],
```
## Modified files: modeling_bert.py
In class BertEmbeddings, BertModel and BertForQuestionAnswering, add new forward parameters **input_type_ids** which is corresponding to the 
**type_ids** in util_squad.py.

Besides, label smooth loss function is created and used in BertForQuestionAnswering architecture.

## Added files: Dataset_analysis.py
It is used to calculate the distribution of data in a data set.

## Added files: grid_search.py
Used to find the optimal hyperparameters combination by grid search, and the best model file and prediction file are saved.

## Added files: main2.py - Using roberta model.
All the code is the same as the original code, but with the following package statement:
```code
from transformers import RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer
```
To run this code, transformers are needed to installed:
```bash
pip install transformers
```
The three pre-training files are loaded directly using roberta's classes:
```code
model_name_or_path = args.model_name_or_path
config = RobertaConfig.from_pretrained(model_name_or_path)
tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
model = RobertaForQuestionAnswering.from_pretrained(model_name_or_path, config=config)
```

## Added files: util_squad2.py - Using roberta model.
Consistent with the original code, the reason for creating it is that the input type embedding is added to the squad.py.

=======
# bertQA
>>>>>>> 8149ce8d3d1f77b321a2201c07e20a12dfe9a38f
