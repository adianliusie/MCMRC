# MCMRC

### Dependencies
You'll need to install transformers, torch, wandb, scipy, 

### Training
To train a standard Multiple Choice Reading Comprehension system, run the command

```
python run_train.py --exp_path 'path_to_save_model' --data_set race 
```

To train a standard Multiple Choice Reading Comprehension system that uses only the question and options run

```
python run_train.py --exp_path 'path_to_save_model' --data_set race --formatting O 
```


