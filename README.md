# MCMRC

### Dependencies
You'll need to install transformers, torch, wandb, scipy

### Training
To train a standard Multiple Choice Reading Comprehension system, run the command

```
python run_train.py --path 'path_to_save_model' --data-set race --formatting standard
```

- ```--formatting standard``` uses full context in training. To train with no context, use the argument ```--formatting QO```
- ```--data-set race``` sets the dataset to be race in training. To use other datasets load your own datasets and interface the code with src/data_utils/data_loader
- other training arguments can be modified, look into run_train.py to see what extra arguments can be used.


### Evaluation

To evaluate a trained model, use the command

```
python evaluation.py --path 'path_to_save_model' --data-set race
```
- ```--path 'path_to_save_model'``` is the path to the model to be evaluated. this is the same --path argument used in training
- ```--data-set race``` evaluates the models on race. Again, other datasets can be used if they are loaded into the data_utils. 
- ```--mode dev``` is an optional argument to evaluate other data splits. The default is ```test```

Note that ensemble performance is reported. Evaluation caches results, and so the second time the same evaluation runs, the results are generated (nearly) instantaneously. These means that evaluations are stale and any changes will still have the same cached preds (shouldn't be an issue unless the dataset changes)

