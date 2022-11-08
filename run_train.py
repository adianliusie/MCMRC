import argparse
import os
import shutil
import pprint

from src.handlers.trainer import Trainer
from src.utils.general import save_script_args

#== ArgParse for Model details =================================================================================
model_parser = argparse.ArgumentParser(description='Arguments for system and model configuration')

temp_dir = 'trained_models/temp'
group = model_parser.add_mutually_exclusive_group(required=True)
group.add_argument('--path', type=str,         help='name to save the experiment as')
group.add_argument('--temp', action='store_const',  const=temp_dir, dest='path', help='if set, the exp_name is temp')

model_parser.add_argument('--transformer',  default='electra_large', type=str,  help='[bert, roberta, electra ...]')
model_parser.add_argument('--formatting',   default='standard',     type=str,  help='[standard, Q, O, QO]')
model_parser.add_argument('--max-len',      default=512,            type=int,  help='max length of transformer inputs')

model_parser.add_argument('--num-seeds',  default=1,            type=int,  help='number of seeds to train')
model_parser.add_argument('--seed-num',   default=None,         type=int,  help='if an extra seed is to be trained')
model_parser.add_argument('--force',      action='store_true',  help='if set, will overwrite any existing directory')

#== ArgParse for Training details ==============================================================================
train_parser = argparse.ArgumentParser(description='Arguments for training the system')

train_parser.add_argument('--data-set',  default='race',  type=str,  help='')
train_parser.add_argument('--lim',       default=None,    type=int, help='size of data subset to use (for debugging)')
train_parser.add_argument('--print-len', default=1200,     type=int,  help='logging training print size')

train_parser.add_argument('--epochs',   default=10,    type=int,     help='numer of epochs to train')
train_parser.add_argument('--lr',       default=2e-6,  type=float,   help='training learning rate')
train_parser.add_argument('--bsz',      default=4,     type=int,     help='training batch size')
train_parser.add_argument('--epsilon',  default=1e-6,  type=str,     help='Specify the AdamW loss epsilon')

train_parser.add_argument('--optim',   default='adamw', type=str,  help='[adam, adamw, sgd]')
train_parser.add_argument('--wandb',   default=None,    type=str,  help='experiment name to use for wandb (and to enable)')
train_parser.add_argument('--device',  default='cuda',  type=str,  help='device to use [cuda, cpu]')

train_parser.add_argument('--no-save', action='store_false', dest='save', help='whether to not save model')

#== Main training Script =======================================================================================

if __name__ == '__main__':
    model_args, other_args_1 = model_parser.parse_known_args()
    train_args, other_args_2 = train_parser.parse_known_args()
    assert set(other_args_1).isdisjoint(other_args_2), f"{set(other_args_1) & set(other_args_2)}"
    
    save_script_args()
    pprint.pprint(model_args.__dict__), print()
    pprint.pprint(train_args.__dict__), print()
    
    # Overwrites directory if it exists
    if model_args.force:
        path = model_args.path
        exp_folders = path.split('/')
        if exp_folders[0] == 'trained_models' and os.path.isdir(path) and len(exp_folders)>=2:
            shutil.rmtree(path)
        else:
            print('cant overwrite path')

    # If extra seed, set
    if model_args.seed_num:
        seed_path = model_args.path+'/'+str(model_args.seed_num)
        assert(os.path.isdir(model_args.path) and not os.path.isdir(seed_path))
        trainer = Trainer(seed_path, model_args)
        trainer.train(train_args)

    # Normal training routine
    else:
        for i in range(model_args.num_seeds):
            path = model_args.path + '/' + str(i)
            trainer = Trainer(path, model_args)
            trainer.train(train_args)
        
    
