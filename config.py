from argparse import Namespace
import torch

def get_opt():
    opt = Namespace()

    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.num_workers = 10

    opt.num_epochs = 12
    opt.learning_rate = 1e-3
    opt.batch_size = 12

    opt.model = "TextScenes"
    opt.description = f"{opt.model}"

    opt.eval_every = 1  # Epochs (Evaluate)
    opt.log_every = 10  # Batches (Log Loss)

    opt.mode = "train" # "train" or "eval"
    opt.resume = False # True or False
    opt.checkpoint = " " #path to checkpoint
    
    opt.hierarchical_mode = 'codependent' #independent or codependent

    # Data Files
    opt.txt_file = " " #path to train txt file
    opt.val_txt_file = " " #path to val txt file
    opt.key_file = " " #for MSLS sequnce keys

    return opt


def opt_to_dict(opt):
    return vars(opt)
