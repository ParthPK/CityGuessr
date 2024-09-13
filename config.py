from argparse import Namespace
import torch

def get_opt():
    opt = Namespace()

    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.num_workers = 10

    opt.num_epochs = 30
    opt.learning_rate = 1e-3
    opt.batch_size = 12

    opt.model = "VanillaTextScenes"
    opt.description = f"{opt.model}"

    opt.eval_every = 1  # Epochs (Evaluate)
    opt.log_every = 10  # Batches (Log Loss)

    opt.mode = "eval" # "train" or "eval"
    opt.resume = True # True or False
    opt.checkpoint = "model_best.pth.tar"
    
    opt.hierarchical_mode = 'independent'

    # Data Files
    #opt.txt_file = "misc/train_sf.txt"
    opt.txt_file = "misc/train_mean_text_soft_w_16_scenes.txt"
    #opt.val_txt_file = "misc/test.txt"
    opt.val_txt_file = "misc/val_labels_w_16_scenes.txt"
    #opt.val_txt_file = "misc/mapilliary_val_majority_w_16_scenes.txt"
    opt.key_file = "misc/mapilliary_sequence_keys.txt"

    return opt


def opt_to_dict(opt):
    return vars(opt)
