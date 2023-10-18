import os
import torch
from config import Config
from torch.utils.data import DataLoader
from models.SGANet import SGANet
from utils.tools import safe_load_weights
from utils.train_eval_utils import evaluate
from dataset.dataset import CorrespondencesDataset, collate_fn


if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    conf = Config()

    # Load data
    test_dataset = CorrespondencesDataset(conf.data_te, conf)

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=False,
                             num_workers=0,
                             collate_fn=collate_fn)

    print('Using {} dataloader workers every process'.format(conf.num_workers))

    # Create model
    model = SGANet(conf).cuda()
    weights_dict = torch.load(os.path.join('./weights', 'best_model_yfcc.pth'), map_location="cuda")
    safe_load_weights(model, weights_dict['state_dict'])

    aucs5, aucs10, aucs20, va_res, precisions, recalls, f_scores = evaluate(model, test_loader, conf, epoch=0)
    va = [aucs5, aucs10, aucs20, va_res[0] * 100, va_res[1] * 100, va_res[3] * 100, precisions * 100, recalls * 100, f_scores * 100]

    output = ''
    name = ["AUC@5", "AUC@10", "AUC@20", "mAP5", "mAP10", "mAP20", "Precisions", "Recalls", "F_scores"]
    for i, j in enumerate(va):
        output += name[i] + ": " + str(j) + "\n"

    print(output)