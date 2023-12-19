import os
from datetime import datetime


class Config(object):
    def __init__(self):
        self.model = "SGANet"

        # pre_processing method
        self.pre_processing = 'sift-2000'

        # data related
        self.dataset = 'yfcc'  # yfcc or sun3d
        self.data_te = './data/yfcc-sift-2000-test.hdf5'  # yfcc-sift-2000-test.hdf5 or sun3d-sift-2000-test.hdf5
        self.data_tr = './data/yfcc-sift-2000-train.hdf5'  # yfcc-sift-2000-train.hdf5 or sun3d-sift-2000-train.hdf5
        self.data_va = './data/yfcc-sift-2000-val.hdf5'  # yfcc-sift-2000-val.hdf5 or or sun3d-sift-2000-val.hdf5
        self.data_te_k = './data/yfcc-sift-2000-testknown.hdf5'

        # network related
        self.use_fundamental = False  # train fundamental matrix estimation
        self.use_ratio = 0  # use ratio test. 0: not use, 1: use before network, 2: use as side information
        self.use_mutual = 0  # use mutual nearest neighbor check. 0: not use, 1: use before network, 2: use as side information
        self.ratio_test_th = 0.8  # ratio test threshold
        self.sr = 0.5
        self.thr = 3e-5

        # loss related
        self.geo_loss_margin = 0.1  # clamping margin in geometry loss
        self.ess_loss_margin = 0.1  # clamping margin in contrastive loss
        self.loss_classif = 1.0  # weight of the classification loss
        self.loss_essential = 0.5  # weight of the essential loss
        self.weight_decay = 0  # l2 decay
        self.momentum = 0.9

        # objective related
        self.obj_geod_th = 1e-4  # theshold for the good geodesic distance
        self.obj_geod_type = 'episym'  # type of geodesic distance
        self.obj_num_kp = 2000  # number of keypoints per image
        self.obj_top_k = -1  # number of keypoints above the threshold to use for essential matrix estimation, put -1 to use all

        # training related
        self.num_workers = 8
        self.canonical_bs = 32
        self.canonical_lr = 1e-3
        self.writer_dir = os.path.join('runs', datetime.now().strftime("[" + self.model + "]-" + "[%Y_%m_%d]-[%H_%M_%S]-[debugging]"))
        self.epochs = 29  # yfcc: 29 epochs is approximately equal to 500k iterations; sun3d: 16 epochs is approximately equal to 500k iterations
        self.checkpoint_path = './checkpoint/' + self.model + '/'
        self.resume = './checkpoint/' + self.model + '/checkpoint0.pth'
        if self.use_fundamental:
            self.best_model_path = './best_model/' + self.model + '/' + self.dataset + '/fundamental/' + self.pre_processing + '/'
        else:
            self.best_model_path = './best_model/' + self.model + '/' + self.dataset + '/essential/' + self.pre_processing + '/'

        # testing related
        self.use_ransac_auc = False  # use ransac when testing
        self.use_ransac_map = False  # use ransac when testing
        self.tag = 'epi'  # logit or epi
        self.post_processing = 'RANSAC'  # RANSAC, PROSAC, MAGSAC

        # loss related
        self.loss_essential_init_iter = int(self.canonical_bs * 20000 // self.canonical_bs)  # initial iterations to run only the classification loss

        # multi gpu info
        self.device = 'cuda'
        self.rank = 0
        self.world_size = 1
        self.gpu = 0
        self.distributed = False
        self.dist_backend = 'nccl'
        self.dist_url = 'env://'
