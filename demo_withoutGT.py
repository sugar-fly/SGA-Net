import torch
import os
import cv2
import numpy as np
from config import Config
from models.SGANet import SGANet
from utils.tools import safe_load_weights
import matplotlib.pyplot as plt
from utils.demo_utils import get_input_withoutGT


class ExtractSIFT(object):
    def __init__(self, num_kp, contrastThreshold=1e-5):
        self.sift = cv2.SIFT_create(nfeatures=num_kp, contrastThreshold=contrastThreshold)
        self.num_kp = num_kp

    def run(self, img_path):
        img = cv2.imread(img_path)
        h, w = img.shape[0], img.shape[1]
        cv_kp, desc = self.sift.detectAndCompute(img, None)
        kp = np.array([[_kp.pt[0], _kp.pt[1]] for _kp in cv_kp])

        return kp[:self.num_kp], desc[:self.num_kp], w, h


def draw_results(img_path1, img_path2, inlier_pt1, inlier_pt2):
    img1 = cv2.imread(img_path1)
    b, g, r = cv2.split(img1)
    img1 = cv2.merge([r, g, b])
    img2 = cv2.imread(img_path2)
    b, g, r = cv2.split(img2)
    img2 = cv2.merge([r, g, b])

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    vis = np.ones((max(h1, h2), w1 + 25 + w2, 3), np.uint8) * 255
    vis[:h1, :w1] = img1
    vis[:h2, w1 + 25:w1 + w2 + 25] = img2

    fig = plt.figure()
    plt.imshow(vis)
    ax = plt.gca()

    # inlier
    for i in range(inlier_pt1.shape[0]):
        x1 = int(inlier_pt1[i, 0])
        y1 = int(inlier_pt1[i, 1])
        x2 = int(inlier_pt2[i, 0] + w1 + 25)
        y2 = int(inlier_pt2[i, 1])

        ax.add_artist(plt.Circle((x1, y1), radius=2.5, color='#04FE05'))
        ax.add_artist(plt.Circle((x2, y2), radius=2.5, color='#04FE05'))
        ax.plot([x1, x2], [y1, y2], c='#E6E52F', linestyle='-', linewidth=1.5)

    plt.axis('off')
    name = img_path1.split('/')[-1] + img_path2.split('/')[-1]
    save_path = './assets/' + 'withoutGT_' + name  + '.png'
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    # plt.show()
    plt.close("all")


if __name__ == '__main__':
    conf = Config()

    model = SGANet(conf).cuda()
    weight_path = './weights/best_model_yfcc.pth'
    weights_dict = torch.load(os.path.join(weight_path), map_location='cpu')
    safe_load_weights(model, weights_dict['state_dict'])
    model.eval()

    dataset_name = 'yfcc100m'
    scene_name = 'sacre_coeur'
    img_name1 = '90319881_5214536200.jpg' # your image
    img_name2 = '17616986_7085723791.jpg'
    data_path = './assets/' + scene_name + '/test/'

    img_path1 = data_path + 'images/' + img_name1
    img_path2 = data_path + 'images/' + img_name2

    detector = ExtractSIFT(2000)
    kp1, kp2, xs = get_input_withoutGT(detector, img_path1, img_path2)
    xs = torch.from_numpy(xs)[:, None, :, :].float().cuda()
    pesudo_ys = xs[:, 0, :, 0]

    logits, ys_ds, e_hat_list, y_hat, xs_ds = model(xs, pesudo_ys)

    # visualization of results
    predicted_inliers = (logits[-1] >= 0).squeeze().cpu()
    N = kp1.shape[0]
    w0 = torch.sort(logits[1].squeeze(0), dim=-1, descending=True)[1][:int(N * conf.sr)].cpu().numpy().astype(np.int32)
    w1 = torch.sort(logits[3].squeeze(0), dim=-1, descending=True)[1][:int(N * conf.sr**2)].cpu().numpy().astype(np.int32)
    kpts1 = kp1[w0]
    kpts2 = kp2[w0]
    kpts1 = kpts1[w1]
    kpts2 = kpts2[w1]
    draw_results(img_path1, img_path2, kpts1[predicted_inliers], kpts2[predicted_inliers])