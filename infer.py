import argparse
import json
import os

import vit_rollout

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pickle
from torchvision import transforms
from dataset.augmentation import get_transform
# from dataset.multi_label.coco import COCO14
from metrics.pedestrian_metrics import get_pedestrian_metrics
from models.backbone import swin_transformer2
from models.model_factory import build_backbone, build_classifier

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import cfg, update_config
from dataset.pedes_attr.pedes import PedesAttr,PedesAttrPETA
from metrics.ml_metrics import get_map_metrics, get_multilabel_metrics
from models.base_block import FeatClassifier
# from models.model_factory import model_dict, classifier_dict
from PIL import Image
from tools.function import get_model_log_path, get_reload_weight
from tools.utils import set_seed, str2bool, time_str
from losses import bceloss, scaledbceloss
import cv2
import matplotlib.pyplot as plt
from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout

set_seed(605)
# # 新增GradCAM类
# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.gradients = []
#         self.activations = []
#
#         target_layer.register_forward_hook(self.save_activation)
#         target_layer.register_backward_hook(self.save_gradient)
#
#     def save_activation(self, module, input, output):
#         self.activations.append(output.detach())
#
#     def save_gradient(self, module, grad_input, grad_output):
#         self.gradients.append(grad_output[0].detach())
#
#     def get_cam(self, target_class=None):
#         gradients = self.gradients[-1].cpu().numpy()
#         activations = self.activations[-1].cpu().numpy()
#
#         weights = np.mean(gradients, axis=(2, 3), keepdims=True)
#         cam = np.sum(weights * activations, axis=1)
#         cam = np.maximum(cam, 0)  # ReLU
#         cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)
#         return cam
#
#     # 在GradCAM类中添加
#     def get_vit_cam(self):
#         gradients = self.gradients[-1].mean(dim=2)  # [B, num_patches]
#         activations = self.activations[-1][:, 1:]  # 忽略class token
#         cam = (gradients @ activations.transpose(1, 2)).squeeze()
#         return cam.reshape(-1, 18, 9)  # 假设14x14 patch
def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def main(cfg, args):
    exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)
    model_dir, log_dir = get_model_log_path(exp_dir, cfg.NAME)

    train_tsfm, valid_tsfm = get_transform(cfg)
    print(valid_tsfm)
    train_set = PedesAttr(cfg=cfg, split=cfg.DATASET.TRAIN_SPLIT, transform=train_tsfm,
                          target_transform=cfg.DATASET.TARGETTRANSFORM,
                          root_path="/data/2Tssd/yuhan/project/dataset/data/PA100k/release_data/release_data")

    valid_set = PedesAttr(cfg=cfg, split=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
                          target_transform=cfg.DATASET.TARGETTRANSFORM,
                          root_path="/data/2Tssd/yuhan/project/dataset/data/PA100k/release_data/release_data")

    # train_set = PedesAttrPETA(cfg=cfg, split=cfg.DATASET.TRAIN_SPLIT, transform=valid_tsfm,
    #                         target_transform=cfg.DATASET.TARGETTRANSFORM)
    # valid_set = PedesAttrPETA(cfg=cfg, split=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
    #                         target_transform=cfg.DATASET.TARGETTRANSFORM)


    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=64,
        pin_memory=True,
    )

    print(f'{cfg.DATASET.TRAIN_SPLIT} set: {len(train_loader.dataset)}, '
          f'{cfg.DATASET.TEST_SPLIT} set: {len(valid_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')

    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)


    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        nattr=train_set.attr_num,
        c_in=1024,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale =cfg.CLASSIFIER.SCALE
    )

    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    model = get_reload_weight(model_dir, model, pth='/data/2Tssd/yuhan/project/CT2_duibi/exp_result/PA100k/swin_b.sm08/img_model/ckpt_max_multi_evavit_swinT_pa100k.pth')
    # 反归一化参数（根据实际预处理设置）
    IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    IMG_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    model.eval()
    preds_probs = []
    gt_list = []
    path_list = []

    attn_list = []
    # grad_rollout = VITAttentionGradRollout(model.module, model.module.backbone.vit.blocks[-1].attn.attn_drop, 0.9)
    attention_rollout = VITAttentionRollout(model.module, model.module.backbone.vit.blocks[5].attn.attn_drop)
    # attention_rollout = VITAttentionRollout(model.module, model.module.classifier.cross_att.attn_drop)
    # 初始化GradCAM
    # grad_cam = GradCAM(model.module, model.module.backbone.vit.blocks[-1])
    with torch.enable_grad():
        for step, (imgs, gt_label, imgname) in enumerate(tqdm(valid_loader)):
            imgs = imgs.cuda().requires_grad_(True)
            gt_label = gt_label.cuda()
            valid_logits, attns = model(imgs, gt_label) # 这里显示的就是热力图

            valid_probs = torch.sigmoid(valid_logits[0]) # 没有什么软用

            # 反向传播计算梯度
            model.zero_grad()
            batch_size, num_attr = valid_probs.shape
            for i in range(batch_size):
                # img = imgs[i]
                transform = transforms.Compose([
                    transforms.Resize((128, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
                img = Image.open(os.path.join('/data/2Tssd/yuhan/project/dataset/data/PA100k/release_data/release_data',imgname[i]))
                img = img.resize((128, 256))
                input_tensor = transform(img).unsqueeze(0)
                # input_tensor = input_tensor.cuda()
                # img = img.resize((256, 128))
                # # 选择概率最高的属性
                # target_class = torch.argmax(valid_probs[i]).item()
                # loss = valid_probs[i, target_class]
                # loss.backward(retain_graph=True)
                # # valid_probs[i, target_class].backward(retain_graph=True)

                # 获取CAM
                # cam = grad_cam.get_vit_cam()[0]

                # a = (valid_logits[0])[i]
                # 探究的激活位置应该是哪里？？真的应该是logits嘛？
                # mask = grad_rollout(input_tensor, 2, (valid_logits[0])[i].unsqueeze(0),i)

                vit_rollout.rollout(attns[i], 0.9, 'mean', i)
                # mask = attention_rollout(input_tensor,(valid_logits[0])[i].unsqueeze(0),i)
                name = "attention_rollout_{:.3f}_{}.png".format(0.5, 'mean')
                # name = "grad_rollout_{}_{:.3f}_{}.png".format(2, 0.9, 'mean')

                np_img = np.array(img)[:, :, ::-1]
                mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
                mask = show_mask_on_image(np_img, mask)
                # cv2.imshow("Input Image", np_img)
                # cv2.imshow(name, mask)
                cv2.imwrite("input.png", np_img)
                cv2.imwrite(name, mask)
                cv2.waitKey(-1)
                # 反向传播计算梯度
                model.zero_grad()


                # # 保存结果
                # base_name = os.path.basename(imgname[i]).split('.')[0]
                # cv2.imwrite(os.path.join("./logs_save", f"{base_name}_cam.jpg"),
                #             np.uint8(255*cam))

            path_list.extend(imgname)
            gt_list.append(gt_label.cpu().numpy())
            preds_probs.append(valid_probs.cpu().numpy())
            attn_list.append(attns.cpu().numpy())


    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    attn_list = np.concatenate(attn_list, axis=0)


    if cfg.METRIC.TYPE == 'pedestrian':
        valid_result = get_pedestrian_metrics(gt_label, preds_probs)
        valid_map, _ = get_map_metrics(gt_label, preds_probs)

        print(f'Evaluation on test set, \n',
              'ma: {:.4f},  map: {:.4f}, label_f1: {:4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                  valid_result.ma, valid_map, np.mean(valid_result.label_f1), np.mean(valid_result.label_pos_recall),
                  np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1)
              )

        with open(os.path.join(model_dir, 'results_test_feat_best.pkl'), 'wb+') as f:
            pickle.dump([valid_result, gt_label, preds_probs, attn_list, path_list], f, protocol=4)

    elif cfg.METRIC.TYPE == 'multi_label':
        if not cfg.INFER.SAMPLING:
            valid_metric = get_multilabel_metrics(gt_label, preds_probs)

            print(
                'Performance : mAP: {:.4f}, OP: {:.4f}, OR: {:.4f}, OF1: {:.4f} CP: {:.4f}, CR: {:.4f}, '
                'CF1: {:.4f}'.format(valid_metric.map, valid_metric.OP, valid_metric.OR, valid_metric.OF1,
                                     valid_metric.CP, valid_metric.CR, valid_metric.CF1))

            with open(os.path.join(model_dir, 'results_train_feat_baseline.pkl'), 'wb+') as f:
                pickle.dump([valid_metric, gt_label, preds_probs, attn_list, path_list], f, protocol=4)

        print(f'{time_str()}')
        print('-' * 60)

def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", default='./configs/pa100k.yaml', help="decide which cfg to use", type=str,
    )
    parser.add_argument("--debug", type=str2bool, default="true")

    args = parser.parse_args()

    return args




if __name__ == '__main__':
    args = argument_parser()
    update_config(cfg, args)

    main(cfg, args)
