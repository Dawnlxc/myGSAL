import random

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import copy
import math
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore')

from loss import BCEDiceLoss, AdptWeightBCEDiceLoss
from gen_net import Generator
from dsc_net import Discriminator
from config import Configure, Arguments
from dt import DistanceTransform
from loader import get_data, ISIC, check_mkdir, Onehot
from sklearn.metrics import precision_recall_curve, confusion_matrix

from torch.utils.data import DataLoader
from evaluation import evaluate

def validate(val_loader, img_size, G, device, smooth=1e-8):
    G.eval()
    n = len(val_loader)
    acc, dice, jcd, se, sp = 0.0, 0.0, 0.0, 0.0, 0.0
    # loss = 0.0

    with torch.no_grad():
        for i, (img, gt) in enumerate(val_loader):
            img = img.to(device)
            gt = gt.to(device)

            _, _, seg = G(img)
            seg = nn.Sigmoid()(seg)

            gt_flt = torch.flatten(gt)
            pred_flt = torch.flatten(seg)

            precisions, recalls, thresholds = precision_recall_curve(gt_flt.cpu(), pred_flt.cpu())
            f1 = 2*(precisions * recalls) / (precisions + recalls + smooth)
            max_value = np.argmax(f1)
            precision, recall, thres = precisions[max_value], recalls[max_value], thresholds[max_value]
            pred_mask = (pred_flt > thres)
            pred_label = pred_mask*1
            # plt.imshow(pred_label.cpu().numpy().reshape(img_size), cmap='gray')
            # plt.show()

            tn, fp, fn, tp = confusion_matrix(gt_flt.cpu(), pred_label.cpu()).ravel()
            acc += (tp + tn) / (tp + tn + fp + fn + smooth)
            jcd += tp / (tp + fp + fn + smooth)
            dice += 2*tp / (2*tp + fp + fn)
            se += recall
            sp += tn / (tn + fp + smooth)
    acc /= n
    jcd /= n
    dice /= n
    se /= n
    sp /= n
    # print('Acc: {:.4f}, Dice: {:.4f}, Jaccard: {:.4f}, SE: {:.4f}, SP: {:.4f}'.format(acc, dice, jcd, se, sp))
    return acc, dice, jcd, se, sp
def train_demo(args):
    X, y = get_data(args.path, args.img_size)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    train_dataset = ISIC(X_train, y_train)
    val_dataset = ISIC(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=True)

    G = Generator(in_channels=3, out_channels=1)
    D = Discriminator(in_channels=1, out_channels=1, patch_size=(32, 32))

    G_optim = torch.optim.RMSprop(G.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    D_optim = torch.optim.RMSprop(D.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)

    all_errD = torch.zeros(args.n_epochs)
    all_errG = torch.zeros(args.n_epochs)

    loss_bcedice = BCEDiceLoss()
    loss_wbcedice = AdptWeightBCEDiceLoss()

    print('Starting training Loop...')

    best_acc, best_dice, best_jcd, best_se, best_sp = 0.0, 0.0, 0.0, 0.0, 0.0
    path = os.path.join(args.root, 'models')
    check_mkdir(path)
    for n in range(args.n_epochs):
        G.train()
        # Iterate the loader by batch
        for i, data in enumerate(train_loader):
            imgs, gts = data
            imgs.to(device=args.device, dtype=torch.float32)
            gts.to(device=args.device, dtype=torch.float32)
            x_ske, x_bound, x_seg = G(imgs)
            x_ske, x_bound, x_seg = x_ske.detach(), x_bound.detach(), x_seg.detach()
            dt_ske, dt_bound = DistanceTransform()(gts)

            '''
                Train skeleton-like discriminator
                    argmin (BCE(D(x), 1) + BCE(D(G(z)), 1))
            '''
            D.zero_grad()
            real, fake = dt_ske, x_ske
            labels = torch.full((args.batch_size, ), 1., dtype=torch.float, device=args.device)
            D_real, D_fake = D(real).view(-1), D(fake).view(-1)

            errD_real = loss_bcedice(D_real, labels)
            errD_real.backward()

            labels.fill_(0.)
            errD_fake = loss_bcedice(D_fake, labels)
            errD_fake.backward()

            errD = errD_real + errD_fake

            D_optim.step()

            # for p in D.parameters():
            #     p.data.clamp_(-0.05, 0.05)

            G.zero_grad()
            labels.fill_(1.)
            errG_gan = loss_bcedice(D(fake).view(-1), labels)
            errG_local = loss_bcedice(x_ske, dt_ske)
            errG_seg = loss_wbcedice(x_seg, gts)
            errG = errG_gan + args.lamb * (errG_local + errG_seg)
            errG.backward()
            G_optim.step()

            all_errG[n] = errG.item()


        # Model Validation
        acc, dice, jcd, se, sp = validate(val_loader=val_loader, img_size=args.img_size, G=G, device=args.device)
        if best_jcd < jcd:
            best_jcd = jcd
            torch.save(G.state_dict(), os.path.join(path, 'epoch{}_acc{:.4f}_dice{:.4f}_jcd{:.4f}_se{:.4f}_sp{:.4f}.pth'.format(n, acc, dice, jcd, se, sp)))
            print('Model saved at epoch {}, best jaccard = {:.4f}'.format(n, best_jcd))
        print('[epoch {}/{}][errD_real:{:.4f}][errD_fake:{:.4f}][errG:{:.4f}]'.format(n, args.n_epochs, errD_real.item(), errD_fake.item(), errG))
        print('[epoch {}/{}][acc:{:.4f}][dice:{:.4f}][jaccard:{:.4f}][se:{:.4f}][sp:{:.4f}]'.format(n, args.n_epochs, acc, dice, jcd, se, sp))

    return all_errG


def train(args):
    X, y = get_data(args.path, args.img_size)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    train_dataset = ISIC(X_train, y_train)
    val_dataset = ISIC(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=True)

    G = Generator(in_channels=3, out_channels=1).to(args.device)
    D_ske = Discriminator(in_channels=1, out_channels=1, patch_size=(32, 32)).to(args.device)
    D_bound = Discriminator(in_channels=1, out_channels=1, patch_size=(32, 32)).to(args.device)

    # G_optim = torch.optim.RMSprop(G.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    # D_ske_optim = torch.optim.RMSprop(D_ske.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    # D_bound_optim = torch.optim.RMSprop(D_bound.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)

    G_optim = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    D_ske_optim = torch.optim.Adam(D_ske.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    D_bound_optim = torch.optim.Adam(D_bound.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    all_errD_ske = torch.zeros(args.n_epochs)
    all_errD_bound = torch.zeros(args.n_epochs)
    all_errG = torch.zeros(args.n_epochs)

    loss_bcedice = BCEDiceLoss()
    loss_wbcedice = AdptWeightBCEDiceLoss()
    # ske_criterion = nn.BCELoss()
    # bound_criterion = nn.BCELoss()
    # seg_criterion = nn.BCELoss()

    print('Starting training Loop...')

    # G.train()
    # D_ske.train()
    # D_bound.train()

    best_acc, best_dice, best_jcd, best_se, best_sp = 0.0, 0.0, 0.0, 0.0, 0.0
    path = os.path.join(args.root, 'models')
    check_mkdir(path)
    for n in range(args.n_epochs):
        G.train()
        for i, data in enumerate(train_loader):
            imgs, gts = data
            imgs = imgs.to(device=args.device, dtype=torch.float32)
            gts = gts.to(device=args.device, dtype=torch.float32)
            x_ske, x_bound, x_seg = G(imgs)
            x_ske, x_bound, x_seg = x_ske.detach(), x_bound.detach(), x_seg.detach()
            dt_ske, dt_bound = DistanceTransform()(gts)

            # Label inversion
            real_label = random.uniform(0, 0.1)
            fake_label = random.uniform(0.9, 1)

            # Save Skeleton / Boundary GTs
            # for j in range(len(dt_ske)):
            #     dir = os.path.join(args.root, 'DistanceMap', 'skeleton')
            #     check_mkdir(dir)
            #     plt.imsave(os.path.join(dir, f'skeletion{n*args.batch_size+j}.png'), dt_ske[j, 0, :, :], cmap='gray')
            #     dir = os.path.join(args.root, 'DistanceMap', 'boundary')
            #     check_mkdir(dir)
            #     plt.imsave(os.path.join(dir, f'boundary{n*args.batch_size+j}.png'), dt_bound[j, 0, :, :], cmap='gray')

            '''
                Train skeleton-like discriminator
                    argmin (BCE(D_ske(x), real_label) + BCE(D_ske(G(z)), fake_label) + L1)
            '''
            D_ske.zero_grad()
            ske_real, ske_fake = dt_ske, x_ske
            labels = torch.full((args.batch_size, ), real_label, dtype=torch.float, device=args.device)

            # Add noise to real labels
            n_noise = int(args.batch_size * 0.1)
            noise_idx = [random.randint(0, args.batch_size-1) for _ in range(n_noise)]
            labels[noise_idx] = fake_label

            labels.to(args.device)
            D_ske_real_aux, D_ske_real = D_ske(ske_real)
            D_ske_fake_aux, D_ske_fake = D_ske(ske_fake)

            D_ske_real, D_ske_fake, labels = D_ske_real.view(-1).to(args.device), D_ske_fake.view(-1).to(args.device), labels.to(args.device)
            errD_ske_real = loss_bcedice(D_ske_real, labels)
            errD_ske_real.backward(retain_graph=True)

            labels.fill_(fake_label)

            # Add noise to fake labels
            noise_idx = [random.randint(0, args.batch_size-1) for _ in range(n_noise)]
            labels[noise_idx] = real_label

            errD_ske_fake = loss_bcedice(D_ske_fake, labels)
            errD_ske_fake.backward(retain_graph=True)

            # Compute L1 loss for D_ske
            D_ske_real_aux, D_ske_fake_aux = D_ske_real_aux.to(args.device), D_ske_fake_aux.to(args.device)
            errD_ske_l1 = torch.mean(torch.abs(D_ske_real_aux - D_ske_fake_aux))
            errD_ske_l1.backward()

            errD_ske = errD_ske_real + errD_ske_fake + errD_ske_l1

            D_ske_optim.step()
            '''
                Train boundary-like discriminator
                    argmin (BCE(D_bound(x), real_label) + BCE(D_bound(G(z)), fake_label) + L1)
            '''
            D_bound.zero_grad()
            bound_real, bound_fake = dt_bound, x_bound
            labels.fill_(real_label)
            # Add noise to real labels
            noise_idx = [random.randint(0, args.batch_size-1) for _ in range(n_noise)]
            labels[noise_idx] = fake_label

            D_bound_real_aux, D_bound_real = D_bound(bound_real)
            D_bound_fake_aux, D_bound_fake = D_bound(bound_fake)

            D_bound_real, D_bound_fake = D_bound_real.view(-1).to(args.device), D_bound_fake.view(-1).to(args.device)

            errD_bound_real = loss_bcedice(D_bound_real, labels)
            errD_bound_real.backward(retain_graph=True)

            labels.fill_(fake_label)
            # Add noise to fake labels
            noise_idx = [random.randint(0, args.batch_size-1) for _ in range(n_noise)]
            labels[noise_idx] = real_label

            errD_bound_fake = loss_bcedice(D_bound_fake, labels)
            errD_bound_fake.backward(retain_graph=True)

            # Compute L1 loss for D_bound
            D_bound_real_aux, D_bound_fake_aux = D_bound_real_aux.to(args.device), D_bound_fake_aux.to(args.device)
            errD_bound_l1 = torch.mean(torch.abs(D_bound_real_aux - D_bound_fake_aux))
            errD_bound_l1.backward()

            errD_bound = errD_bound_real + errD_bound_fake + errD_bound_l1

            D_bound_optim.step()

            '''
                Train Generator
                    argmin(BCE(D_ske(G(z)), real_label) + BCE(D_bound(G(z)), real_label) + L_local + L_seg)
                    L_local = BCE(x_ske, dt_ske) + BCE(x_bound, dt_bound) 
                    L_seg = WBCE(x_seg, gts)
            '''
            G.zero_grad()
            # Fill hard real label
            labels.fill_(0.)
            errG_ske_gan = loss_bcedice(D_ske(ske_fake)[1].view(-1), labels)
            errG_bound_gan = loss_bcedice(D_bound(bound_fake)[1].view(-1), labels)
            errG_gan = errG_ske_gan + errG_bound_gan
            errG_local = loss_bcedice(x_ske, dt_ske) + loss_bcedice(x_bound, dt_bound)
            errG_seg = loss_wbcedice(x_seg, gts)
            errG = errG_gan + args.lamb * (errG_local + errG_seg)
            errG.backward()
            G_optim.step()

            # D_ske.zero_grad()
            # labels = torch.full((args.batch_size, ), 1., dtype=torch.float, device=args.device)
            # ske_real_outs = D_ske(dt_ske.detach()).view(-1)
            # errD_ske_real = ske_criterion(ske_real_outs, labels)
            # errD_ske_real.backward()
            #
            # labels = labels.fill_(0.)
            # ske_fake_outs = D_ske(x_ske.detach()).view(-1)
            # errD_ske_fake = ske_criterion(ske_fake_outs, labels)
            # errD_ske_fake.backward()
            #
            # errD_ske = errD_ske_real + errD_ske_fake
            # D_ske_optim.step()
            #
            # D_bound.zero_grad()
            # labels = labels.fill_(1.)
            # bound_real_outs = D_bound(dt_bound.detach()).view(-1)
            # errD_bound_real = bound_criterion(bound_real_outs, labels)
            # errD_bound_real.backward()
            #
            # labels = labels.fill_(0.)
            # bound_fake_outs = D_bound(x_bound.detach()).view(-1)
            # errD_bound_fake = bound_criterion(bound_fake_outs, labels)
            # errD_bound_fake.backward()
            #
            # errD_bound = errD_bound_real + errD_bound_fake
            # D_bound_optim.step()
            #
            # for p in D_ske.parameters():
            #     p.data.clamp_(-0.05, 0.05)
            # for p in D_bound.parameters():
            #     p.data.clamp_(-0.05, 0.05)
            #
            # '''
            #     Train Generator
            #         argmin (BCE(D_ske(G(z)), 1) + BCE(D_bound(G(z)), 1) + BCE(G(z), dt_ske) + BCE(G(z), dt_bound))
            # '''
            # G.zero_grad()
            # x_ske, x_bound, x_seg = G(imgs)
            # labels.fill_(1.)
            #
            # # Evaluate Generator ske/bound generation
            # errG_ske = ske_criterion(D_ske(x_ske.detach()).view(-1), labels)
            # errG_bound = bound_criterion(D_bound(x_bound.detach()).view(-1), labels)
            # # Evaluate Generator segmentation ability
            # errG_seg = seg_criterion(x_seg, gts)
            #
            # errG_ske_local = ske_criterion(x_ske, dt_ske)
            # errG_bound_local = bound_criterion(x_bound, dt_bound)
            # errG_local = errG_ske_local + errG_bound_local
            #
            #
            # errG =  errG_ske + errG_bound + args.lamb * (errG_seg + errG_local)
            #
            # errG.backward()
            # G_optim.step()

            # Save loss for the current iteration
            all_errG[n] = errG.item()
            all_errD_ske[n] = errD_ske.item()
            all_errD_bound[n] = errD_bound.item()


        # Model Validation
        acc, dice, jcd, se, sp = validate(val_loader=val_loader, img_size=args.img_size, G=G, device=args.device)
        if n > 0:
            if best_jcd < jcd:
                best_jcd = jcd
                torch.save(G.state_dict(), os.path.join(path, 'epoch{}_acc{:.4f}_dice{:.4f}_jcd{:.4f}_se{:.4f}_sp{:.4f}.pth'.format(n, acc, dice, jcd, se, sp)))
                print('Model saved at epoch {}, best jaccard = {:.4f}'.format(n, best_jcd))
        print('[epoch {}/{}][errD_ske:{:.4f}][errD_bound:{:.4f}][errD_ske_L1:{:.4f}][errD_bound_L1:{:.4f}][errG:{:.4f}]'.format(n, args.n_epochs, errD_ske, errD_bound, errD_ske_l1, errD_bound_l1, errG))
        print('[epoch {}/{}][acc:{:.4f}][dice:{:.4f}][jaccard:{:.4f}][se:{:.4f}][sp:{:.4f}]'.format(n, args.n_epochs, acc, dice, jcd, se, sp))

    return all_errG, all_errD_ske, all_errD_bound

train_path = '/Users/dawn/Desktop/GSAL/data/ISIC2016_Segmentation'
test_path = '/Users/dawn/Desktop/GSAL/data/ISIC2016_Segmentation_Test'
ROOT = '/Users/dawn/Desktop/GSAL'
train_args = Arguments(
    path=train_path,
    root=ROOT,
    img_size=(256, 256),
    n_epochs=100,
    batch_size=5,
    device='cpu',
    lr=0.0001,
    beta1=0.5,
    lamb=1.,
)
test_args = Arguments(
    path=test_path,
    root=ROOT,
    img_size=(256, 256),
    n_epochs=10,
    batch_size=1,
    device='cpu',
    lr=0.001,
    beta1=0.5,
    lamb=0.1,
)
if __name__ == '__main__':
    errG, errD_ske, errD_bound = train(train_args)
    # X_test, y_test = get_data(test_path, test_args.img_size)
    # test_dataset = ISIC(X_test, y_test)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)
    # model = Generator(in_channels=3, out_channels=1)
    # model_path = '/Users/dawn/Desktop/GSAL/models/epoch99_acc0.9791_dice0.9533_jcd0.9107_se0.9359_sp0.9919.pth'
    # model.load_state_dict(torch.load(model_path))
    # # for i, (img, mask) in enumerate(val_loader):
    # #     _, _, seg = model(img)
    # #     seg = F.sigmoid(seg)
    # #     plt.imshow(seg.detach().numpy()[i, 0, :, :], cmap='gray')
    # #     plt.show()
    # #     break
    # acc, dice, jcd, se, sp = evaluate(test_args, test_loader, test_args.img_size, model, test_args.device)
    # print('[RESULT][acc:{:.4f}][dice:{:.4f}][jaccard:{:.4f}][se:{:.4f}][sp:{:.4f}]'.format(acc, dice, jcd, se, sp))
    fig = plt.figure(figsize=(10, 5))
    plt.plot(errG, label='Generator')
    plt.plot(errD_ske, label='Skeleton Discriminator')
    plt.plot(errD_bound, label='Boundary Discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    check_mkdir(os.path.join(ROOT, 'figures'))
    plt.savefig(os.path.join(ROOT, 'figures', 'LossCurve.png'))
    plt.show()