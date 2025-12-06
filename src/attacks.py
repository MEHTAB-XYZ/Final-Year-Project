# src/attacks.py

import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------
# FGSM Attack
# ------------------------------
def fgsm_attack(model, img, label, eps=0.01):
    img = img.clone().detach().to(DEVICE)
    label = torch.tensor([label]).to(DEVICE)

    img.requires_grad = True
    logits = model(img)
    loss = nn.CrossEntropyLoss()(logits, label)
    loss.backward()

    perturbed = img + eps * img.grad.sign()
    perturbed = torch.clamp(perturbed, -3, 3)
    return perturbed.detach()


# ------------------------------
# Basic Iterative Method (BIM)
# ------------------------------
def bim_attack(model, img, label, eps=0.03, alpha=0.005, iters=10):
    adv = img.clone().detach().to(DEVICE)
    label = torch.tensor([label]).to(DEVICE)

    for _ in range(iters):
        adv.requires_grad = True
        logits = model(adv)
        loss = nn.CrossEntropyLoss()(logits, label)
        loss.backward()

        adv = adv + alpha * adv.grad.sign()
        eta = torch.clamp(adv - img, min=-eps, max=eps)
        adv = torch.clamp(img + eta, -3, 3).detach()

    return adv


# ------------------------------
# PGD Attack
# ------------------------------
def pgd_attack(model, img, label, eps=0.03, alpha=0.01, iters=20):
    ori = img.clone().detach().to(DEVICE)
    adv = ori + torch.empty_like(ori).uniform_(-eps, eps).to(DEVICE)
    adv = torch.clamp(adv, -3, 3)
    label = torch.tensor([label]).to(DEVICE)

    for _ in range(iters):
        adv.requires_grad = True
        logits = model(adv)
        loss = nn.CrossEntropyLoss()(logits, label)
        loss.backward()

        adv = adv + alpha * adv.grad.sign()
        eta = torch.clamp(adv - ori, min=-eps, max=eps)
        adv = torch.clamp(ori + eta, -3, 3).detach()

    return adv
