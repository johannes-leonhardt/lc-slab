import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy

from data.dataset import match_dense_to_sparse_labels
from model.sparse_lc_classifier import SparseLCClassifier

class PartialCrossEntropy(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, result):

        lc_hat, lc, idx, n_out, _ = result
        lc_hat, lc = match_dense_to_sparse_labels(lc_hat, lc, idx, n_out)
        loss = F.cross_entropy(lc_hat, lc)
        acc = accuracy(lc_hat, torch.argmax(lc, dim=1), task="multiclass", num_classes=n_out, average="micro")

        return loss, acc
    
PartialCrossEntropyPx = PartialCrossEntropy # Alias
    
class DenseCrossEntropy(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, result):

        lc_hat, lc, n_out = result
        loss = F.cross_entropy(lc_hat, lc)
        acc = accuracy(lc_hat, lc, task="multiclass", num_classes=n_out, average="micro")

        return loss, acc

class PCE_ImageLevelLoss(nn.Module):

    def __init__(self, weight):

        super().__init__()
        self.partial_cross_entropy = PartialCrossEntropy()
        self.weight = weight

    def forward(self, result):

        pce_loss, acc = self.partial_cross_entropy(result)
        lc_hat, lc, _, n_out, _ = result
        image_level_loss = F.cross_entropy(torch.mean(lc_hat, dim=(2,3)), F.one_hot(lc, num_classes=n_out).float())
        ovr_loss = pce_loss + self.weight * image_level_loss

        return ovr_loss, acc
    
class PCE_TotalVariation(nn.Module):

    def __init__(self, weight):

        super().__init__()
        self.partial_cross_entropy = PartialCrossEntropy()
        self.weight = weight

    def forward(self, result):

        pce_loss, acc = self.partial_cross_entropy(result)
        lc_hat, _, _, _, _ = result
        vertical_diff = torch.mean((lc_hat[:,:,1:,:] - lc_hat[:,:,:-1,:]) ** 2)
        horizontal_diff = torch.mean((lc_hat[:,:,:,1:] - lc_hat[:,:,:,:-1]) ** 2)
        total_variation_loss = (vertical_diff + horizontal_diff) / 2
        ovr_loss = pce_loss + self.weight * total_variation_loss

        return ovr_loss, acc
    
class PCE_SelfLearning(nn.Module):

    def __init__(self, teacher_model_path, weight):

        super().__init__()
        self.partial_cross_entropy = PartialCrossEntropy()
        self.teacher_model = SparseLCClassifier.load_from_checkpoint(teacher_model_path)
        self.teacher_model.eval()
        self.teacher_cross_entropy = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, result):

        pce_loss, acc = self.partial_cross_entropy(result)
        lc_hat, _, _, _, network_input = result
        with torch.no_grad():
            lc_hat_teacher = F.softmax(self.teacher_model.refinement_network(network_input), dim=1)
        teacher_loss = self.teacher_cross_entropy(lc_hat, lc_hat_teacher)
        ovr_loss = pce_loss + self.weight * teacher_loss

        return ovr_loss, acc