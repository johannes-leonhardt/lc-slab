
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
import lightning.pytorch as pl
from tqdm import tqdm

from data.dataset import remap_esawc

class SparseLCClassifier(pl.LightningModule):

    def __init__(self, input_constructor, network, loss_function, postprocessing_function):

        super().__init__()

        self.input_constructor = input_constructor
        self.network = network
        self.loss_function = loss_function
        self.postprocessing_function = postprocessing_function
        self.save_hyperparameters()
    
    def training_step(self, batch, batch_idx):

        _, lc, idx, _, _, _, _ = batch
        network_input = self.input_constructor(batch)
        lc_hat, _ = self.network(network_input)
        lc_hat = self.postprocessing_function([lc_hat, network_input])
        loss, acc = self.loss_function([lc_hat, lc, idx, self.network.n_out, network_input])
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, batch_size=idx.shape[0])
        self.log("train_acc", acc, prog_bar=True, sync_dist=True, batch_size=idx.shape[0])

        return loss
    
    def validation_step(self, batch, batch_idx):

        _, lc, idx, _, _, _, _ = batch
        network_input = self.input_constructor(batch)
        lc_hat, _ = self.network(network_input)
        lc_hat = self.postprocessing_function([lc_hat, network_input])
        loss, acc = self.loss_function([lc_hat, lc, idx, self.network.n_out, network_input])
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=idx.shape[0])
        self.log("val_acc", acc, prog_bar=True, sync_dist=True, batch_size=idx.shape[0])
    
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, mode='max', patience=1)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_acc"}
    
    @torch.inference_mode()
    def apply_to_ds(self, ds, batch_size=128):

        self.eval()
        dl = torch.utils.data.DataLoader(ds, shuffle=False, batch_size=batch_size)
        lc_hat_all, lc_all, idx_all = [], [], []
        for batch in tqdm(dl, desc=f"Testing {ds.country}, {ds.region}..."):
            _, lc, idx, _, _, _, _ = batch
            batch[0], batch[3] = batch[0].to(self.device), batch[3].to(self.device) # Img and seg are the only tensors needed for inference
            network_input = self.input_constructor(batch)
            lc_hat, _ = self.network(network_input)
            lc_hat = self.postprocessing_function([lc_hat, network_input]).detach().cpu()
            lc_hat_all.append(lc_hat)
            lc_all.append(lc)
            idx_all.append(idx)
        lc_hat_all, lc_all, idx_all = torch.cat(lc_hat_all), torch.cat(lc_all), torch.cat(idx_all)

        return lc_hat_all, lc_all, idx_all
    
    @torch.inference_mode()
    def apply_to_single_sample(self, ds, idx):

        self.eval()
        ds = Subset(ds, [idx])
        dl = torch.utils.data.DataLoader(ds, shuffle=False, batch_size=1)
        di = iter(dl)
        batch = next(di)
        batch[0], batch[3] = batch[0].to(self.device), batch[3].to(self.device) # Img and seg are the only tensors needed for inference
        network_input = self.input_constructor(batch)
        lc_hat, _ = self.network(network_input)
        lc_hat = self.postprocessing_function([lc_hat, network_input]).detach().cpu()

        return lc_hat.squeeze()#, network_input.feats.squeeze()
    
class DenseLCClassifier(pl.LightningModule):

    def __init__(self, input_constructor, network, loss_function, postprocessing_function):

        super().__init__()

        self.input_constructor = input_constructor
        self.network = network
        self.loss_function = loss_function
        self.postprocessing_function = postprocessing_function
        self.save_hyperparameters()
    
    def training_step(self, batch, batch_idx):

        _, _, _, _, _, lc, _ = batch
        lc = remap_esawc(lc).squeeze()
        network_input = self.input_constructor(batch)
        lc_hat, _ = self.network(network_input)
        lc_hat = self.postprocessing_function([lc_hat, network_input])
        loss, acc = self.loss_function([lc_hat, lc, self.network.n_out])
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, batch_size=lc.shape[0])
        self.log("train_acc", acc, prog_bar=True, sync_dist=True, batch_size=lc.shape[0])

        return loss
    
    def validation_step(self, batch, batch_idx):

        _, _, _, _, _, lc, _ = batch
        lc = remap_esawc(lc).squeeze()
        network_input = self.input_constructor(batch)
        lc_hat, _ = self.network(network_input)
        lc_hat = self.postprocessing_function([lc_hat, network_input])
        loss, acc = self.loss_function([lc_hat, lc, self.network.n_out])
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=lc.shape[0])
        self.log("val_acc", acc, prog_bar=True, sync_dist=True, batch_size=lc.shape[0])
    
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, mode='min', patience=1)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}