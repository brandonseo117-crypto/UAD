from dataset import test_loader
from vae import VAEModel, ELBOvae
from vit import VitAE, VitDualDomainLoss
from architecture import MambaUAD, DualDomainLoss
import torch
import numpy as np
from calflops import calculate_flops
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.utils.flop_counter import FlopCounterMode
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
import time
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

vitae_model = VitAE().to(device)
vitae_state_dict = torch.load('vit_weights.pth', weights_only=True)
vitae_model.load_state_dict(vitae_state_dict)
model = vitae_model.eval()
criterion = VitDualDomainLoss()

batch_latencies = []

ad_loss, ad_psnr, ad_ssim = [], [], []
mci_loss, mci_psnr, mci_ssim = [], [], []
cn_loss, cn_psnr, cn_ssim = [], [], []

flops, macs, params = calculate_flops(model=model, input_shape=(1, 1, 128, 128, 128), print_results=True)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    for idx, (data, label, vis_date, pt_id) in enumerate(test_loader):
        input_data = data.to(device)
        torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        output, hidden_state = model(input_data)
        torch.cuda.synchronize(device) 
        end_time = time.perf_counter()
        batch_latencies.append(end_time - start_time)
        loss = criterion(output, input_data, hidden_state)
        psnr_val = peak_signal_noise_ratio(output, input_data, data_range=6.53).item()
        ssim_val = structural_similarity_index_measure(output, input_data, data_range=6.53).item()
        
        if label.item() == 0:
            ad_loss.append(loss.item())
            ad_psnr.append(psnr_val)
            ad_ssim.append(ssim_val)
        elif label.item() == 1:
            cn_loss.append(loss.item())
            cn_psnr.append(psnr_val)
            cn_ssim.append(ssim_val)
        else:
            mci_loss.append(loss.item())
            mci_psnr.append(psnr_val)
            mci_ssim.append(ssim_val)

peak_vram_bytes = torch.cuda.max_memory_allocated(device)
peak_vram_mb = peak_vram_bytes / (1024 ** 2)
print(f"Peak VRAM Used: {peak_vram_mb:.3f} MB")

mean_batch_latency = sum(batch_latencies) / len(batch_latencies)
overall_throughput = 1 / mean_batch_latency
print(f'Mean batch latency: {mean_batch_latency} and overall throughput: {overall_throughput}')

def get_auc_metrics(control_values, anomaly_values, loss_is=True):
    true = np.array([0] * len(control_values) + [1] * len(anomaly_values))
    if loss_is==True:
        scores = np.array(control_values + anomaly_values)
    else:
        scores = -1 * np.array(control_values + anomaly_values)
    
    roc_auc = roc_auc_score(y_true=true, y_score=scores)
    precision, recall, _ = precision_recall_curve(y_true=true, probas_pred=scores)
    prc_auc = auc(recall, precision)

    print(f'ROC AUC: {roc_auc:.4f}')
    print(f'PRC AUC: {prc_auc:.4f}')

    return roc_auc, prc_auc

loss_roc1, loss_prc1 = get_auc_metrics(cn_loss, ad_loss)
loss_roc2, loss_prc2 = get_auc_metrics(cn_loss, mci_loss)
psnr_roc1, psnr_prc1 = get_auc_metrics(cn_psnr, ad_psnr, loss_is=False)
psnr_roc2, psnr_prc2 = get_auc_metrics(cn_psnr, mci_psnr, loss_is=False)
ssim_roc1, ssim_prc1 = get_auc_metrics(cn_ssim, ad_ssim, loss_is=False)
ssim_roc2, ssim_prc2 = get_auc_metrics(cn_ssim, mci_ssim, loss_is=False)