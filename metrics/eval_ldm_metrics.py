import torch
import time
import traceback
import polyscope as ps
import os
import numpy as np
# My modules
from data import InvalidPatternDefError
from .eval_pcd_metrics import _log_batch_pcd
import torchvision.transforms as T

def eval_ldm_metrics(model, criterion, data_warpper, save_to, rank=0, section='test', ):

    device = 'cuda:{}'.format(rank) if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    criterion.to(device)
    criterion.eval()
    criterion.with_quality_eval()
    data_warpper.dataset.set_training(False)

    with torch.no_grad():
        loader = data_warpper.get_loader(data_section=section)
        print("eval on {}, len = {}".format(section, len(loader)))
        return _eval_ldm_metrics_per_loader(model, criterion, loader, device, section, data_warpper, save_to)

def _eval_ldm_metrics_per_loader(model, criterion, loader, device, section, data_warpper, save_to):
    current_metrics = dict.fromkeys(['full_loss'], [])
    recon_current_metrics = dict.fromkeys(['full_loss'], [])
    counter = 0
    loader_iter = iter(loader)
    start_time = time.time()

    collect_keys = ["st_f1s"]
    while True:
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        except InvalidPatternDefError as e:
            print(e)
            continue
        pcd, gt, images = batch["pcd"].to(device), batch["ground_truth"], batch["image"]

        cond = images.to(device)
        pcd = pcd.to(device)
        recon_output = model.module.reconstruct_first_stage(pcd, return_stitches=False)
        output = model.module.sample(cond, device, sample_times=1, guidance_scale=7.5, return_intermediates=False, return_stitches=False)[0]
        save_to = os.path.join(save_to, section)
        
        full_loss, loss_dict = criterion(output, gt, epoch = -1)
        recon_full_loss, recon_loss_dict = criterion(recon_output, gt, epoch = -1)
        if "posterior" in output.keys():
            del output["posterior"]
        if "posterior" in recon_output.keys():
            del recon_output["posterior"]
        if counter == 0:
            _log_batch_ldm_output(output, batch, data_warpper, save_to)
            _log_batch_pcd(recon_output, batch, data_warpper, save_to)
        
        # gathering up
        current_metrics['full_loss'].append(full_loss.cpu().numpy())
        for key, value in loss_dict.items():
            if key not in current_metrics:
                current_metrics[key] = []  # init new metric
            if value is not None:  # otherwise skip this one from accounting for!
                value = value.cpu().numpy() if isinstance(value, torch.Tensor) else value
                if isinstance(value, list):
                    current_metrics[key].extend(value)
                else:
                    current_metrics[key].append(value)
                    
        # gathering up
        recon_current_metrics['full_loss'].append(recon_full_loss.cpu().numpy())
        for key, value in loss_dict.items():
            if key not in recon_current_metrics:
                recon_current_metrics[key] = []  # init new metric
            if value is not None:  # otherwise skip this one from accounting for!
                value = value.cpu().numpy() if isinstance(value, torch.Tensor) else value
                if isinstance(value, list):
                    recon_current_metrics[key].extend(value)
                else:
                    recon_current_metrics[key].append(value)
                    
        counter += 1
        if counter % 10 == 0:
            print("eval progress: {}, time cost={:.2f}".format(counter, time.time() - start_time))
    print(f"Total eval batches: {counter}, time cost={time.time() - start_time:.2f}")

    # sum & normalize 
    for metric in current_metrics:
        if len(current_metrics[metric]):
            current_metrics[metric] = sum(current_metrics[metric]) / len(current_metrics[metric])
        else:
            current_metrics[metric] = None
            
    # sum & normalize 
    for metric in recon_current_metrics:
        if len(recon_current_metrics[metric]):
            recon_current_metrics[metric] = sum(recon_current_metrics[metric]) / len(recon_current_metrics[metric])
        else:
            recon_current_metrics[metric] = None

    return current_metrics, recon_current_metrics


def _log_batch_ldm_output(outputs, batch_sample, data_warpper, save_to):
    with torch.no_grad():
        try:
            batch_size = 10
            image = batch_sample["image"][:batch_size]
            name, folder = batch_sample["name"][:batch_size], batch_sample["data_folder"][:batch_size]

            batch_img_files = data_warpper.dataset.save_prediction_batch(
                outputs, 
                name, 
                folder, 
                save_to=save_to,
                image=image)
            mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
            std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
            for i in range(batch_size):
                T.ToPILImage()(image * std + mean).save(os.path.join(save_to, f"input_image{i}.png")) 
                
        except BaseException as e:
            print(e)
            traceback.print_exc()
            print('Error::On saving pattern prediction for image logging. Nothing logged')
        
