import torch
import time
import traceback
import polyscope as ps
import os
import numpy as np
# My modules
from data import InvalidPatternDefError

def eval_pcd_metrics(model, criterion, data_warpper, save_to, rank=0, section='test', ):

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
        return _eval_pcd_metrics_per_loader(model, criterion, loader, device, section, data_warpper, save_to)

def _eval_pcd_metrics_per_loader(model, criterion, loader, device, section, data_warpper, save_to):
    current_metrics = dict.fromkeys(['full_loss'], [])
    counter = 0
    loader_iter = iter(loader)
    start_time = time.time()

    while True:
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        except InvalidPatternDefError as e:
            print(e)
            continue
        images, gt = batch["pcd"].to(device), batch["ground_truth"]
        img_name = batch["pcd_fn"]

        outputs = model(images)
        save_to = os.path.join(save_to, section)
        full_loss, loss_dict = criterion(outputs, gt, epoch = -1)
        if "posterior" in outputs.keys():
            del outputs["posterior"]
        if counter == 0:
            _log_batch_pcd(outputs, batch, data_warpper, save_to)
        
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

    return current_metrics


def _log_batch_pcd(outputs, batch_sample, data_warpper, save_to):
    with torch.no_grad():
        try:
            batch_size = 10
            pcd = batch_sample["pcd"][:batch_size]
            gt = {key:val[:batch_size] for key, val in batch_sample["ground_truth"].items()}
            name, folder = batch_sample["name"][:batch_size], batch_sample["data_folder"][:batch_size]

            batch_img_files = data_warpper.dataset.save_recon_batch(
                outputs, 
                name, 
                folder, 
                save_to=save_to,
                pcds=pcd)
            batch_gt_files = data_warpper.dataset.save_gt_batch_imgs(
                    gt, name, folder, save_to=save_to
                )
            for i in range(batch_size):
                np.savetxt(os.path.join(save_to, name[i] + f"_{i}.txt"), batch_sample["pcd"][0].cpu().numpy())
        except BaseException as e:
            print(e)
            traceback.print_exc()
            print('Error::On saving pattern prediction for image logging. Nothing logged')