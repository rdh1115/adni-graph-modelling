# --------------------------------------------------------
# References:
# mae_st: https://github.com/facebookresearch/mae_st
# --------------------------------------------------------

import torch
import utils.misc as misc
from utils.meters import ClassTestMeter, PredTestMeter
from timm.utils import accuracy


@torch.no_grad()
def test(data_loader, model, device, args, fp32=False):
    task = args.task

    if task == 'class':
        softmax = torch.nn.Softmax(dim=1).cuda()
        metric_logger = ClassTestMeter(
            delimiter="  ",
            ensemble_method='max' if args.max_pooling else 'sum'
        )
    elif task == 'pred':
        criterion = torch.nn.MSELoss()
        metric_logger = PredTestMeter(delimiter="  ")
    else:
        raise NotImplementedError()

    header = "Test:"

    # switch to evaluation atlas
    model.eval()

    for batched_data in metric_logger.log_every(data_loader, 10, header):
        batched_data = misc.prepare_batch(batched_data, device=device)
        scaler = None if 'scaler' not in batched_data else batched_data['scaler']

        samples, targets, target_shape = misc.get_samples_targets(batched_data, task, device, args)

        # compute output
        with torch.cuda.amp.autocast(enabled=False):
            preds = model(batched_data)
            if task == 'pred':
                if scaler:
                    # print(f'before norm: {preds}')
                    outputs = scaler.inverse_transform(preds, args)
                    # print('after norm: {outputs}')
                    targets = scaler.inverse_transform(targets, args)
                else:
                    outputs = preds
            elif task == 'class':
                outputs = softmax(preds)

        batch_size = samples.shape[0]
        if task == 'class':
            subject_ids = [int(features['rids'].detach()) for features in batched_data['add_features']]
            metric_logger.store_predictions(outputs.detach(), targets.detach(), subject_ids)

            k = 5 if args.pred_num_classes > 3 else 2
            acc1, acck = accuracy(outputs, targets, topk=(1, k))

            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters[f"acc{k}"].update(acck.item(), n=batch_size)
        elif task == 'pred':
            metric_logger.store_predictions(outputs.detach(), targets.detach())

            metrics = misc.forecasting_acc(
                outputs,
                targets,
                target_shape=None if args.n_hist == args.n_pred or args.n_pred > args.n_hist
                else target_shape
            )

            metric_logger.meters['mae'].update(metrics['MAE'].item(), n=batch_size)
            metric_logger.meters['rmse'].update(metrics['RMSE'].item(), n=batch_size)
            metric_logger.meters['mape'].update(metrics['MAPE'].item(), n=batch_size)

    if task == 'class':
        metrics = metric_logger.finalize_metrics(args=args)
        print(
            f"*****************************************************************************\n"
            f"Top-1 accuracy of the network on {len(data_loader)} points: {metrics['acc1']}\n"
            f"Top-2 accuracy of the network on {len(data_loader)} points: {metrics['acc2']}\n"
            f"Precision of the network on {len(data_loader)} points: {metrics['precision']}\n"
            f"Recall of the network on {len(data_loader)} points: {metrics['recall']}\n"
            f"ROC-AUC of the network on {len(data_loader)} points: {metrics['roc_auc']}\n"
        )
    elif task == 'pred':
        metrics = metric_logger.finalize_metrics(
            target_shape=None if args.n_hist == args.n_pred or args.n_pred > args.n_hist
            else target_shape
        )
        print(
            f"*****************************************************************************\n"
            f"MAE of the network on {len(data_loader)} points: {metrics['MAE']:.4f}\n"
            f"MAPE of the network on {len(data_loader)} points: {metrics['MAPE']:.7f}\n"
            f"RMSE of the network on {len(data_loader)} points: {metrics['RMSE']:.4f}\n"
        )
    else:
        raise NotImplementedError()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
