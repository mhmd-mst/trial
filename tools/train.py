import torch
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import RandomSampler
from semseg.models.ddrnet import DDRNet
from semseg.datasets.ai4mars import ai4mars
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
from val import evaluate


def main(cfg, gpu, save_dir):
    start = time.time()
    best_mIoU = 0.0
    num_workers = mp.cpu_count()
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']

    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train')
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val')
    # data_sampler = RandomSampler(trainset, num_samples=len(valset))
    model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], trainset.n_classes)
    print(
        f"The number of parameters of {model_cfg['NAME']} is: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print("")
    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE']
    # class_weights = trainset.class_weights.to(device)
    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None)
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])

    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, epochs * iters_per_epoch, sched_cfg['POWER'],
                              iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])

    if train_cfg['RESUME_TRAIN']:
        checkpoint = torch.load(model_cfg['PRETRAINED'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
    else:
        model.init_pretrained(model_cfg['PRETRAINED'])
        start_epoch = 0
    model = model.to(device)

    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, pin_memory=True)
    valloader = DataLoader(valset, batch_size=1, num_workers=1, pin_memory=True)
    # trainsubsetloader = DataLoader(trainset, batch_size=1, num_workers=1, pin_memory=True, sampler=data_sampler)

    scaler = GradScaler(enabled=train_cfg['AMP'])
    writer = SummaryWriter(str(save_dir / 'logs'))

    for epoch in range(start_epoch, epochs):

        if epoch == 0:
            results = evaluate(model, valloader, device, loss_fn)
            acc, macc, test_loss, miou = results[0], results[1], results[-2], results[-1]
            print(f"Current mIoU: {miou} Best mIoU: {best_mIoU}")
            print(f"Current accuracy: {acc}")
            print(f"Current maccuracy: {macc}")
            print("")
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('mIoU/test', miou, epoch)
            results = evaluate(model, trainloader, device, loss_fn)
            acc, macc, miou = results[0], results[1], results[-1]
            writer.add_scalar('mIoU/train', miou, epoch)
            print("Evaluating on train")
            print(f"Current mIoU: {miou}")
            print(f"Current accuracy: {acc}")
            print(f"Current maccuracy: {macc}")
            print(f"Current maccuracy: {macc}")
            print("")
            print("="*20)
            print("")

        model.train()

        train_loss = 0.0
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch,
                    desc=f"Epoch: [{epoch + 1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")

        for iter, (img, lbl) in pbar:
            optimizer.zero_grad(set_to_none=True)

            img = img.to(device)
            lbl = lbl.to(device)

            with autocast(enabled=train_cfg['AMP']):
                logits = model(img)
                loss = loss_fn(logits, lbl)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            skip_lr_sched = (scale > scaler.get_scale())
            if not skip_lr_sched:
                scheduler.step()
            torch.cuda.synchronize()

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            train_loss += loss.item()

            pbar.set_description(
                f"Epoch: [{epoch + 1}/{epochs}] Iter: [{iter + 1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter + 1):.8f}")

        train_loss /= iter + 1
        writer.add_scalar('train/loss', train_loss, epoch)
        torch.cuda.empty_cache()

        if (epoch + 1) % train_cfg['EVAL_INTERVAL'] == 0 and epoch != 0:
            results = evaluate(model, valloader, device, loss_fn)
            acc, macc, test_loss, miou = results[0], results[1], results[-2], results[-1]
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('mIoU/test', miou, epoch)

            checkpoint = {
                'loss': train_loss,
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}
            torch.save(checkpoint, save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}.pth")
            if miou > best_mIoU:
                best_mIoU = miou
            print(f"Current mIoU: {miou} Best mIoU: {best_mIoU}")
            print(f"Current accuracy: {acc}")
            print(f"Current maccuracy: {macc}")
            print(f"Current maccuracy: {macc}")
            print("")
            results = evaluate(model, trainloader, device, loss_fn)
            acc, macc, miou = results[0], results[1], results[-1]
            writer.add_scalar('mIoU/train', miou, epoch)
            print("Evaluating on train")
            print(f"Current mIoU: {miou}")
            print(f"Current accuracy: {acc}")
            print(f"Current maccuracy: {macc}")
            print(f"Current maccuracy: {macc}")
            print("")
            print("="*20)
            print("")

    writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ['Best mIoU', f"{best_mIoU:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    print(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/ai4mars.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)
    main(cfg, gpu, save_dir)
    cleanup_ddp()
