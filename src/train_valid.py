from src.utils.train import AvgMeter, get_lr
from tqdm import tqdm

img_id = 6
capt_id = 7

def train_epoch(config, model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter(name='Train loss')
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        # image representation
        batch[img_id] = batch[img_id].to(config["device"])
        # caption representation
        batch[capt_id] = batch[capt_id].to(config["device"])
        z_i, z_t = model(batch)
        loss = model.symmetric_contrastive_loss(z_i, z_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        # get image representation
        count = batch[img_id].size(0)
        loss_meter.update(val=loss.item(), count=count)
        # loss_meter.append_loss_value(val=loss.item())

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(config, model, valid_loader):
    loss_meter = AvgMeter(name='Validation loss')

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch[img_id] = batch[img_id].to(config["device"])
        batch[capt_id] = batch[capt_id].to(config["device"])
        z_i, z_t = model(batch)
        loss = model.symmetric_contrastive_loss(z_i, z_t)

        # get image representation
        count = batch[img_id].size(0)
        loss_meter.update(val=loss.item(), count=count)
        # loss_meter.append_loss_value(val=loss.item())
        # print(f"Loss.item(): {loss.item()}, count: {count}")

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter