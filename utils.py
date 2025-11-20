import importlib
import torch


def get_class(class_path):
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def train_one_epoch(epoch_index, tb_writer, dataloader, optimizer, model,
                    loss_func, device, seq_len=None):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(dataloader):
        # Every data instance is an input + label pair
        spa, ang, labels = data

        if seq_len:
            idx = torch.randperm(spa.shape[1])[:seq_len]
            idx, _ = idx.sort()
            spa = spa[:, idx].to(device)
            ang = ang[:, idx].to(device)
        else:
            spa = spa.to(device)
            ang = ang.to(device)

        labels = labels.to(device).float()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(spa, ang)

        # Compute the loss and its gradients
        loss = loss_func(outputs, labels.reshape(-1, 1))
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 50 == 49:
            last_loss = running_loss / 50 # loss per batch
            print('batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(dataloader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            tb_writer.add_scalar('train/batch_loss', last_loss, tb_x)
            running_loss = 0.

        # last_loss = running_loss # loss per batch
        # print('batch {} loss: {}'.format(i + 1, last_loss))
        # tb_x = epoch_index * len(dataloader) + i + 1
        # # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        # tb_writer.add_scalar('train/batch_loss', last_loss, tb_x)
        # running_loss = 0.

    return last_loss