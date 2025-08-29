

def train_one_epoch(epoch_index, tb_writer, dataloader, optimizer, model, loss_func):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(dataloader):
        # Every data instance is an input + label pair
        spa, ang, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(spa, ang)

        # Compute the loss and its gradients
        loss = loss_func(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 20 == 19:
            last_loss = running_loss / 20. # loss per batch
            print('batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(dataloader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            tb_writer.add_scalar('train/batch_loss', last_loss, tb_x)
            running_loss = 0.

    return last_loss