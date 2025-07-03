import torch
from tqdm import tqdm

def fit_one_epoch(net, softmaxloss, epoch, epoch_size, epoch_size_val, gen, gen_test, Epoch, cuda, optimizer, test_dataset_len):
    total_loss = 0

    net.train()
    print('\nStart train')
    with tqdm(total=epoch_size, desc=f'Epoch {epoch+1}/{Epoch}', mininterval=0.3) as pbar:
        for iteration, (images, targets) in enumerate(gen):
            if cuda:
                images = images.cuda()
                targets = targets.cuda()

            optimizer.zero_grad()
            outputs = net(images)
            loss = softmaxloss(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(**{'loss': total_loss / (iteration + 1), 'lr': optimizer.param_groups[0]['lr']})
            pbar.update(1)

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()

    net.eval()
    print('\nStart test')
    test_correct = 0
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch+1}/{Epoch}', mininterval=0.3) as pbar:
        for iteration, (images, targets) in enumerate(gen_test):
            if cuda:
                images = images.cuda()
                targets = targets.cuda()

            outputs = net(images)
            _, preds = torch.max(outputs.data, 1)
            test_correct += torch.sum(preds == targets.data).item()
            pbar.set_postfix(**{'test AP': 100.0 * test_correct / test_dataset_len})
            pbar.update(1)

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()
