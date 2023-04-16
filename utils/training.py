from .utils import DTYPE


def train_reg_epoch(epoch, global_step, train_loader, model, optimizer, opt, writer, device):
    losses = model.fit(train_loader, optimizer, device, dtype=DTYPE)
    metrix_strs = {}
    for i in range(len(losses)):
        metrix_strs['loss%d' % (i+1)] = losses[i].item()
    metrix_strs['loss'] = losses.sum().item()
    l = ', '.join(['loss%d: %.3f' % (i, losses[i]) for i in range(len(losses))])
    print('Epoch: [%d], %s, loss_sum: %.2f' %
          (epoch + 1, l, losses.sum().item()))

def train_clf_epoch(epoch, global_step, train_loader, model, optimizer, opt, writer, device):
    metrix_strs = {}
    if opt.framework == 'pytorch':
        losses = model.fit( train_loader, optimizer, device, dtype=DTYPE)
        metrix_strs['loss'] = losses.sum().item()
        print('Epoch: [%d], loss_sum: %.2f' %
              (epoch + 1, losses.sum().item()))
    else:
        raise NotImplementedError

def train_epoch(epoch, global_step, train_loader, net, optimizer, opt, writer, device):
    if opt.clfsetting == 'regression':
        train_reg_epoch(epoch, global_step, train_loader, net, optimizer, opt, writer, device)
    else:
        train_clf_epoch(epoch, global_step, train_loader, net, optimizer, opt, writer, device)
