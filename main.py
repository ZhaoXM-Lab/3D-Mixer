import logging

logging.basicConfig(level='WARNING')
import os

os.environ['MKL_THREADING_LAYER'] = 'GNU'
import torch
import socket
from datetime import datetime
import torch.optim as optim
from utils.tools import ModifiedReduceLROnPlateau
from utils.utils import mf_save_model, mf_load_model
from utils.utils import DTYPE
from utils.datasets import get_dataset
from utils.training import train_epoch
from utils.validation import validate
from utils.utils import count_params
from models.resnet import resnet18
from models import LDMIL_BN, LDMIL, clfmixer, regmixer, fusemixer, DAMIDL, ViT
from utils.opts import parse_opts, mod_opt
from utils.tools import Save_Checkpoint

method_map = {'Res18': resnet18, 'LDMIL': LDMIL, 'LDMIL_BN': LDMIL_BN, 'DAMIDL': DAMIDL,
              'ViT': ViT, 'ClfMixer': clfmixer, 'RegMixer': regmixer, 'FuseMixer': fusemixer}

if __name__ == "__main__":
    opt = parse_opts()
    logging.info("%s" % opt)
    if opt.no_cuda:
        device = 'cpu'
    else:
        device = torch.device("cuda:%d" % opt.cuda_index if torch.cuda.is_available() else "cpu")

    opt.framework = 'pytorch'
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    opt.num_classes = 1
    opt.aux = True if 'Aux' in opt.method else False
    opt.mask_mat = None
    opt.resample_patch = None

    if opt.save_path is None:
        opt.save_path = os.path.join(
            'saved_model', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname() + opt.method)
        opt.save_path += '.pth' if opt.framework == 'pytorch' else '.h5'

    # Specify the hyper-parameters for each method
    opt = mod_opt(opt.method, opt)
    opt.method_setting.update(eval(opt.method_para))

    print('Dataset: ', opt.dataset)
    train_loader, val_loader, test_loader = get_dataset(dataset=opt.dataset,
                                                        opt=opt, resample_patch=opt.resample_patch)

    if opt.val_as_train:
        train_loader = val_loader
    if opt.val_as_test:
        test_loader = val_loader

    Net = method_map[opt.method]
    net = Net(**opt.method_setting)
    count_params(net, framework=opt.framework)

    save_checkpoint = Save_Checkpoint(save_func=mf_save_model, framework=opt.framework, verbose=True,
                                      path=opt.save_path, trace_func=print, mode='min')
    if opt.pretrain_path:
        net = mf_load_model(net, opt.pretrain_path, framework=opt.framework, device=device)

    if opt.framework == 'pytorch':
        net.to(device=device, dtype=DTYPE)
        optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate)
    else:
        raise NotImplementedError
    scheduler = ModifiedReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience, factor=opt.lr_mul_factor,
                                          verbose=True, framework=opt.framework)

    for epoch in range(opt.n_epochs):
        if not (opt.no_train and opt.no_val):
            global_step = len(train_loader.dataset) * epoch
        if not opt.no_train:
            train_epoch(epoch, global_step, train_loader, net, optimizer, opt, None, device)
        if not opt.no_val:
            metrix_figs, metrix_strs, val_metrix = validate(
                global_step, val_loader, net, opt, device, None)
            print(metrix_strs)
        else:
            val_metrix = epoch

        if opt.no_train:
            break

        scheduler.step(-val_metrix)
        save_checkpoint(-val_metrix, net)

    if not opt.no_train:
        net = mf_load_model(model=net, path=opt.save_path, framework=opt.framework, device=device)

    metrix_figs, metrix_strs, val_metrix = validate(None, test_loader, net, opt, device, writer=None)

    for fig in metrix_figs.values():
        fig.show()
    print('Testing Result: \r\n %s' % metrix_strs)
    output_str = {'Method': opt.method, 'Dataset': opt.dataset, 'test_result': '%s' % metrix_strs,
                  'opt': '%s' % opt, }
    print(output_str)
