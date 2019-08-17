import torch


def init_weight(m, init_type='normal', init_gain=0.02):
    classname = m.__class__.__name__
    if classname in ('Conv2d', 'ConvTranspose2d', 'Linear'):
        if init_type == 'normal':
            torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == 'kaiming':
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname == 'BatchNorm2d':
        if m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
