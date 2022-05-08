from .network_g import UnetGenerator
from .util import get_norm_layer, init_net
def define_TFM(input_nc=9, output_nc=4, num_downs=6, ngf=64, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    net = UnetGenerator(input_nc, output_nc, num_downs, ngf, norm_layer, use_dropout)

    return init_net(net, init_type, init_gain, gpu_ids)
