import torch
import torch.nn as nn
import numpy as np

import time
import copy
import logging
import argparse
import wandb
import matplotlib.pyplot as plt

import utils
import utils.eval_utils as eval_utils
from utils.ptmq_recon import ptmq_reconstruction
from utils.fold_bn import search_fold_and_remove_bn, StraightThrough
from utils.model_utils import quant_modules, load_model, set_qmodel_block_wqbit
from utils.eval_utils import DataSaverHook, StopForwardException, parse_config, validate_model

from quant.quant_state import enable_calib_without_quant, enable_quantization, disable_all
from quant.quant_module import QuantizedLayer, QuantizedBlock
from quant.fake_quant import QuantizeBase
from quant.observer import ObserverBase

logger = logging.getLogger('ptmq')

CONFIG_PATH = '/content/ptmq_log_after/config/gpu_config.yaml'
cfg = parse_config(CONFIG_PATH)

def collect_scale_factors(quant_block):

    """
        Collect scale facotrs from a QuantBasicBlock for low, middle, and high bit quantized layers.
    """

    scale_factors={"low":{}, "mid":{}, "high":{}}

    for layer_name, q_layer in [
        ("conv1_relu_low", quant_block.conv1_relu_low),
        ("conv1_relu_mid", quant_block.conv1_relu_mid),
        ("conv1_relu_high", quant_block.conv1_relu_high),
        ("conv2_low", quant_block.conv2_low),
        ("conv2_mid", quant_block.conv2_mid),
        ("conv2_high", quant_block.conv2_high),
    ]:
        
        if hasattr(q_layer, "weight_fake_quant") and hasattr(q_layer.weight_fake_quant, "scale"):
            scale_factors["low"][layer_name] = q_layer.weight_fake_quant.scale.cpu().detach().numpy()
            scale_factors["mid"][layer_name] = q_layer.weight_fake_quant.scale.cpu().detach().numpy()
            scale_factors["high"][layer_name] = q_layer.weight_fake_quant.scale.cpu().detach().numpy()

    return scale_factors

def save_scale_factors(scale_factors, w_bits, a_bit, filename_template ="resnet18_W{}_A{}_scale_factors.txt"):
    """
        Save scale factors to a text file 
    """
    filename = filename_template.format(",".join(map(str, w_bits)),a_bit)
    print(filename, "'s scale facor")
    print(scale_factors)
    with open (filename, "w") as f:
        for bit_type, layers in scale_factors.items():
            f.write(f"Scale factors for {bit_type} bit quant:\n")
            for layer_name, scale in layers.items():
                f.write(f"   Layer {layer_name} : {scale.tolist()}\n")

            f.write("\n")

    print(f"Scale factors saved to {filename}")


def quantize_model(model, config):
    def replace_module(module, config, qoutput=True):
        children = list(iter(module.named_children()))
        ptr, ptr_end = 0, len(children)
        prev_qmodule = None
        
        while (ptr < ptr_end):
            tmp_qoutput = qoutput if ptr == ptr_end-1 else True
            name, child_module = children[ptr][0], children[ptr][1]
            
            if type(child_module) in quant_modules:
                setattr(module, name, quant_modules[type(child_module)](child_module, config, tmp_qoutput))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantizedLayer(child_module, None, config, w_qconfig=config.quant.w_qconfig, qoutput=tmp_qoutput))

                #setattr(module, name, QuantizedLayer(child_module, None, config, qoutput=tmp_qoutput))
                prev_qmodule = getattr(module, name)
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_qmodule is not None:
                    prev_qmodule.activation = child_module
                    setattr(module, name, StraightThrough())
                else:
                    pass
            elif isinstance(child_module, StraightThrough):
                pass
            else:
                replace_module(child_module, config, tmp_qoutput)
            ptr += 1
    
    # we replace all layers to be quantized with quantization-ready layers
    replace_module(model, config, qoutput=False)
    
    for name, module in model.named_modules():
        print(name, type(module))
    
    
    for name, module in model.named_modules():
        if isinstance(module, QuantizedBlock):
            print(name, module.out_mode)
    
    # we store all modules in the quantized model (weight_module or activation_module)


    w_list, a_list = [], []
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase):
            if 'weight' in name:
                w_list.append(module)
            elif 'act' in name:
                a_list.append(module)
    
    print(w_list)
    print(a_list)
    
    # set first and last layer to 8-bit
    w_list[0].set_bit(8)
    w_list[-1].set_bit(8)
    
    # set the last layer's output to 8-bit
    a_list[-1].set_bit(8)
    
    logger.info(f'Finished quantizing model: {str(model)}')
    
    return model


def get_calib_data(train_loader, num_samples):
    calib_data = []
    for batch in train_loader:
        calib_data.append(batch[0])
        if len(calib_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(calib_data, dim=0)[:num_samples]


def main(config_path):
    # get config for applying ptmq
    config = eval_utils.parse_config(config_path)
    eval_utils.set_seed(config.process.seed)
    
    train_loader, val_loader = eval_utils.load_data(**config.data)
    calib_data = get_calib_data(train_loader, config.quant.calibrate).cuda()
    
    model = load_model(config.model) # load original model
    search_fold_and_remove_bn(model) # remove+fold batchnorm layers
    
    # quanitze model if config.quant is defined
    if hasattr(config, 'quant'):
        model = quantize_model(model, config)
        
    model.cuda() # move model to GPU
    model.eval() # set model to evaluation mode
    
    fp_model = copy.deepcopy(model) # save copy of full precision model
    disable_all(fp_model) # disable all quantization
    
    # set names for all ObserverBase modules
    # ObserverBase modules are used to store intermediate values during calibration
    for name, module in model.named_modules():
        if isinstance(module, ObserverBase):
            module.set_name(name)
    
    # calibration
    print("Starting model calibration...")
    with torch.no_grad():
        tik = time.time()
        enable_calib_without_quant(model, quantizer_type='act_fake_quant')
        model(calib_data[:256]).cuda()
        enable_calib_without_quant(model, quantizer_type='weight_fake_quant')
        model(calib_data[:2]).cuda()
        tok = time.time()
        logger.info(f"Calibration of {str(model)} took {tok - tik} seconds")
    print("Completed model calibration")
    
    print("Starting block reconstruction...")
    tik = time.time()
    # Block reconstruction (layer reconstruction for first & last layers)a
    if hasattr(config.quant, 'recon'):
        enable_quantization(model)
        
        def recon_model(module, fp_module):
            for name, child_module in module.named_children():
                if isinstance(child_module, (QuantizedLayer, QuantizedBlock)):
                    logger.info(f"Reconstructing module {str(child_module)}")
                    ptmq_reconstruction(model, fp_model, child_module, name, getattr(fp_module, name), calib_data, config.quant)
                else:
                    recon_model(child_module, getattr(fp_module, name))
        
        recon_model(model, fp_model)
    tok = time.time()
    print("Completed block reconstruction")
    print(f"PTMQ block reconstruction took {tok - tik:.2f} seconds")
    
    w_qmodes = ["low", "med", "high"]
    a_qbit = config.quant.a_qconfig_med.bit,
    w_qbits = [config.quant.w_qconfig_low, 
               config.quant.w_qconfig_med, 
               config.quant.w_qconfig_high, 
               ]
    enable_quantization(model) # reconsturction 모델을 activate 하기

    for w_qmode, w_qbit in zip(w_qmodes, w_qbits):
        set_qmodel_block_wqbit(model,w_qmode)

        scale_factors = collect_scale_factors(model)

        save_scale_factors(scale_factors, w_qbit, a_qbit)

        for name, module in model.named_modules():
            if isinstance(module, QuantizedBlock):
                print(name, module.out_mode)
        
        print(f"Starting model evaluation of W{w_qbit}A{a_qbit} block reconsutciotn ({w_qmode}....)")
        acc1, acc5 = eval_utils.validate_model(val_loader, model)

        print(f"Top-1 accuracy: {acc1:.2f}, Top-5 accuracy: {acc5:.2f}")
    # validate_model(val_loader, model) # validation 데이터 셋 이용해서 성능 평가하기 

    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='configuration',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default='/content/ptmq_log_after/config/gpu_config.yaml', type=str, help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)
