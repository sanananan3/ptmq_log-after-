import torch
import torchvision
import torch.nn as nn
import gc

from models.resnet import BasicBlock, Bottleneck, resnet18, resnet50
from quant.quant_module import QuantizedLayer, QuantizedBlock, Quantizer
from models.mobilenetv2 import InvertedResidual, mobilenetv2 
#from models.regnet import Bottleneck as RegNetBottleneck
# 1. QuantBasicBlock -> resnet18 

class QuantBasicBlock(QuantizedBlock): 
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    qoutput:    whether to quantize the block output
    out_mode:   setting inference block output mode
        - "mixed":  mixed feature output. used only for block reconstruction
        - "low":    low bit feature output
        - "med":    medium bit feature output
        - "high":   high bit feature output
    """
    def __init__(self, orig_module: BasicBlock, config, qoutput=True, out_mode="calib"):
        super().__init__()
        self.out_mode = out_mode
        self.qoutput = qoutput


        # weight 에 대한 quantizer 
       
        self.conv1_relu_low = QuantizedLayer(orig_module.conv1, orig_module.relu1, config, w_qconfig=config.quant.w_qconfig_low, qoutput=False)
        self.conv1_relu_mid = QuantizedLayer(orig_module.conv1, orig_module.relu1, config, w_qconfig=config.quant.w_qconfig_med, qoutput=False)
        self.conv1_relu_high = QuantizedLayer(orig_module.conv1, orig_module.relu1, config, w_qconfig=config.quant.w_qconfig_high, qoutput=False)

        self.conv2_low = QuantizedLayer(orig_module.conv2, None, config,  w_qconfig=config.quant.w_qconfig_low, qoutput=False)
        self.conv2_mid = QuantizedLayer(orig_module.conv2, None, config, w_qconfig=config.quant.w_qconfig_med, qoutput=False)
        self.conv2_high = QuantizedLayer(orig_module.conv2, None, config, w_qconfig=config.quant.w_qconfig_high, qoutput=False)
        
        if orig_module.downsample is None:
            self.downsample_low = None
            self.downsample_mid = None
            self.downsample_high = None
        else:
            self.downsample_low = QuantizedLayer(orig_module.downsample[0], None, config, w_qconfig=config.quant.w_qconfig_low, qoutput=False)
            self.downsample_mid= QuantizedLayer(orig_module.downsample[0], None, config, w_qconfig=config.quant.w_qconfig_med, qoutput=False)
            self.downsample_high= QuantizedLayer(orig_module.downsample[0], None, config, w_qconfig=config.quant.w_qconfig_high, qoutput=False)

        self.activation = orig_module.relu2

        if self.qoutput:
            # self.block_post_act_fake_quantize_low = Quantizer(None, config.quant.a_qconfig_low)
            self.block_post_act_fake_quantize_med = Quantizer(None, config.quant.a_qconfig_med)
            # self.block_post_act_fake_quantize_high = Quantizer(None, config.quant.a_qconfig_high)
            
            self.f_l = None
            self.f_m = None
            self.f_h = None
            self.f_lmh = None
            
            self.lambda1 = config.quant.ptmq.lambda1
            self.lambda2 = config.quant.ptmq.lambda2
            self.lambda3 = config.quant.ptmq.lambda3
            self.mixed_p = config.quant.ptmq.mixed_p
            
    def forward(self, x):
        residual_low = x if self.downsample_low is None else self.downsample_low(x)
        residual_mid = x if self.downsample_mid is None else self.downsample_mid(x)
        residual_high = x if self.downsample_high is None else self.downsample_high(x)

        # Conv1 -> low, mid, high bit-width로 양자화된 출력 계산 
        
        out_low = self.conv1_relu_low(x)
        out_mid = self.conv1_relu_mid(x)
        out_high = self.conv1_relu_high(x)

        # Conv2 -> low, mid, hight 비트로 양자화된 출력 계산 
        out_low = self.conv2_low(out_low)
        out_mid = self.conv2_mid(out_mid)
        out_high = self.conv2_high(out_high)

        out_low += residual_low
        out_mid += residual_mid
        out_high += residual_high

          # 활성화 함수 적용
        out_low = self.activation(out_low)
        out_mid = self.activation(out_mid)
        out_high = self.activation(out_high)


        if self.qoutput:
            if self.out_mode == "calib":
   
                self.f_l = self.block_post_act_fake_quantize_med(out_low)
                self.f_m = self.block_post_act_fake_quantize_med(out_mid)
                self.f_h = self.block_post_act_fake_quantize_med(out_high)
                
                self.f_lmh = self.lambda1 * self.f_l + self.lambda2 * self.f_m + self.lambda3 * self.f_h
                f_mixed = torch.where(torch.rand_like(out_mid) < self.mixed_p, out_mid, self.f_lmh)
                
                out = f_mixed
                
            elif self.out_mode == "low":
               # out = self.block_post_act_fake_quantize_low(out)
               out = self.block_post_act_fake_quantize_med(out_low)
            elif self.out_mode == "med":
               # out = self.block_post_act_fake_quantize_med(out)
               out= self.block_post_act_fake_quantize_med(out_mid)
            elif self.out_mode == "high":
               # out = self.block_post_act_fake_quantize_high(out)
               out = self.block_post_act_fake_quantize_med(out_high)
            else:
                raise ValueError(f"Invalid out_mode '{self.out_mode}': only ['low', 'med', 'high'] are supported")
        return out

# 2. QuantBottleneck -> resnet-50 

class QuantBottleneck(QuantizedBlock):
    """
    Implementation of Quantized Bottleneck Block used in ResNet-50, Resnet-101, and ResNet-152.
    """
    # weight multi bit 은 차후에
    def __init__(self, orig_module, config, qoutput=True, out_mode="calib"):
        super().__init__()
        self.out_mode = out_mode

        self.qoutput = qoutput

        self.conv1_relu_low = QuantizedLayer(orig_module.conv1, orig_module.relu1, config, w_qconfig=config.quant.w_qconfig_low, qoutput=False)
        self.conv1_relu_mid = QuantizedLayer(orig_module.conv1, orig_module.relu1, config, w_qconfig=config.quant.w_qconfig_med, qoutput=False)
        self.conv1_relu_high = QuantizedLayer(orig_module.conv1, orig_module.relu1, config, w_qconfig=config.quant.w_qconfig_high, qoutput=False)

        self.conv2_relu_low = QuantizedLayer(orig_module.conv2, orig_module.relu2, config, w_qconfig= config.quant.w_qconfig_low, qoutput=False)
        self.conv2_relu_mid = QuantizedLayer(orig_module.conv2, orig_module.relu2, config, w_qconfig= config.quant.w_qconfig_med, qoutput=False)
        self.conv2_relu_high = QuantizedLayer(orig_module.conv2, orig_module.relu2, config, w_qconfig= config.quant.w_qconfig_high, qoutput=False)

        self.conv3_low = QuantizedLayer(orig_module.conv3, None, config, w_qconfig=config.quant.w_qconfig_low, qoutput=False)
        self.conv3_mid = QuantizedLayer(orig_module.conv3, None, config, w_qconfig=config.quant.w_qconfig_med, qoutput=False)
        self.conv3_high = QuantizedLayer(orig_module.conv3, None, config, w_qconfig = config.quant.w_qconfig_high, qoutput=False)

        
        if orig_module.downsample is None:
            self.downsample_low = None
            self.downsample_mid = None
            self.downsample_high = None
 
        else:
            self.downsample_low = QuantizedLayer(orig_module.downsample[0], None, config, config.quant.w_qconfig_low, qoutput=False)
            self.downsample_mid = QuantizedLayer(orig_module.downsample[0], None, config, config.quant.w_qconfig_med, qoutput=False)
            self.downsample_high = QuantizedLayer(orig_module.downsample[0], None, config, config.quant.w_qconfig_high, qoutput=False)

        self.activation = orig_module.relu3
        if self.qoutput:
            #self.block_post_act_fake_quantize = Quantizer(None, config.quant.a_qconfig)
            # self.block_post_act_fake_quantize_low = Quantizer(None, config.quant.a_qconfig_low)
            self.block_post_act_fake_quantize_med = Quantizer(None, config.quant.a_qconfig_med)
            # self.block_post_act_fake_quantize_high = Quantizer(None, config.quant.a_qconfig_high)
            self.f_l = None 
            self.f_m = None 
            self.f_h = None 
            self.f_lmh = None 

            self.lambda1 = config.quant.ptmq.lambda1
            self.lambda2 = config.quant.ptmq.lambda2
            self.lambda3 = config.quant.ptmq.lambda3

            self.mixed_p = config.quant.ptmq.mixed_p 



    def forward(self, x):
        residual_low = x if self.downsample_low is None else self.downsample_low(x)
        residual_mid = x if self.downsample_mid is None else self.downsample_mid(x)
        residual_high = x if self.downsample_high is None else self.downsample_high(x)


        out_low = self.conv1_relu_low(x)
        out_mid = self.conv1_relu_mid(x)
        out_high = self.conv1_relu_high(x)

        out_low = self.conv2_relu_low(out_low)
        out_mid = self.conv2_relu_mid(out_mid)
        out_high = self.conv2_relu_high(out_high)

        out_low = self.conv3_low(out_low)
        out_mid = self.conv3_mid(out_mid)
        out_high = self.conv3_high(out_high)


        out_low += residual_low
        out_mid += residual_mid
        out_high += residual_high

        out_low = self.activation(out_low)
        out_mid = self.activation(out_mid)
        out_high = self.activation(out_high)


        if self.qoutput:
            if self.out_mode == "calib":
                self.f_l = self.block_post_act_fake_quantize_med(out_low)
                self.f_m = self.block_post_act_fake_quantize_med(out_mid)
                self.f_h = self.block_post_act_fake_quantize_med(out_high)

                self.f_lmh = self.lambda1*self.f_l + self.lambda2*self.f_m + self.lambda3*self.f_h

                f_mixed = torch.where(torch.rand_like(out_mid) < self.mixed_p , out_mid, self.f_lmh)

                x = f_mixed 
            elif self.out_mode == "low":
                x = self.block_post_act_fake_quantize_med(out_low)
            elif self.out_mode == "med":
                x = self.block_post_act_fake_quantize_med(out_mid)
            elif self.out_mode == "high":
                x = self.block_post_act_fake_quantize_med(out_high)

            else :
                raise ValueError(
                    f"Invalid out_mode '{self.out_mode}: only ['low','med','high'] are supported."
                )
            
        return x 
    

# 3. QuantRegNetBottleneck -> regnetx-600mf /// 일단 regnet 보류 ... 


# class QuantRegNetBottleneck(QuantizedBlock):
#     """
#     Implementation of Quantized Bottleneck Block used in RegNet (X, Y) models.
#     """

#     def __init__(
#         self, orig_module: RegNetBottleneck, config, qoutput=True, out_mode="calib"
#     ):
#         super().__init__()
#         self.out_mode = out_mode
#         self.qoutput = qoutput

#         # Copy over attributes from the original module
#         self.conv1 = orig_module.conv1
#         self.conv1.conv = QuantizedLayer(self.conv1.conv, orig_module.act3, config)
#         self.conv2 = orig_module.conv2
#         self.conv2.conv = QuantizedLayer(self.conv2.conv, orig_module.act3, config)
#         self.conv3 = orig_module.conv3
#         self.conv3.conv = QuantizedLayer(self.conv3.conv, None, config, qoutput=False)
#         self.se = orig_module.se
#         self.downsample = orig_module.downsample
#         self.drop_path = orig_module.drop_path
#         self.act3 = orig_module.act3

#         # Handle downsample layer
#         if self.downsample is not None:
#             if hasattr(self.downsample, "conv"):
#                 self.downsample.conv = QuantizedLayer(
#                     self.downsample.conv, None, config, qoutput=False
#                 )

#         # The rest of your quantization code remains the same
#         if self.qoutput:
#             self.block_post_act_fake_quantize_low = Quantizer(
#                 None, config.quant.a_qconfig_low
#             )
#             self.block_post_act_fake_quantize_med = Quantizer(
#                 None, config.quant.a_qconfig_med
#             )
#             self.block_post_act_fake_quantize_high = Quantizer(
#                 None, config.quant.a_qconfig_high
#             )

#             self.f_l = None
#             self.f_m = None
#             self.f_h = None
#             self.f_lmh = None

#             self.lambda1 = config.quant.ptmq.lambda1
#             self.lambda2 = config.quant.ptmq.lambda2
#             self.lambda3 = config.quant.ptmq.lambda3

#             self.mixed_p = config.quant.ptmq.mixed_p

#     def forward(self, x):
#         shortcut = x
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.se(x)
#         x = self.conv3(x)
#         if self.downsample is not None:
#             x = self.drop_path(x) + self.downsample(shortcut)
#         x = self.act3(x)

#         if self.qoutput:
#             if self.out_mode == "calib":
#                 self.f_l = self.block_post_act_fake_quantize_low(x)
#                 self.f_m = self.block_post_act_fake_quantize_med(x)
#                 self.f_h = self.block_post_act_fake_quantize_high(x)

#                 self.f_lmh = (
#                     self.lambda1 * self.f_l
#                     + self.lambda2 * self.f_m
#                     + self.lambda3 * self.f_h
#                 )
#                 f_mixed = torch.where(torch.rand_like(x) < self.mixed_p, x, self.f_lmh)

#                 x = f_mixed
#             elif self.out_mode == "low":
#                 x = self.block_post_act_fake_quantize_low(x)
#             elif self.out_mode == "med":
#                 x = self.block_post_act_fake_quantize_med(x)
#             elif self.out_mode == "high":
#                 x = self.block_post_act_fake_quantize_high(x)
#             else:
#                 raise ValueError(
#                     f"Invalid out_mode '{self.out_mode}': only ['low', 'med', 'high'] are supported"
#                 )
#         return x



# 4. QuantInvertedResidual -> mobilenetv2

class QuantInvertedResidual(QuantizedBlock):
    def __init__ (
            self, orig_module: InvertedResidual, config, qoutput = True, out_mode = "calib"
    ):
        super().__init__()
        self.out_mode = out_mode 
        self.qoutput = qoutput
        self.use_res_connect = orig_module.use_res_connect


        q_layers_low = []
        q_layers_mid = []
        q_layers_high = []

        for module in orig_module.conv:
            if isinstance(module, torchvision.ops.Conv2dNormActivation):
                q_layers_low.append(QuantizedLayer(module[0], module[2], config, w_qconfig=config.quant.w_qconfig_low))
                q_layers_mid.append(QuantizedLayer(module[0], module[2], config, w_qconfig=config.quant.w_qconfig_med))
                q_layers_high.append(QuantizedLayer(module[0], module[2], config, w_qconfig=config.quant.w_qconfig_high))
            elif isinstance(module, nn.Conv2d):
                q_layers_low.append(QuantizedLayer(module, None, config,  w_qconfig=config.quant.w_qconfig_low, qoutput=False))
                q_layers_mid.append(QuantizedLayer(module, None, config,  w_qconfig=config.quant.w_qconfig_med, qoutput=False))
                q_layers_high.append(QuantizedLayer(module, None, config,  w_qconfig=config.quant.w_qconfig_high, qoutput=False))


            else: 
                q_layers_low.append(module)
                q_layers_mid.append(module)
                q_layers_high.append(module)


            self.conv_low = nn.Sequential(*q_layers_low)
            self.conv_mid = nn.Sequential(*q_layers_mid)
            self.conv_high = nn.Sequential(*q_layers_high)


            if self.qoutput:

                self.block_post_act_fake_quantize_med = Quantizer(
                    None, config.quant.a_qconfig_med
                )

                self.f_l, self.f_m, self.f_l, self.f_lmh = None, None, None, None
                self.lambda1, self.lambda2, self.lambda3 = config.quant.ptmq.lambda1,  config.quant.ptmq.lambda2,  config.quant.ptmq.lambda3
                self.mixed_p = config.quant.ptmq.mixed_p
    def forward(self, x):
        if self.use_res_connect:
            # x = x+self.conv(x)
            out_low = x + self.conv_low(x)
            out_mid = x + self.conv_mid(x)
            out_high = x + self.conv_high(x)

        else:
            #x = self.conv(x)
            out_low = self.conv_low(x)
            out_mid = self.conv_mid(x)
            out_high = self.conv_high(x)

        if self.qoutput:
            if self.out_mode == "calib":
                
                self.f_l = self.block_post_act_fake_quantize_med(out_low)
                self.f_m = self.block_post_act_fake_quantize_med(out_mid)
                self.f_h = self.block_post_act_fake_quantize_med(out_high)

                self.f_lmh = (  self.lambda1 * self.f_l
                    + self.lambda2 * self.f_m
                    + self.lambda3 * self.f_h)
                
                f_mixed = torch.where(torch.rand_like(out_mid)<self.mixed_p, out_mid, self.f_lmh)

                x = f_mixed 

            elif self.out_mode == "low":
                x = self.block_post_act_fake_quantize_med(out_low)
            elif self.out_mode == "med":
                x = self.block_post_act_fake_quantize_med(out_mid)
            elif self.out_mode == "high":
                x= self.block_post_act_fake_quantize_med(out_high)
            else: 
                raise ValueError(
                    f"Invalid out_mode '{self.out_mode}': only ['low', 'med', 'high'] are supported"
                )
        return x


# 5. vit , deit
quant_modules = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    InvertedResidual: QuantInvertedResidual
}


def load_model(config):
    # kwargs 기본값 설정
    config['kwargs'] = config.get('kwargs', dict())
    
    # 모델 초기화
    model = eval(config['type'])(**config['kwargs'])
    
    # 체크포인트 로드
    checkpoint = torch.load(config['path'], map_location='cpu')
    
    # state_dict 추출
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']  # 필요한 state_dict만 추출
    else:
        state_dict = checkpoint  # state_dict가 없으면 checkpoint 자체 사용
    
    # state_dict를 모델에 로드
    model.load_state_dict(state_dict, strict=False)  # strict=False로 일부 누락된 키 무시 가능
    
    return model


def set_qmodel_block_aqbit(model, out_mode):
    for name, module in model.named_modules():
        if isinstance(module, QuantizedBlock):
            # print(name)
            module.out_mode = out_mode

def set_qmodel_block_wqbit(model, out_mode):
    for name, module in model.named_modules():
        if isinstance(module, QuantizedBlock):
            print(name)
            module.out_mode = out_mode
