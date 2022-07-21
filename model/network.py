import torch
import torch.nn as nn
import torch.nn.functional as F

 
    
def default_conv(in_channels, out_channels, kernel_size):        
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=((kernel_size)//2), bias=True)      

class ResBlock(nn.Module):                                     
    def __init__(self, conv, n_channel, kernel_size):           
        super(ResBlock, self).__init__()

        m = []
        for i in range(2):
            m.append(conv(n_channel, n_channel, kernel_size))   
            if i == 0: m.append(nn.ReLU())
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

    
class Modulation(nn.Module):                                                  
    def __init__(self, conv):                                                   
        super(Modulation, self).__init__()

        self.mlp_1 = nn.Sequential(conv(64, 64, 1))      
        self.mlp_2 = nn.Sequential(conv(64, 64, 1))  
        self.mlp_3 = nn.Sequential(conv(2 * 64, 64, 1))                       
        
    def forward(self, input):
        f_guide, f_source = input                                               
        f_guide = self.mlp_1(f_guide)                                                    
        f_source = self.mlp_2(f_source)                                       

        # Source-to-Guide
        source_neighbor = 11                                                         
        B, C, H, W = f_source.shape
        fs_unfold = F.unfold(f_source, kernel_size=source_neighbor, dilation=1, stride=1, padding=source_neighbor//2)
        fs_unfold = fs_unfold.view(B, C, source_neighbor**2, H, W)
        filters = (fs_unfold * f_guide.unsqueeze(2)).sum(1).softmax(1)          
        f_s2g = (fs_unfold * filters.unsqueeze(1)).sum(2)                       

        # Guide-to-Source
        guide_neighbor = 5                                                          
        B, C, H, W = f_guide.shape
        fg_unfold = F.unfold(f_guide, kernel_size=guide_neighbor, dilation=1, stride=1, padding=guide_neighbor//2)
        fg_unfold = fg_unfold.view(B, C, guide_neighbor**2, H, W)
        filters = (fg_unfold * f_source.unsqueeze(2)).sum(1).softmax(1)         
        f_g2s = (fg_unfold * filters.unsqueeze(1)).sum(2)                      

        fusion = self.mlp_3(torch.cat([f_s2g,f_g2s], dim=1))                                   
        return fusion 
                                
                             
class MMSR_net(nn.Module):                     
    def __init__(self, conv=default_conv, weights_regularizer=None):                                     
        super(MMSR_net, self).__init__()

        self.branch_guide = nn.Sequential(conv(3, 64, 3), nn.ReLU(), conv(64, 64, 3), nn.ReLU(), 
                                          ResBlock(conv, 64, 3), ResBlock(conv, 64, 3))       
        self.branch_source = nn.Sequential(conv(1, 64, 1), nn.ReLU(), conv(64, 64, 1), nn.ReLU(), 
                                           ResBlock(conv, 64, 1), ResBlock(conv, 64, 1))
        self.modulation = nn.Sequential(Modulation(conv))     
        self.branch_pred = nn.Sequential(ResBlock(conv, 64, 1), ResBlock(conv, 64, 1), ResBlock(conv, 64, 1),
                                         conv(64, 1, 1)) 

        reg_guide, reg_source, reg_modu, reg_pred = weights_regularizer[:4]
        self.params_with_regularizer = [{'params':self.branch_guide.parameters(),'weight_decay':reg_guide},\
                                        {'params':self.branch_source.parameters(),'weight_decay':reg_source},\
                                        {'params':self.modulation.parameters(),'weight_decay':reg_modu},\
                                        {'params':self.branch_pred.parameters(),'weight_decay':reg_pred}]
        
    def forward(self, input):
        guide, source = input                                                 
        source = F.interpolate(source, [guide.shape[2],guide.shape[3]], mode='bilinear', align_corners=False)  
        fusion = self.modulation([self.branch_guide(guide), self.branch_source(source)])                         
        predict = self.branch_pred(fusion)                      
        return predict
