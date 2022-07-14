import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
import math
import numpy as np
 

class MLP(nn.Module):
    def __init__(self, layer_sizes=[512, 512], last_act=False):

        super(MLP, self).__init__()

        self.MLP = nn.Sequential()
        
        if last_act:
         for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=True))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.LeakyReLU(negative_slope=0.1, inplace=True))
        else:
         for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=True))
            if i < (len(layer_sizes[:-1])-1):
               self.MLP.add_module(name="A{:d}".format(i), module=nn.LeakyReLU(negative_slope=0.1, inplace=True))
        
        

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                 nn.init.constant_(m.bias, 0.0)

        
    def forward(self, x):
        x = self.MLP(x)
        return x

class MLP_Decoder(nn.Module):
    def __init__(self, layer_sizes=[512, 512], last_act=False):

        super(MLP_Decoder, self).__init__()

        self.MLP = nn.Sequential()
        
        if last_act:
         for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=True))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU(inplace=True))
        else:
         for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=True))
            if i < (len(layer_sizes[:-1])-1):
               self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU(inplace=True))
        
        

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                 nn.init.constant_(m.bias, 0.0)

        
    def forward(self, x):
        x = self.MLP(x)
        return x


class NP_HEAD(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes=10, memory_max_length=2560 ):
        super(NP_HEAD, self).__init__()
 
        self.latent_path = MLP(layer_sizes=[input_dim+num_classes, latent_dim, latent_dim], last_act=True)
        self.deterministic_path = MLP(layer_sizes=[input_dim+num_classes, latent_dim, latent_dim], last_act=True)
        self.mean_net = MLP(layer_sizes=[latent_dim, latent_dim, latent_dim])
        self.log_var_net = MLP(layer_sizes=[latent_dim, latent_dim, latent_dim]) 
        self.num_classes = num_classes
        self.fc_decoder =  MLP(layer_sizes=[2 * latent_dim + input_dim, input_dim, input_dim], last_act=True)
        #print(self.fc_decoder)
        self.classifier = nn.Linear(input_dim, num_classes, bias=True)
        self.memory_max_length = memory_max_length 

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                 nn.init.constant_(m.bias, 0.0)

    def reparameterize(self, mean, std): 
        eps = torch.randn_like(std)
        return eps * std + mean


    def forward(self, x_in, deterministic_memory, latent_memory, forward_times=10, phase_train=True, x_context_in=None, labels_in=None, labels_context_in=None, update_deterministic_memory=True):

      
      
      if phase_train:
 
        x_combine_deterministic_input = torch.cat((x_context_in, labels_context_in), dim=-1)  
        x_representation_deterministic =   self.deterministic_path(x_combine_deterministic_input) 
        x_combine_latent_input = torch.cat((x_in, labels_in), dim=-1)  
        x_representation_latent =  self.latent_path(x_combine_latent_input)
        mean_x = self.mean_net(x_representation_latent.mean(0))
        log_var_x = self.log_var_net(x_representation_latent.mean(0))
        sigma_x = 0.1 + 0.9 * F.softplus(log_var_x) 
       

        latent_z_target = None
        B = x_in.size(0)
        for i in range(0, forward_times):
          z = self.reparameterize(mean_x, sigma_x)
          z = z.unsqueeze(0)
          if i == 0:
              latent_z_target = z
          else:
              latent_z_target = torch.cat((latent_z_target, z))
        
       
        x_target_in_expand = x_in.unsqueeze(0).expand(forward_times ,-1, -1)
        context_representation_deterministic_expand = x_representation_deterministic.mean(0).unsqueeze(0).unsqueeze(1).expand(forward_times, B, -1) 
        latent_z_target_expand = latent_z_target.unsqueeze(1).expand(-1, B, -1)
        

        if update_deterministic_memory:
           deterministic_memory = torch.cat((deterministic_memory, x_representation_deterministic.detach()), dim=0)
           if deterministic_memory.size(0) > self.memory_max_length:
                   Diff = deterministic_memory.size(0) -  self.memory_max_length
                   deterministic_memory = deterministic_memory[Diff:, :]               


        latent_memory = torch.cat((latent_memory, x_representation_latent.detach()), dim=0)
        if latent_memory.size(0) > self.memory_max_length:
                   Diff = latent_memory.size(0) -  self.memory_max_length
                   latent_memory = latent_memory[Diff:, :]

   
        ################## decoder ##################

        decoder_input_cat = torch.cat((latent_z_target_expand, x_target_in_expand, context_representation_deterministic_expand), dim=-1)
        T, B, D = decoder_input_cat.size()
        output_function = self.fc_decoder(decoder_input_cat)
        output = self.classifier(output_function)
         
        return output, mean_x, sigma_x, deterministic_memory, latent_memory
      else:

    
        mean = self.mean_net(latent_memory.mean(0))
        log_var = self.log_var_net(latent_memory.mean(0))
        sigma  = 0.1 + 0.9 * F.softplus(log_var ) 
 

        latent_z_target = None
        B = x_in.size(0)

        for i in range(0, forward_times):
          z = self.reparameterize(mean , sigma)
          z = z.unsqueeze(0)
          if i == 0:
              latent_z_target = z
          else:
              latent_z_target = torch.cat((latent_z_target, z))


        x_target_in_expand = x_in.unsqueeze(0).expand(forward_times ,-1, -1)
        latent_z_target_expand = latent_z_target.unsqueeze(1).expand(-1, B, -1)
       
        context_representation_deterministic_expand = deterministic_memory.mean(0).unsqueeze(0).unsqueeze(1).expand(forward_times, B, -1) 

        
        decoder_input_cat = torch.cat((latent_z_target_expand, x_target_in_expand, context_representation_deterministic_expand), dim=-1)
        T, B, D = decoder_input_cat.size()
        output_function = self.fc_decoder(decoder_input_cat)
        output = self.classifier(output_function)
        
        return output
