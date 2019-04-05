# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch
Tested in PyTorch 0.2.0 and 0.3.0

@author: Junxiao Song
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def my_mul(A, B):
    Sum=0
    for i in range(8): 
        Sum+=A[i]*B[i]
    return Sum

class my_conv:
    def __init__(self, channel_num):
        self.channel_num = channel_num
        Dic = [[0,0],[0,1],[0,2],[0,3],[1,1],[1,2],[1,3],[2,2],[2,3],[3,3]]
        My_list = [Variable(torch.rand(10).cuda(),requires_grad=True) for i in range(channel_num)]
        His_list = [Variable(torch.rand(10).cuda(),requires_grad=True) for i in range(channel_num)]
        #My_list = [Variable(torch.rand(10),requires_grad=True) for i in range(channel_num)]
        #His_list = [Variable(torch.rand(10),requires_grad=True) for i in range(channel_num)]

        self.D = Variable(torch.Tensor(2,channel_num,8,8).cuda())

        for j in range(channel_num):
            for i in range(10):
                x = Dic[i][0]
                y = Dic[i][1]
                self.D[0,j,x,y]=My_list[j][i]
                self.D[0,j,y,x]=My_list[j][i]
                self.D[1,j,x,y]=His_list[j][i]
                self.D[1,j,y,x]=His_list[j][i]

                x = 7-Dic[i][0]
                y = Dic[i][1]
                self.D[0,j,x,y]=My_list[j][i]
                self.D[0,j,y,x]=My_list[j][i]
                self.D[1,j,x,y]=His_list[j][i]
                self.D[1,j,y,x]=His_list[j][i]
            
                x = Dic[i][0]
                y = 7-Dic[i][1]
                self.D[0,j,x,y]=My_list[j][i]
                self.D[0,j,y,x]=My_list[j][i]
                self.D[1,j,x,y]=His_list[j][i]
                self.D[1,j,y,x]=His_list[j][i]

                x = 7- Dic[i][0]
                y = 7- Dic[i][1]
                self.D[0,j,x,y]=My_list[j][i]
                self.D[0,j,y,x]=My_list[j][i]
                self.D[1,j,x,y]=His_list[j][i]
                self.D[1,j,y,x]=His_list[j][i]

    def __call__(self, input):
        Batch_size = input.size()[0]
        #Output =[]
        final = Variable(torch.Tensor(Batch_size, 92*self.channel_num).cuda())

        for b in range(Batch_size):

            cnt = 0
            for j in range(self.channel_num):
                for i in range(8):
                    final[b,cnt]=torch.dot(input[b,0,i],self.D[0,j,i])
                    cnt+=1
                    final[b,cnt]=torch.dot(input[b,1,i],self.D[1,j,i])
                    cnt+=1
                    #row_final.append(my_mul(input[b][0][i],self.D[0][j][i]))
                    #row_final.append(my_mul(input[b][1][i],self.D[1][j][i]))
                    #final = torch.cat((final,

            tb0 = torch.t(input[b,0])
            tb1 = torch.t(input[b,1])
            for j in range(self.channel_num):
                for i in range(8):
                    #cl_final.append(torch.matmul(torch.t(torch.t(input[b][0])[i]),self.D[0][j][i]))
                    #cl_final.append(torch.matmul(torch.t(torc0.t(input[b][1])[i]),self.D[1][j][i]))
                    final[b,cnt]=torch.dot(tb0[i],self.D[0,j,i])
                    cnt+=1
                    final[b,cnt]=torch.dot(tb1[i],self.D[1,j,i])
                    cnt+=1

            for j in range(self.channel_num):
                for i in range(15):
                    x=0
                    y=0
                    if i<=7:
                        y=i
                    else:
                        x=i-7
                    Sum0 = 0
                    Sum1 = 0
                    while x<=7 and y<=7 and x>=0 and y>=0:
                        Sum0 += input[b,0,x,y]*self.D[0,j,x,y]
                        Sum1 += input[b,1,x,y]*self.D[1,j,x,y]
                        x+=1
                        y+=1
                    final[b,cnt]=Sum0
                    cnt+=1
                    final[b,cnt]=Sum1
                    cnt+=1

            for j in range(self.channel_num):
                for i in range(15):
                    x=0
                    y=7
                    if i<=7:
                        y=i
                    else:
                        x=i-7
                    Sum0 = 0
                    Sum1 = 0
                    while x<=7 and y<=7 and x>=0 and y>=0:
                        Sum0 += input[b,0,x,y]*self.D[0,j,x,y]
                        Sum1 += input[b,1,x,y]*self.D[1,j,x,y]
                        x+=1
                        y-=1
                    final[b,cnt]=Sum0
                    cnt+=1
                    final[b,cnt]=Sum1
                    cnt+=1

            #final = diag_final_2+diag_final_1+row_final+cl_final
            #final = torch.cat(final)
            #Output.append(final)
         
        #Output=torch.stack(Output)
        return final


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()
        self.board_width = board_width
        self.board_height = board_height

        self.myconv = my_conv(3)

        # common layers
        #self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        #self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        #self.my_conv = my_conv

        # action policy layers
        #self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(276, 100)
        self.act_fc2 = nn.Linear(100, board_width*board_height)

        # state value layers
        #self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(276, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.myconv(state_input))
        #x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))

        # action policy layers
        #x_act = F.relu(self.act_conv1(x))
        #x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.relu(self.act_fc1(x))
        x_act = F.log_softmax(self.act_fc2(x_act))

        # state value layers
        #x_val = F.relu(self.val_conv1(x))
        #x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        fname = 'current_policy.model'
        if os.path.isfile(fname) and model_file is None:
            model_file = fname
        if model_file:
            if use_gpu:
                self.policy_value_net.load_state_dict(torch.load(model_file))
            else:
                self.policy_value_net.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 2, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return loss.data[0], entropy.data[0]

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
