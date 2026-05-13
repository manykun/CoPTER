import torch.nn as nn

class DualHeadNN(nn.Module):
    def __init__(self, state_dim, k_min_dim, k_max_dim):
        super(DualHeadNN, self).__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.k_min_head = nn.Linear(32, k_min_dim)
        self.k_max_head = nn.Linear(32, k_max_dim)

    def forward(self, x):
        x = self.shared_net(x)
        return self.k_min_head(x), self.k_max_head(x)
    
# # 三头神经网络（三个独立输出）
# class TripleHeadNN(nn.Module):
#     # 输入参数：状态维度、kmin维度、kmax维度、pmax维度
#     def __init__(self, state_dim, k_min_dim, k_max_dim, p_max_dim):
#         super(TripleHeadNN, self).__init__()
#         # 共享网络：实现state的特征提取，将输入状态转化为32维的共享特征，分别经过全连接层->激活函数->全连接层...
#         self.shared_net = nn.Sequential(
#             nn.Linear(state_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#         )
#         # 三头输出层：简单的线性层
#         self.k_min_head = nn.Linear(32, k_min_dim)
#         self.k_max_head = nn.Linear(32, k_max_dim)
#         self.p_max_head = nn.Linear(32, p_max_dim)
#     # 向前传播函数：定义数据通过网络的计算流程 输入状态->得到32维度共享特征->得到三个输出头
#     def forward(self, x):
#         x = self.shared_net(x)
#         return self.k_min_head(x), self.k_max_head(x), self.p_max_head(x)
#     # def __init__(self, state_dim, k_min_dim, k_max_dim, p_max_dim):
#     #     super(TripleHeadNN, self).__init__()
#     #     # 增强版共享网络：提升维度至128，增加特征表达能力
#     #     self.shared_net = nn.Sequential(
#     #         nn.Linear(state_dim, 64),       # 初始维度提升至64
#     #         nn.ReLU(),
#     #         nn.Linear(64, 128),             # 提升至128维，增强特征提取
#     #         nn.ReLU(),
#     #         nn.Linear(128, 128),            # 保持128维，深化特征学习
#     #         nn.ReLU(),
#     #         nn.Linear(128, 64),             # 适当收缩至64维
#     #         nn.ReLU(),
#     #     )
#     #     # 三头输出层：输入维度调整为64，匹配共享网络输出
#     #     self.k_min_head = nn.Linear(64, k_min_dim)
#     #     self.k_max_head = nn.Linear(64, k_max_dim)
#     #     self.p_max_head = nn.Linear(64, p_max_dim)
    
#     # # 向前传播函数：定义数据通过网络的计算流程
#     # def forward(self, x):
#     #     x = self.shared_net(x)
#     #     return self.k_min_head(x), self.k_max_head(x), self.p_max_head(x)
class TripleHeadACC(nn.Module):
    def __init__(self, state_dim, k_min_dim, k_max_dim, p_max_dim):
        super(TripleHeadACC, self).__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.k_min_head = nn.Linear(32, k_min_dim)
        self.k_max_head = nn.Linear(32, k_max_dim)
        self.p_max_head = nn.Linear(32, p_max_dim)
    
    def forward(self, x):
        x = self.shared_net(x)
        return self.k_min_head(x), self.k_max_head(x), self.p_max_head(x)

class TripleHeadCoPTER(nn.Module):
    def __init__(self, state_dim, k_min_dim, k_max_dim, p_max_dim):
        super(TripleHeadCoPTER, self).__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.k_min_head = nn.Linear(64, k_min_dim)
        self.k_max_head = nn.Linear(64, k_max_dim)
        self.p_max_head = nn.Linear(64, p_max_dim)
    
    def forward(self, x):
        x = self.shared_net(x)
        return self.k_min_head(x), self.k_max_head(x), self.p_max_head(x)