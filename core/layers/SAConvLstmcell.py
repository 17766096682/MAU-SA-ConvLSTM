import torch
import torch.nn as nn


class self_attention_memory_module(nn.Module):  # SAM
    def __init__(self, input_dim, hidden_dim):
        super(self_attention_memory_module, self).__init__()
        self.layer_q = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_k = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_k2 = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_v = nn.Conv2d(input_dim, input_dim, 1)
        self.layer_v2 = nn.Conv2d(input_dim, input_dim, 1)
        self.layer_z = nn.Conv2d(input_dim * 2, input_dim * 2, 1)
        self.layer_m = nn.Conv2d(input_dim * 3, input_dim * 3, 1)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(self, h, m):
        batch_size, channel, H, W = h.shape
        # feature aggregation
        ##### hidden h attention #####
        K_h = self.layer_k(h)
        Q_h = self.layer_q(h)
        K_h = K_h.view(batch_size, self.hidden_dim, H * W)
        Q_h = Q_h.view(batch_size, self.hidden_dim, H * W)
        Q_h = Q_h.transpose(1, 2)
        A_h = torch.softmax(torch.bmm(Q_h, K_h), dim=-1)  # batch_size, H*W, H*W
        V_h = self.layer_v(h)
        V_h = V_h.view(batch_size, self.input_dim, H * W)
        Z_h = torch.matmul(A_h, V_h.permute(0, 2, 1))

        # memory m attention #
        K_m = self.layer_k2(m)
        V_m = self.layer_v2(m)
        K_m = K_m.view(batch_size, self.hidden_dim, H * W)
        V_m = V_m.view(batch_size, self.input_dim, H * W)
        A_m = torch.softmax(torch.bmm(Q_h, K_m), dim=-1)
        V_m = self.layer_v2(m)
        V_m = V_m.view(batch_size, self.input_dim, H * W)
        Z_m = torch.matmul(A_m, V_m.permute(0, 2, 1))
        Z_h = Z_h.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        Z_m = Z_m.transpose(1, 2).view(batch_size, self.input_dim, H, W)

        W_z = torch.cat([Z_h, Z_m], dim=1)
        Z = self.layer_z(W_z)
        # Memory Updating
        combined = self.layer_m(torch.cat([Z, h], dim=1))  # 3 * input_dim
        mo, mg, mi = torch.split(combined, self.input_dim, dim=1)
        # 논문의 수식과 같습니다(figure)
        mi = torch.sigmoid(mi)
        new_m = (1 - mi) * m + mi * torch.tanh(mg)
        new_h = torch.sigmoid(mo) * new_m

        return new_h, new_m


class SA_Convlstm_cell(nn.Module):
    def __init__(self, in_channel, hidden_dim, configs):
        super(SA_Convlstm_cell, self).__init__()
        # hyperparrams
        self.input_channels = in_channel
        self.hidden_dim = hidden_dim
        self.kernel_size = configs.filter_size
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.device = configs.device

        self.attention_layer = self_attention_memory_module(input_dim=hidden_dim, hidden_dim=hidden_dim)
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels + self.hidden_dim, out_channels=4 * self.hidden_dim,
                      kernel_size=self.kernel_size, padding=self.padding)
            , nn.GroupNorm(4 * self.hidden_dim, 4 * self.hidden_dim))  # (num_groups, num_channels)

    def forward(self, x, hidden):
        h, c, m = hidden
        combined = torch.cat([x, h], dim=1)  # combine 해서 (batch_size, input_dim + hidden_dim, img_size[0], img_size[1])
        combined_conv = self.conv2d(combined)  # conv2d 후 (batch_size, 4 * hidden_dim, img_size[0], img_size[1])
        i, f, o, g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        h_next, m_next = self.attention_layer(h_next, m)

        return h_next, (h_next, c_next, m_next)  # 第一个用来外面的norm_layer

    def init_hidden(self, batch_size, img_size):  # h, c, m initalize
        h, w = img_size
        return (torch.zeros(batch_size, self.hidden_dim, h, w).to(self.device),
                torch.zeros(batch_size, self.hidden_dim, h, w).to(self.device),
                torch.zeros(batch_size, self.hidden_dim, h, w).to(self.device))


if __name__ == '__main__':
    from configs.radar_train_configs import configs

    parse = configs()
    configs = parse.parse_args()
    print(configs.num_hidden)

    model = SA_Convlstm_cell(configs.num_hidden, configs.num_hidden, configs).cuda()
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
