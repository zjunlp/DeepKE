import torch
import torch.nn as nn
import torch.nn.functional as F
from deepke.model import BasicModule, Embedding, VarLenLSTM


class Capsule(BasicModule):
    def __init__(self, vocab_size, config):
        super(Capsule, self).__init__()
        self.model_name = 'Capsule'
        self.vocab_size = vocab_size
        self.word_dim = config.model.word_dim
        self.pos_size = config.model.pos_size
        self.pos_dim = config.model.pos_dim
        self.hidden_dim = config.model.hidden_dim

        self.num_primary_units = config.capsule.num_primary_units
        self.num_output_units = config.capsule.num_output_units
        self.primary_channels = config.capsule.primary_channels
        self.primary_unit_size = config.capsule.primary_unit_size
        self.output_unit_size = config.capsule.output_unit_size
        self.num_iterations = config.capsule.num_iterations

        self.embedding = Embedding(self.vocab_size, self.word_dim, self.pos_size, self.pos_dim)
        self.input_dim = self.word_dim + self.pos_dim * 2
        self.lstm = VarLenLSTM(
            self.input_dim,
            self.hidden_dim,
        )
        self.capsule = CapsuleNet(self.num_primary_units, self.num_output_units, self.primary_channels,
                                  self.primary_unit_size, self.output_unit_size, self.num_iterations)

    def forward(self, input):
        *x, mask = input
        x = self.embedding(x)
        x_lens = torch.sum(mask.gt(0), dim=-1)
        _, hn = self.lstm(x, x_lens)
        out = self.capsule(hn)
        return out  # B, num_output_units, output_unit_size

    def predict(self, output):
        v_mag = torch.sqrt((output**2).sum(dim=2, keepdim=False))
        pred = v_mag.argmax(1, keepdim=False)
        return pred

    def loss(self, input, target, size_average=True):
        batch_size = input.size(0)

        v_mag = torch.sqrt((input**2).sum(dim=2, keepdim=True))

        max_l = torch.relu(0.9 - v_mag).view(batch_size, -1)**2
        max_r = torch.relu(v_mag - 0.1).view(batch_size, -1)**2

        loss_lambda = 0.5
        T_c = target
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
        L_c = L_c.sum(dim=1)

        if size_average:
            L_c = L_c.mean()

        return L_c


class CapsuleNet(nn.Module):
    def __init__(self, num_primary_units, num_output_units, primary_channels, primary_unit_size, output_unit_size,
                 num_iterations):
        super(CapsuleNet, self).__init__()
        self.primary = CapsuleLayer(in_units=0,
                                    out_units=num_primary_units,
                                    in_channels=primary_channels,
                                    unit_size=primary_unit_size,
                                    use_routing=False,
                                    num_iterations=0)

        self.iteration = CapsuleLayer(in_units=num_primary_units,
                                      out_units=num_output_units,
                                      in_channels=primary_unit_size,
                                      unit_size=output_unit_size,
                                      use_routing=True,
                                      num_iterations=num_iterations)

    def forward(self, input):
        return self.iteration(self.primary(input))


class ConvUnit(nn.Module):
    def __init__(self, in_channels):
        super(ConvUnit, self).__init__()
        self.conv0 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=8,  # fixme constant
            kernel_size=9,  # fixme constant
            stride=2,  # fixme constant
            bias=True)

    def forward(self, x):
        return self.conv0(x)


class CapsuleLayer(nn.Module):
    def __init__(self, in_units, out_units, in_channels, unit_size, use_routing, num_iterations):
        super(CapsuleLayer, self).__init__()
        self.in_units = in_units
        self.out_units = out_units
        self.in_channels = in_channels
        self.unit_size = unit_size
        self.use_routing = use_routing

        if self.use_routing:
            self.W = nn.Parameter(torch.randn(1, in_channels, out_units, unit_size, in_units))
            self.num_iterations = num_iterations
        else:

            def create_conv_unit(unit_idx):
                unit = ConvUnit(in_channels=in_channels)
                self.add_module("unit_" + str(unit_idx), unit)
                return unit

            self.units = [create_conv_unit(i) for i in range(self.out_units)]

    @staticmethod
    def squash(s):
        # This is equation 1 from the paper.
        mag_sq = torch.sum(s**2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        if self.use_routing:
            return self.routing(x)
        else:
            return self.no_routing(x)

    def no_routing(self, x):
        # Each unit will be (batch, channels, feature).
        u = [self.units[i](x) for i in range(self.out_units)]

        # Stack all unit outputs (batch, unit, channels, feature).
        u = torch.stack(u, dim=1)

        # Flatten to (batch, unit, output).
        u = u.view(x.size(0), self.out_units, -1)

        # Return squashed outputs.
        return CapsuleLayer.squash(u)

    def routing(self, x):
        batch_size = x.size(0)

        # (batch, in_units, features) -> (batch, features, in_units)
        x = x.transpose(1, 2)

        # (batch, features, in_units) -> (batch, features, out_units, in_units, 1)
        x = torch.stack([x] * self.out_units, dim=2).unsqueeze(4)

        # (batch, features, out_units, unit_size, in_units)
        W = torch.cat([self.W] * batch_size, dim=0)

        # Transform inputs by weight matrix.
        # (batch_size, features, out_units, unit_size, 1)
        u_hat = torch.matmul(W, x)

        # Initialize routing logits to zero.
        b_ij = torch.zeros(1, self.in_channels, self.out_units, 1).to(x.device)

        # Iterative routing.
        num_iterations = self.num_iterations
        for iteration in range(num_iterations):
            # Convert routing logits to softmax.
            c_ij = F.softmax(b_ij, dim=1)

            # (batch, features, out_units, 1, 1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            # Apply routing (c_ij) to weighted inputs (u_hat).
            # (batch_size, 1, out_units, unit_size, 1)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # (batch_size, 1, out_units, unit_size, 1)
            v_j = CapsuleLayer.squash(s_j)

            # (batch_size, features, out_units, unit_size, 1)
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)

            # (1, features, out_units, 1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            # Update b_ij (routing)
            b_ij = u_vj1

        # (batch_size, out_units, unit_size, 1)
        return v_j.squeeze()


if __name__ == '__main__':
    net = CapsuleNet(num_primary_units=8,
                     num_output_units=13,
                     primary_channels=10,
                     primary_unit_size=8,
                     output_unit_size=20,
                     num_iterations=5)
    inputs = torch.randn(4, 10, 10)
    outs = net(inputs)
    print(outs.shape)  # (4, 13, 20)
