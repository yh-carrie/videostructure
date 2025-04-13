import torch


class BoQBlock(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8):
        super(BoQBlock, self).__init__()

        self.encoder = torch.nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=nheads,
            dim_feedforward=4 * in_dim,
            batch_first=True,
            dropout=0.0,
        )
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))

        # the following two lines are used during training only, you can cache their output in eval.
        self.self_attn = torch.nn.MultiheadAttention(
            in_dim, num_heads=nheads, batch_first=True
        )
        self.norm_q = torch.nn.LayerNorm(in_dim)
        #####

        self.cross_attn = torch.nn.MultiheadAttention(
            in_dim, num_heads=nheads, batch_first=True
        )
        self.cross_attn0 = torch.nn.MultiheadAttention(
            in_dim, num_heads=nheads, batch_first=True
        )
        self.norm_qx = torch.nn.LayerNorm(in_dim)
        self.norm_x0 = torch.nn.LayerNorm(in_dim)

    def forward(self, x, x0):
        B = x.size(0)
        x = self.encoder(x)

        q = self.queries.repeat(B, 1, 1)

        # the following two lines are used during training.
        # for stability purposes
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)
        #######

        qx, attn = self.cross_attn(q, x, x)
        qx = self.norm_qx(qx)

        x0 = x0 + self.cross_attn0(x0, qx, qx)[0]
        x0 = self.norm_x0(x0)
        return x, x0, attn


class BoQ(torch.nn.Module):
    def __init__(
        self,
        in_channels=1024,
        proj_channels=1024,
        num_queries=16,
        num_layers=2,
        row_dim=32,
    ):
        super().__init__()
        self.proj_c = torch.nn.Conv2d(
            in_channels, proj_channels, kernel_size=3, padding=1
        )
        self.norm_input = torch.nn.LayerNorm(proj_channels)

        in_dim = proj_channels
        self.boqs = torch.nn.ModuleList(
            [
                BoQBlock(in_dim, num_queries, nheads=in_dim // 64)
                for _ in range(num_layers)
            ]
        )

        self.fc = torch.nn.Linear(num_layers * num_queries, row_dim)
        self.self_attn = torch.nn.MultiheadAttention(
            in_dim, num_heads=in_dim // 64, batch_first=True
        )
        self.norm_q = torch.nn.LayerNorm(in_dim)
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        self.cross_attn1 = torch.nn.MultiheadAttention(
            in_dim, num_heads=in_dim // 64, batch_first=True
        )
        self.norm_qx1 = torch.nn.LayerNorm(in_dim)

    def forward(self, x):
        # reduce input dimension using 3x3 conv when using ResNet
        # x = self.proj_c(x)
        # x = x.flatten(2).permute(0, 2, 1)
        # x = self.norm_input(x)
        # q = self.queries.repeat(x.size(0), 1, 1)
        # q = q + self.self_attn(q, q, q)[0]
        # q = self.norm_q(q)

        # qx = self.cross_attn1(q, x, x)[0]
        # qx = self.norm_qx1(qx)

        x0 = x[:, :1]
        x = x[:, 1:]
        B, _, C = x.shape
        x = x.permute(0, 2, 1).contiguous().view(B, C, 18, 9)
        x = self.proj_c(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)
        for i in range(len(self.boqs)):
            x, x0, attn = self.boqs[i](x, x0)
        return x0,attn
