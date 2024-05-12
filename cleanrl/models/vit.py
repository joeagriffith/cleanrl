import torch
from rvit import RegisteredVisionTransformer

class ViT(RegisteredVisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x)
    
    def encode(self, x, tokens=False):
        if not tokens:
            x = self._process_input(x)
        n = x.shape[0]

        batch_cls = self.class_token.expand(n, -1, -1)
        x = torch.cat((batch_cls, x), dim=1)

        x = self.encoder(x)

        return x


