class Arguments(object):
    def __init__(self,
                 path,
                 img_size=(256, 256),
                 n_epochs=70,
                 ):
        self.img_size = img_size
        self.n_epochs = n_epochs


class Configure(object):
    def __init__(self,
                 in_channels,
                 out_channels,
                 patch_size,
                 hidden_dims,
                 n_hidden_layers,
                 n_attention_heads,
                 intermediate_dims=1024,
                 max_pos_emb=512,
                 hidden_drop_rate=0.1,
                 attention_drop_rate=0.1,
                 layer_norm_eps=1e-12,
                 sample_rate=4,
                 ):
        self.in_channels = in_channels
        self.in_channels = out_channels
        self.patch_size = patch_size
        self.hidden_dims = hidden_dims
        self.n_hidden_layers = n_hidden_layers
        self.n_attention_heads = n_attention_heads
        self.max_pos_emb = max_pos_emb
        self.hidden_drop_rate = hidden_drop_rate
        self.layer_norm_eps = layer_norm_eps
        self.attention_drop_rate = attention_drop_rate
        self.intermediate_dims = intermediate_dims
        self.sample_rate = sample_rate


ACT = {
    # 'gelu': None,
    # 'relu': F.relu,
}
