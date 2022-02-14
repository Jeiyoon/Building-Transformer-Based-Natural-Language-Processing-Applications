import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self._is_generation_fast = False
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out, padding_mask = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out, padding_mask)

        return decoder_out
