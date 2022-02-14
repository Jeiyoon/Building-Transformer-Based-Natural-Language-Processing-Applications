# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn
import torch.nn.functional as F

class FairseqDecoder(nn.Module):
    """Base class for decoders."""

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary

    def forward(self, prev_output_tokens, encoder_out):
        raise NotImplementedError

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            assert sample is not None and 'target' in sample
            out = self.adaptive_softmax.get_log_prob(net_output[0], sample['target'])
            return out.exp_() if not log_probs else out

        #logits = net_output[0].float()
        logits = net_output[0] #.float()
        if log_probs:
            return F.log_softmax(logits, dim=-1, dtype=torch.float32)
        else:
            return F.softmax(logits, dim=-1, dtype=torch.float32)

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        raise NotImplementedError

    def upgrade_state_dict(self, state_dict):
        return state_dict

class FairseqIncrementalDecoder(FairseqDecoder):
    """Base class for incremental decoders."""

    def __init__(self, dictionary):
        super().__init__(dictionary)

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        raise NotImplementedError

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder incremental state.

        This should be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        def apply_reorder_incremental_state(module):
            if module != self and hasattr(module, 'reorder_incremental_state'):
                module.reorder_incremental_state(
                    incremental_state,
                    new_order,
                )
        self.apply(apply_reorder_incremental_state)

    def set_beam_size(self, beam_size):
        """Sets the beam size in the decoder and all children."""
        if getattr(self, '_beam_size', -1) != beam_size:
            def apply_set_beam_size(module):
                if module != self and hasattr(module, 'set_beam_size'):
                    module.set_beam_size(beam_size)
            self.apply(apply_set_beam_size)
            self._beam_size = beam_size
