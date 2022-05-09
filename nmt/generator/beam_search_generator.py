import torch
from torch import nn
from nmt.generator.utils import tile_by_beam, compute_batch_indices, compute_topk_scores_and_seq, INF

"""
https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/beam_search.py 참고
"""

class BeamSearchGenerator(nn.Module):
    def __init__(self, encoder, decoder, config, device):
        super(BeamSearchGenerator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._tgt_bos_symbol = config.get("tgt_bos_symbol")
        self._tgt_eos_symbol = config.get("tgt_eos_symbol")
        self._decode_max_length = config.get("decode_max_length")
        self._beam_size = config.get("beam_size", 3)
        self._vocab_size = config.get("num_vocabs")
        self._device = device
        self._INF = INF

    __constants__ = ['_tgt_bos_symbol', '_tgt_eos_symbol', '_decode_max_length',
                     '_beam_size', '_vocab_size', '_device']

    def forward(self, src_input):
        batch_size = src_input.size(0)
        beam_size = self._beam_size
        vocab_size = self._vocab_size

        # (batch_size * beam_size, seq_length)
        src_input = tile_by_beam(src_input, beam_size)
        src_input = src_input.to(self._device)

        # calc encoder output before decoding for effiency
        encoder_output, src_key_padding_mask = self.encoder(src=src_input)

        # (batch_size * beam_size, 1)
        tgt_input = torch.full((batch_size * beam_size, 1), 
                               self._tgt_bos_symbol, 
                               dtype=torch.long, 
                               device=self._device)

        # (batch_size, beam_size, 1)
        alive_seq = torch.unsqueeze(tgt_input.reshape(batch_size, beam_size), dim=2).to(self._device)

        # (1, beam_size)
        initial_log_prob = torch.as_tensor([[0.] + [-float('inf')] * (beam_size - 1)]).to(self._device)
        # (batch_size, beam_size)
        alive_log_probs = initial_log_prob.repeat([batch_size, 1])

        # (batch_size, beam_size, 1)
        finished_seq = torch.zeros(alive_seq.shape, dtype=torch.int32).to(self._device)
        # (batch_size, beam_size)
        finished_scores = torch.ones([batch_size, beam_size]).to(self._device) * -self._INF
        finished_flags = torch.zeros([batch_size, beam_size], dtype=torch.bool).to(self._device)
        # (batch_size, )
        batch_finished_flags = torch.zeros([batch_size, ], dtype=torch.bool).to(self._device)

        multiplier = 2
        
        for timestep in range(0, self._decode_max_length - 1):
            # termination condition
            if torch.all(batch_finished_flags):
                break
            # Build decoder graph every step
            tgt_logits = self.decoder(tgt=tgt_input, 
                                      encoder_output=encoder_output,
                                      memory_key_padding_mask=src_key_padding_mask)
            # (batch_size * beam_size, step, vocab_size)
            tgt_logits = tgt_logits.permute(1,0,2)
            # (batch_size * beam_size, vocab_size)
            tgt_logits = tgt_logits[:, -1, :]
            # (batch_size * beam_size, vocab_size)
            curr_log_prob = torch.log_softmax(tgt_logits, dim=1)
            # (batch_size, beam_size, vocab_size)
            norm_log_prob = torch.reshape(curr_log_prob, [batch_size, beam_size, -1])

            # alive_log_prob: (batch_size, beam_size)
            # (batch_size, beam_size, vocab_size)
            log_probs = torch.unsqueeze(alive_log_probs, dim=2) + norm_log_prob

            length_penalty = torch.as_tensor(timestep + 1, dtype=torch.float32)

            # (batch_size, beam_size, vocab_size)
            curr_scores = log_probs / length_penalty
        
            # (batch_size, beam_size * vocab_size)
            flat_curr_scores = torch.reshape(curr_scores, [-1, beam_size * vocab_size])

            # (batch_size, beam_size * multiplier)
            topk_scores, topk_ids = torch.topk(flat_curr_scores, k=beam_size * multiplier, dim=1)

            # recovering the log prob for future use
            # (batch_size, beam_size * multipler)
            topk_log_probs = topk_scores * length_penalty

            # (batch_size, beam_size * multiplier)
            if timestep == 0:
                # for timestep == 0, simply return [0,1,2] rather than [0,0,0]
                topk_beam_index = torch.arange(beam_size).repeat([batch_size * multiplier]).reshape(batch_size, -1)
            else:
                topk_beam_index = topk_ids // vocab_size

            # (batch_size, beam_size * multiplier)
            topk_ids %= vocab_size

            # (batch_size, beam_size * multipler)
            batch_pos = compute_batch_indices(batch_size, beam_size * multiplier)

            # gather up the most probable (beam_size * multiplier) beams
            # (batch_size, beam_size * multiplier, timestep)
            topk_seq = alive_seq[batch_pos, topk_beam_index]
            # (batch_size, beam_size * multiplier, timestep + 1)
            topk_seq = torch.cat([topk_seq, torch.unsqueeze(topk_ids, dim=2)], dim=2)
            # (batch_size, beam_size * multiplier)
            topk_finished = torch.eq(topk_ids, self._tgt_eos_symbol)

            # 2) grow_alive
            # (batch_size, beam_size * multiplier)
            curr_scores = topk_scores + topk_finished.to(torch.float32) * -self._INF
            # alive_seq : (batch_size, beam_size, timestep + 1)
            # alive_log_probs : (batch_size, beam_size)
            alive_seq, alive_log_probs, _ = compute_topk_scores_and_seq(topk_seq, curr_scores, topk_log_probs, topk_finished, beam_size, batch_size)

            # 3) grow_finished
            # padding finished_* with zeros because we can't insert/replace values in while_loop,
            # but only concat and gather
            # (batch_size, beam_size, timestep + 1)
            finished_seq = torch.cat([finished_seq, torch.zeros([batch_size, beam_size, 1], dtype=torch.int32).to(self._device)], dim=2)

            # set the scores of unfinished seq to large negative
            # (batch_size, beam_size * multiplier)
            topk_scores += (1. - topk_finished.to(torch.float32)) * -self._INF
            # set the scores of completely finished seq to large negative
            # (batch_size, beam_size * multiplier)
            topk_scores += torch.unsqueeze(batch_finished_flags, dim=1).to(torch.float32) * -self._INF

            # concatenate the sequence and scores along the beam axis
            # (batch_size, beam_size * multiplier + beam_size, timestep + 1)
            finished_seq = torch.cat([finished_seq, topk_seq], dim=1)
            # (batch_size, beam_size * multiplier + beam_size)
            finished_scores = torch.cat([finished_scores, topk_scores], dim=1)
            # (batch_size, beam_size * multiplier + beam_size)
            finished_flags = torch.cat([finished_flags, topk_finished], dim=1)
            # finished_seq    : (batch_size, beam_size, timestep + 1)
            # finished_scores : (batch_size, beam_size)
            # finished_flags  : (batch_size, beam_size)
            finished_seq, finished_scores, finished_flags = compute_topk_scores_and_seq(finished_seq, finished_scores, finished_scores, finished_flags, beam_size, batch_size)

            # compute termination condition
            # best possible score of the most likely alive seq
            # (batch_size, )
            lower_bound_alive_scores = alive_log_probs[:, 0] / length_penalty

            # minimum score among finished seqs
            # (batch_size, )
            lowest_score_of_fin, _ = torch.min(finished_scores * finished_flags.to(torch.float32), dim=1)
            lowest_score_of_fin += (1. - torch.all(finished_flags, dim=1).to(torch.float32)) * -self._INF

            # (batch_size, 1)
            curr_batch_finished_flags = torch.unsqueeze(torch.all(torch.unsqueeze(torch.greater_equal(lowest_score_of_fin, lower_bound_alive_scores), dim=1), dim=1), dim=1)
            # (batch_size, 1)
            batch_finished_flags = torch.unsqueeze(batch_finished_flags, dim=1)
            # (batch_size, )
            batch_finished_flags = torch.any(torch.cat([batch_finished_flags, curr_batch_finished_flags], dim=1), dim=1)
            # (batch_size * beam_size, timestep + 1)
            tgt_input = torch.reshape(alive_seq, [batch_size * beam_size, -1])
        
        # (batch_size, beam_size, timestep)
        n_best_output = finished_seq
        # (batch_size, timestep)
        one_best_output = n_best_output[:, 0]

        return one_best_output
