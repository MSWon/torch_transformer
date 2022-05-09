import torch


INF = 999


class InValidGeneratorError(Exception):
    def __init__(self, generator_type: str):
        self.generator_type = generator_type

    def __str__(self):
        error_msg = f"'{self.generator_type}' is invalid generator\nValid Generator: ['greedy', 'beam_search']"
        return error_msg


def tile_by_beam(input_tensor: torch.Tensor, beam_size: int):
    """
    input_tensor: tensor([[1, 2, 3, 4],
                         [5, 6, 7, 8]])
    beam_size: 2

    return: tensor([[1, 2, 3, 4],
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [5, 6, 7, 8]])
    """
    tiled_tensor = input_tensor.repeat_interleave(beam_size, dim=0)
    return tiled_tensor

def compute_batch_indices(batch_size: int, beam_size: int):
    '''
    Computes i'th coordinate that contains the batch index for gathers
    Args:
        batch_size : Batch size
        beam_size : Beam size
    Returns:
        batch_pos : [batch_size, beam_size] tensor of ids
    '''
    batch_pos = torch.arange(batch_size * beam_size) // beam_size
    batch_pos = torch.reshape(batch_pos, [batch_size, beam_size])
    return batch_pos

def compute_topk_scores_and_seq(sequences: torch.Tensor, scores: torch.Tensor,
                                scores_to_gather: torch.Tensor, flags: torch.Tensor,
                                beam_size: int, batch_size: int):
    '''
    Given sequences and scores, will gather the top k=beam_size sequence
    Args:
        sequences : Tensor of sequence that we need to gather from
            [batch_size, beam_size, seq_length]
        scores : Tensor of scores for each sequence in sequences. Used to compute the topk
            [batch_size, beam_size]
        scores_to_gather : Tensor of scores for each sequence in sequences. Used to return the gathered score
            [batch_size, beam_size]
        scores and scores_to_gather differs because for finished seq, we return length-penalized score, but for next alive seq, we return log_probs as is.
        flags : Tensor of bools for sequences that say whether a sequence has reached EOS or not
        beam_size : int
        batch_size : int
    Returns:
        Tuple of
            (topk_seq               [batch_size, beam_size, decoded_length],
             topk_gathered_scores   [batch_size, beam_size],
             topk_finished_flags    [batch_size, beam_size],
             topk_indices           [batch_size, beam_size])
    '''
    # (batch_size, beam_size)
    _, topk_indices = torch.topk(scores, k=beam_size, dim=1)

    # (batch_size, beam_size)
    batch_pos = compute_batch_indices(batch_size, beam_size)

    # gather up the highest scoring sequences
    topk_seq = sequences[batch_pos, topk_indices]
    topk_flags = flags[batch_pos, topk_indices]
    topk_gathered_scores = scores_to_gather[batch_pos, topk_indices]
    return topk_seq, topk_gathered_scores, topk_flags