import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here


    N = len(seqs)
    # Determine target length L
    if max_len is None:
        L = max(len(seq) for seq in seqs)
    else:
        L = max_len

    # Infer dtype from pad_value (preserves int/float)
    dtype = np.array(pad_value).dtype

    # Initialize result array with pad_value
    result = np.full((N, L), pad_value, dtype=dtype)

    # Fill each row with the sequence (truncated if necessary)
    for i, seq in enumerate(seqs):
        # Truncate if longer than L
        seq_trimmed = seq[:L]
        # Copy into result row
        result[i, :len(seq_trimmed)] = seq_trimmed

    return result