def find_sliding_window(input_size, target_windows, overlap):
    """
    Finds window_size, hop_size, and number of windows closest to target_windows.

    If `overlap == 0` and `input_size` divides evenly by `target_windows`, returns
    the exact division `(input_size//target_windows, input_size//target_windows, target_windows)`.

    Args:
        input_size (int): Total number of samples in your signal.
        target_windows (int): Desired number of windows.
        overlap (int): Number of samples overlapping between consecutive windows.

    Returns:
        tuple: (window_size, hop_size, num_windows), or None if no valid configuration.
    """
    # Exact division when no overlap
    if overlap == 0 and input_size % target_windows == 0:
        win = input_size // target_windows
        return (win, win, target_windows)

    best_config = None
    best_diff = float('inf')

    # Search for best approximate configuration
    for window_size in range(overlap + 1, input_size + 1):
        hop_size = window_size - overlap
        if hop_size <= 0:
            continue

        num_windows = (input_size - overlap) // hop_size
        if num_windows < 1:
            continue

        diff = abs(num_windows - target_windows)
        divides_evenly = (input_size - overlap) % hop_size == 0
        best_divides = False
        if best_config is not None:
            best_divides = ((input_size - overlap) % best_config[1] == 0)

        # Prefer smaller diff, then even division
        if diff < best_diff or (diff == best_diff and divides_evenly and not best_divides):
            best_diff = diff
            best_config = (window_size, hop_size, num_windows)

    return best_config