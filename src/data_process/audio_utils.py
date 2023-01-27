
def seconds_to_frame(seconds: float, sample_rate: int, hop_size: int) -> float:
    return seconds * sample_rate / hop_size

