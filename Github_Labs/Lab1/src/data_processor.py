def normalize(data):
    """
    Normalizes list of numbers to range [0, 1].
    """
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Input must be a non empty list")
    if not all(isinstance(x, (int, float)) for x in data):
        raise ValueError("All elements must be numbers")
    
    min_val = min(data)
    max_val = max(data)
    
    if max_val == min_val:
        return [0.0] * len(data)
    
    return [(x - min_val) / (max_val - min_val) for x in data]


def standardize(data):
    """
    Standardize a list of numbers using z-score normalization.
    """
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Input must be a non empty list")
    if not all(isinstance(x, (int, float)) for x in data):
        raise ValueError("All elements must be numbers")
    
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std = variance ** 0.5
    
    if std == 0:
        return [0.0] * len(data)
    
    return [(x - mean) / std for x in data]


def fill_missing(data, fill_value=0):
    """
    Fills None values in a list with a specified fill value.
    """
    if not isinstance(data, list):
        raise ValueError("Input must be a list")
    
    return [fill_value if x is None else x for x in data]


def compute_statistics(data):
    """
    Computes basic stat. for a list of numbers and returns a dictionary with the mean, min, max, and count.
    """
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Input must be a non empty list")
    if not all(isinstance(x, (int, float)) for x in data):
        raise ValueError("All elements must be numbers")
    
    return {
        "mean": sum(data) / len(data),
        "min": min(data),
        "max": max(data),
        "count": len(data)
    }
