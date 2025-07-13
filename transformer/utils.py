

def str_to_int_list(value):
    """Convert a comma-separated string to a list of integers.
    Args:
        value (str): A string of integers separated by commas, e.g., "1,2,3".
    Returns:
        list: A list of integers.
    """
    return [int(x) for x in value.split(',')]