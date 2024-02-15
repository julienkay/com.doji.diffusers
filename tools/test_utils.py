import numpy as np

def load_data_f(file_path):
    """
    Reads a text file containing comma-separated values and returns a NumPy array.

    Args:
    - file_path (str): The path to the text file.

    Returns:
    - numpy.ndarray: The NumPy array containing the values from the text file.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    data_string = ''.join(lines)
    return np.fromstring(data_string, dtype=float, sep=',')

def scheduler_samples():
    """
    Returns a NumPy array with deterministic random samples with shape (4, 3, 8, 8)

    Returns:
    - numpy.ndarray: The NumPy array containing thesample values.
    """
    return load_data_f("..\com.doji.diffusers\Tests\Editor\Resources\scheduler_test_random_samples.txt")
