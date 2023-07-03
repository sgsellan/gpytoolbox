import numpy as np
import struct

def read_dmat(file_path):
    """
    Read a binary or ascii DMAT file and return a numpy array.

    Parameters
    ----------
    file_path : str
        Path to the DMAT file.

    Returns
    -------
    data : numpy.ndarray
        The data in the DMAT file.

    Notes
    -----
    dmat files are described well [here](https://libigl.github.io/file-formats/dmat/)

    Examples
    --------
    ```python
    # Read a DMAT file
    data = gpytoolbox.read_dmat('data.dmat')
    print(data)
    ```
    """
    with open(file_path, 'rb') as f:
        # Read the first line to determine if it's an ASCII or binary file
        header = f.readline().decode('ascii').strip()
        cols, rows = map(int, header.split())

        if cols == 0 and rows == 0:  # Binary file
            # Read the binary header
            binary_header = f.readline().decode('ascii').strip()
            cols, rows = map(int, binary_header.split())

            # Read binary data
            data = np.empty((rows, cols))
            for j in range(cols):
                for i in range(rows):
                    # Read 8-byte double precision floating point
                    float_bytes = f.read(8)
                    float_value = struct.unpack('<d', float_bytes)[0]
                    data[i, j] = float_value

        else:  # ASCII file
            # Read ASCII data
            data = np.empty((rows, cols))
            for j in range(cols):
                for i in range(rows):
                    while True:
                        # Read the next coefficient
                        c = f.read(0)
                        if c == b'' or c == b'\n':
                            break
                    # Read the next float value
                    float_value = float(f.readline().decode('ascii').strip())
                    data[i, j] = float_value

    return data
