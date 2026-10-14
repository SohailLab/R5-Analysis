#Runs the square binning can also optionally convert sdt to tiff first 

import numpy as np
import tifffile as tiff
from sdtfile import SdtFile
import os


def bin_array_2d(arr, bin_size): # TO DO maybe cut down on the comments lol 
    """
    Subfunction that does the binning
    """
    #Divide the original array into blocks of bin_size by bin_size
    #arr.shape[0]: Original height (number of rows) of the 2D array.
    #arr.shape[0] // bin_size: The integer division calculates how many full blocks of bin_size can fit
    #along the height. This will be the new height of the binned array.
    #arr.shape[1]: Original width (number of columns) of the 2D array.
    #arr.shape[1] // bin_size: The integer division calculates how many full blocks of bin_size can fit
    #along the width. This will be the new width of the binned array.
    new_shape = (arr.shape[0] // bin_size, bin_size, arr.shape[1] // bin_size, bin_size)

    #Logic: This line creates a trimmed version of the input array arr.
    #It slices the original arr#new_shape[0]*bin_size: This is the largest multiple of bin_size
    #Slicing operation. It's essentially selecting a sub-matrix from the original matrix.
    #For example, arr[:10, :20] selects rows 0-9 and columns 0-19. This ensures that the
    #array can be perfectly divided into bin_size x bin_size blocks
    #without encountering partial blocks at the edges.
    trimmed = arr[:new_shape[0]*bin_size, :new_shape[2]*bin_size]


    return trimmed.reshape(new_shape).mean(axis=(1, 3)) #does by average
    #trimmed.reshape(new_shape): Reshapes the trimmed 2D array into a 4D array using the new_shape calculated earlier.
    #.mean(axis=(1, 3)): Calculates the mean (average) along the specified ax
    #EX: (100, 100) and bin_size is 2, new_shape would be (50, 2, 50, 2).
    #Then the .mean part averages each bin for it average value along axis 1 and 3 (would by 2,2 in this example)



def sdt_to_tiff(input_filename, output_basename, output_selection, bin_size=2):
    """
    Load an SDT file, swap its axes to (T, X, Y), and save TIFF files.

    Args:
        input_filename (str): Path to the input SDT file.
        output_basename (str): The base name for the output TIFF file(s).
        output_selection (int): Determines which file(s) to save.
                                0: Unbinned only.
                                1: Binned only.
                                2: Both unbinned and binned.
        bin_size (int, optional): The factor for spatial binning. Defaults to 2.
                                  A bin_size of 1 effectively disables binning.
    Returns:
        tuple: A tuple containing the unbinned and binned data arrays
               (unbinned_data, binned_data). If binning is not performed,
               binned_data will be None. If an error occurs (e.g., empty file),
               it returns two 1x1x1 zero arrays.
    Raises:
        ValueError: If output_selection is not 0, 1, or 2.
    """
    if output_selection not in [0, 1, 2]:
        raise ValueError("output_selection must be 0 (unbinned), 1 (binned), or 2 (both).")
    
    # Check file size for SDT files
    if os.path.getsize(input_filename) < 5 * 1024 * 1024: # 5 MB in bytes
        print(f"WARNING: SDT file {input_filename} is less than 5MB. Skipping and returning empty array.")
        return np.zeros((341, 341, 341), dtype=np.uint16), np.zeros((341, 341, 341), dtype=np.uint16)

    print(f"Loading data from {input_filename}...")
    sdt = SdtFile(input_filename)

    # Check if the SDT file contains valid 3D data. Sometimes we save empty files not sure why.
    if not sdt.data:
        print(f"WARNING: SDT file contains no data blocks: {input_filename}. Returning empty array.")
        return np.zeros((1, 1, 1), dtype=np.uint16), np.zeros((1, 1, 1), dtype=np.uint16)
    elif sdt.data[0].ndim != 3:
        print(f"WARNING: Expected 3D data array in SDT file, but found array with shape {sdt.data[0].shape} in {input_filename}. Returning empty array.")
        return np.zeros((1, 1, 1), dtype=np.uint16), np.zeros((1, 1, 1), dtype=np.uint16)

    # Original data shape is (time, height, width), swap to (time, x, y)
    # This raw data is always needed, regardless of output.
    unbinned_data = np.swapaxes(sdt.data[0], 0, 2)  # New shape: (T, X, Y)
    binned_data = None

    # This block runs if the user wants the unbinned file (0) or both files (2).
    if output_selection in [0, 2]:
        unbinned_output_filename = f"{output_basename}_unbinned.tiff"
        print(f"Saving unbinned data to {unbinned_output_filename}...")
        tiff.imwrite(unbinned_output_filename, unbinned_data, photometric='minisblack', compression='zlib')

    # This block runs if the user wants the binned file (1) or both files (2).
    if output_selection in [1, 2]:
        if bin_size > 1:
            print(f"Applying {bin_size}x{bin_size} binning across spatial dimensions (X, Y)...")
            # Bin each time frame individually and stack them back together
            binned_frames = [bin_array_2d(unbinned_data[t], bin_size) for t in range(unbinned_data.shape[0])]
            binned_data = np.stack(binned_frames, axis=0)  # New shape: (T, X//bin, Y//bin)
        else:
            print("Bin size is 1 — using unbinned data for 'binned' output.")
            binned_data = unbinned_data

        
        binned_output_filename = f"{output_basename}_binned.tiff"
        print(f"Saving binned data to {binned_output_filename}...")
        tiff.imwrite(binned_output_filename, binned_data.astype(np.float32), photometric='minisblack', compression='zlib')

    print("Done.")
    return unbinned_data, binned_data
