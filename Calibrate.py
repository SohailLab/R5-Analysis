import numpy as np
import tifffile as tiff
import os

def get_calibration_parameters(filename, bin_width=0.0122, freq=80, harmonic=1, tau_ref = 2.5):
    """
    Opens a tiff file, calculates its FFT, and then calibrates data using all pixels.
    """
    im = tiff.imread(filename)
    image = np.moveaxis(im, 0, 2)
    bins = image.shape[2]
    t_arr = np.linspace(bin_width / 2, bin_width * (bins - 1 / 2), bins)

    integral = np.sum(image, axis=2).astype(float)
    integral[integral == 0] = 1e-9

    g = np.sum(image[:, ...] * np.cos(2 * np.pi * freq / 1000 * harmonic * t_arr[:]), axis=2) / integral
    s = np.sum(image[:, ...] * np.sin(2 * np.pi * freq / 1000 * harmonic * t_arr[:]), axis=2) / integral

    coor_g = np.mean(g)
    coor_s = np.mean(s)
    print(f"Centroid of uncalibrated data (using all data): G={coor_g:.4f}, S={coor_s:.4f}")

    freq = float(freq)
    tau_ref = float(tau_ref)
    g_ideal = 1 / (1 + np.power(2 * np.pi * freq / 1000 * tau_ref, 2))
    s_ideal = 2 * np.pi * freq / 1000 * tau_ref / (1 + np.power(2 * np.pi * freq / 1000 * tau_ref, 2))

    phi_ideal = np.arctan2(s_ideal, g_ideal)
    m_ideal = np.sqrt(s_ideal ** 2 + g_ideal ** 2)

    phi_given = np.arctan2(coor_s, coor_g)
    m_given = np.sqrt(coor_g ** 2 + coor_s ** 2)

    theta = phi_ideal - phi_given
    m = m_ideal / m_given

    g_calibrated_check = g * m * np.cos(theta) - s * m * np.sin(theta)
    s_calibrated_check = g * m * np.sin(theta) + s * m * np.cos(theta)
    mean_g_calibrated = np.mean(g_calibrated_check)
    mean_s_calibrated = np.mean(s_calibrated_check)

    tolerance = 1e-4
    if np.isclose(mean_g_calibrated**2 + mean_s_calibrated**2, mean_g_calibrated, atol=tolerance):
        print(f"Mean of calibrated points ({mean_g_calibrated:.4f}, {mean_s_calibrated:.4f}) is on the universal circle.")
        if mean_g_calibrated > 0:
             tau_estimated = (mean_s_calibrated / mean_g_calibrated) / (2 * np.pi * freq / 1000)
             print(f"Estimated lifetime at this point: {tau_estimated:.4f} ns")
        else:
            print("Cannot estimate lifetime as mean G is zero or negative.")
    else:
        print(f"Mean of calibrated points ({mean_g_calibrated:.4f}, {mean_s_calibrated:.4f}) is NOT on the universal circle.")

    return theta, m


def apply_calibration_and_get_phasor(filename, output_basename, theta, m, bin_width=0.0122, freq=80, harmonic=1, min_threshold=0.1, max_threshold=1.0):
    """
    Opens a tiff file, applies calibration, and saves two versions of the G/S TIFFs:
    1. Signal: Based on the primary intensity range (min_threshold to max_threshold).
    2. Background: Based on a dimmer intensity range (0.1*min_threshold to min_threshold).
    """
    im = tiff.imread(filename)
    # The intensity file is already normalized, so its max is 1.0
    # If it's a 3D stack, sum it to get the 2D intensity map for masking.
    intensity_map = np.sum(im, axis=0) if im.ndim == 3 else im

    # The raw lifetime decay data is needed for FFT, which is the input `filename`
    decay_data = tiff.imread(filename)
    decay_data_image = np.moveaxis(decay_data, 0, 2)
    bins = decay_data_image.shape[2]
    t_arr = np.linspace(bin_width / 2, bin_width * (bins - 1 / 2), bins)

    integral = np.sum(decay_data_image, axis=2).astype(float)
    integral[integral == 0] = 1e-9

    g_raw = np.sum(decay_data_image[:, ...] * np.cos(2 * np.pi * freq / 1000 * harmonic * t_arr[:]), axis=2) / integral
    s_raw = np.sum(decay_data_image[:, ...] * np.sin(2 * np.pi * freq / 1000 * harmonic * t_arr[:]), axis=2) / integral

    # --- CALIBRATION APPLICATION (on the full, unthresholded data) ---
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    g_raw_flat = g_raw.flatten()
    s_raw_flat = s_raw.flatten()
    raw_phasor_images = np.vstack((g_raw_flat, s_raw_flat))
    calibrated_image_points = R.dot(raw_phasor_images) * m
    g_calibrated_full = calibrated_image_points[0, :].reshape(g_raw.shape)
    s_calibrated_full = calibrated_image_points[1, :].reshape(s_raw.shape)
    print(f"Applied calibration to full G and S images for {os.path.basename(filename)}...")

    # --- 1. MASK AND SAVE "SIGNAL" TIFFS ---
    signal_mask = (intensity_map >= min_threshold) & (intensity_map <= max_threshold)
    g_signal = g_calibrated_full.copy(); g_signal[~signal_mask] = 0
    s_signal = s_calibrated_full.copy(); s_signal[~signal_mask] = 0

    g_signal_filename = f"{output_basename}_G_signal.tiff"
    s_signal_filename = f"{output_basename}_S_signal.tiff"
    print(f"Saving 'Signal' G/S images (intensity {min_threshold}-{max_threshold})...")
    # AB 1/26/26 Commented out the the lines that that save the tiffs 
    tiff.imwrite(g_signal_filename, g_signal.astype(np.float32))
    tiff.imwrite(s_signal_filename, s_signal.astype(np.float32))

    # --- 2. MASK AND SAVE "BACKGROUND" TIFFS ---
    bg_min = 0.1 * min_threshold
    bg_max = min_threshold
    background_mask = (intensity_map >= bg_min) & (intensity_map < bg_max)
    g_background = g_calibrated_full.copy(); g_background[~background_mask] = 0
    s_background = s_calibrated_full.copy(); s_background[~background_mask] = 0

    g_bg_filename = f"{output_basename}_G_background.tiff"
    s_bg_filename = f"{output_basename}_S_background.tiff"
    print(f"Saving 'Background' G/S images (intensity {bg_min:.3f}-{bg_max:.3f})...")
    tiff.imwrite(g_bg_filename, g_background.astype(np.float32))
    tiff.imwrite(s_bg_filename, s_background.astype(np.float32))

    # --- RETURN THE PRIMARY (SIGNAL) DATA for downstream use ---
    g_filtered_1d = g_calibrated_full[signal_mask].flatten()
    s_filtered_1d = s_calibrated_full[signal_mask].flatten()
    integral_filtered_1d = intensity_map[signal_mask].flatten()

    return g_signal, s_signal, g_filtered_1d, s_filtered_1d, integral_filtered_1d
