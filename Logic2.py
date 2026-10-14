import numpy as np
import tifffile as tiff
import os
import csv
# import matplotlib.pyplot as plt # Used in older version
from sklearn.mixture import GaussianMixture
import pandas as pd
import Calibrate as Cal
import FULLGMM2 as GMM
import ConvertandBin as CB
# import PhasorPlotter as Plotter
#import PDFReport 

def run_master_workflow(**params):
    os.makedirs(params['output_folder'], exist_ok=True)

    if params['analysis_mode'] == 'fit': # Find fit
        run_gmm_fitting_workflow(**params)
    elif params['analysis_mode'] == 'apply': # Fit from google sheet
        run_gmm_application_workflow(**params)
    else:
        raise ValueError(f"Unknown analysis mode: {params['analysis_mode']}")

def get_calibration_parameters(params): 
    """Determines calibration parameters either manually or from a file."""
    if params["calibrate_manually"]:
        print("--- Using Manual Calibration Parameters ---") # Set Phi and M values 
        phi_rad = np.deg2rad(params["manual_phi"])
        m_val = params["manual_m"]
        print(f"Using Manual Calibration: phi={phi_rad:.4f} rad, m={m_val:.4f}")
    else:
        print("--- Calculating Calibration Parameters ---") # Calculates Phi and M from FULLGMM2.py 
        phi_rad, m_val = Cal.get_calibration_parameters(
            filename=params["calibration_filepath"],
            bin_width=params["bin_width_ns"],
            freq=params["freq_mhz"],
            harmonic=params["harmonic"],
            tau_ref=params["tau_ref"]
        )
    return phi_rad, m_val

def _preprocess_file(filepath, base_name, phi_rad, m_val, params): # This applies a bin_size X bin_size square bin to the data ie 3X3 binning cause a 1024X1024 -> 341X341
    """Gets tiff files, bins, and applies calibration."""
    unbinned_tiff = tiff.imread(filepath) 
    if params['bin_size'] > 1:
        print(f"Applying {params['bin_size']}x{params['bin_size']} binning...")
        intensity_data = np.stack([CB.bin_array_2d(frame, params['bin_size']) for frame in unbinned_tiff], axis=0)
    else:
        intensity_data = unbinned_tiff
    
    # Normalize and save temporary calibrated G/S files
    summed_intensity = np.sum(intensity_data, axis=0) 
    if summed_intensity.max() == 0: # This will come up if a TIFF file is empty (does happen on our microscope)
        print(f"Skipping {base_name} due to zero intensity.")
        return None, None, None, None
        
    normalized_intensity = intensity_data.astype(np.float32) / summed_intensity.max()
    temp_norm_path = os.path.join(params['output_folder'], f"{base_name}_binned_normalized.tiff")
    tiff.imwrite(temp_norm_path, normalized_intensity)
    calibrated_basename = os.path.join(params['output_folder'], f"{base_name}_calibrated")
    
    # Note: apply_calibration_and_get_phasor returns 5 values we dont care about first 2 for this usecase 
    # Returns: g_signal, s_signal, g_filtered_1d, s_filtered_1d, integral_filtered_1d 
    # Dev note prolly cut the first two outputs dont need them anymore 
    _, _, g_filt, s_filt, i_filt = Cal.apply_calibration_and_get_phasor(
        filename=temp_norm_path,
        output_basename=calibrated_basename,
        theta=phi_rad,
        m=m_val,
        bin_width=params["bin_width_ns"],
        freq=params["freq_mhz"],
        harmonic=params["harmonic"],
        min_threshold=params["min_threshold"],
        max_threshold=params["max_threshold"]
    ) 
    
    return calibrated_basename, g_filt, s_filt, i_filt

def cleanup_intermediate_files(base_name, output_folder): # Added 1/28/26 to autodelte tiffs if needed. NOTE THIS WILL DELETE ALL TIFFS so be carful
    """Deletes G/S Signal and Background TIFFs to save space."""
    suffixes = ["_G_signal.tiff", "_S_signal.tiff", "_G_background.tiff", "_S_background.tiff"]
    for suffix in suffixes:
        path = os.path.join(output_folder, f"{base_name}_calibrated{suffix}")
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"Deleted intermediate file: {os.path.basename(path)}")
        except OSError as e:
            print(f"Warning: Could not delete {os.path.basename(path)}: {e}")

def run_gmm_fitting_workflow(**params): # GMM fitting work flow 
    """Aggregates phasor data from all files and fits a GMM."""
    phi_rad, m_val = get_calibration_parameters(params)
    print(f"\nCalibration determined: Phi={phi_rad:.4f} radians, M={m_val:.4f}\n")
    
    all_phasor_points = []
    all_intensities = [] 
    num_files = len(params["filepaths"])
    print("--- Processing Files to Aggregate Phasor Data ---")

    for i, fpath in enumerate(params["filepaths"]):
        print(f"Processing file {i+1}/{num_files}: {os.path.basename(fpath)}")
        base_name = os.path.splitext(os.path.basename(fpath))[0]
        
        _, g_filtered, s_filtered, i_filtered = _preprocess_file(fpath, base_name, phi_rad, m_val, params) # We dont need the basename (first return)

        if g_filtered is not None and g_filtered.size > 0:
            print(f"  ...Collected {g_filtered.size} pixels for GMM fitting.")
            phasor_coords = np.stack((g_filtered, s_filtered), axis=-1)
            # Appends data 
            all_phasor_points.append(phasor_coords)
            all_intensities.append(i_filtered) 
        else:
            print("  ...No pixels passed the filter in this file.")

    if not all_phasor_points:
        raise RuntimeError("No valid phasor data could be extracted from any input files.") # See https://docs.python.org/3/library/exceptions.html#RuntimeError

    phasor_data = np.vstack(all_phasor_points)
    intensity_data = np.concatenate(all_intensities) # had some issues with data format and this was an AI suggestion (should check if Vstack works)
    
    print(f"\nTotal phasor points for GMM: {phasor_data.shape[0]}")
    if phasor_data.shape[0] < params['gmm_components']:
         raise RuntimeError(f"Not enough data points ({phasor_data.shape[0]}) for GMM fitting.")

    model_type = params.get("model_type", "BIC+SD")

    if model_type == "BIC+SD": # Modfied this on 1/28/26 to try and see if
            print(f"Searching for optimal GMM components (up to {params['gmm_components']}) using BIC Elbow method...")

            best_n, _, _, _ = GMM.find_optimal_gmm_by_elbow(
                phasor_data[:, 0],
                phasor_data[:, 1],
                min_mag=0.2, # Data already pre filtered? (Hopefully)
                max_components=params['gmm_components']
            )
            print(f"Optimal number of components found via BIC Elbow: {best_n}")
            if best_n is None:
                print("Elbow detection failed. Defaulting to 1 component.")
                best_n = 1
                gmm_model = GaussianMixture(n_components=best_n, random_state=42).fit(phasor_data)
            else:
                gmm_model = GaussianMixture(n_components=best_n, random_state=42).fit(phasor_data)
                while GMM.check_ellipse_overlap(gmm_model): # Recall this returns true for overlap
                    print(f"1st SD overlapped! Reduicng number of componets from {best_n} to {best_n - 1}")
                    best_n = best_n -1
                    gmm_model = GaussianMixture(n_components=best_n, random_state=42).fit(phasor_data)
                print(f"Optimal number of components found via BIC Elbow: {best_n}")
            gmm_model = GaussianMixture(n_components=best_n, random_state=42).fit(phasor_data)
            output_filename = "gmm_results_optimal_fit_BIC.csv"
    elif model_type == "AIC": # similar to BIC but with diffrent weights: 
        # TLDR its: 
        # BIC = 2ln(L) + k*ln(N)
        # AIC = -2ln(L) + 2k
        # L is the max likelihood of model, k is the est paramaters, and N is the number of data points 
        # BIC has a harsher penalty for model complexity but wanted to add both...
        print(f"Searching for optimal GMM components (up to {params['gmm_components']}) using AIC Elbow method...")
        _, best_n, _, _ = GMM.find_optimal_gmm_by_elbow(
            phasor_data[:, 0],
            phasor_data[:, 1],
            min_mag=0.2,
            max_components=params['gmm_components']
        )
        if best_n is None:
            print("Elbow detection failed. Defaulting to 1 component.")
            best_n = 1
        print(f"Optimal number of components found via AIC Elbow: {best_n}")
        gmm_model = GaussianMixture(n_components=best_n, random_state=42).fit(phasor_data)
        output_filename = "gmm_results_optimal_fit_AIC.csv"
    else: # "Manual" - See scikit documentation for how GaussianMixture works. Note 42 was chosen cause hitchhikers guide to the galaxy (could pick any number)
        num_components = params["gmm_components"]
        print(f"Fitting GMM with specified {num_components} components...")
        gmm_model = GaussianMixture(n_components=num_components, random_state=42).fit(phasor_data)
        output_filename = "gmm_results_manual_fit.csv"

    print("GMM fitting complete. Saving results...")
    output_path = os.path.join(params["output_folder"], output_filename)
    with open(output_path, 'w', newline='') as csvfile: # Saves results as CSV
        writer = csv.writer(csvfile)
        writer.writerow(['Component', 'Weight', 'Mean_G', 'Mean_S', 'Covariance_GG', 'Covariance_GS', 'Covariance_SG', 'Covariance_SS'])
        for i in range(gmm_model.n_components):
            mean, cov = gmm_model.means_[i], gmm_model.covariances_[i]
            writer.writerow([i + 1, gmm_model.weights_[i], mean[0], mean[1], cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]])
    print(f"GMM results saved to: {output_path}")

    print("Generating summary phasor plot of the GMM fit...")
    GMM.plot_gmm_phasor_fit(
        gmm=gmm_model,
        X_filtered=phasor_data,
        intensities=intensity_data,
        title="Aggregated GMM Fit",
        output_path=params['output_folder'],
        analysis_name="Aggregated_GMM_Fit_Result",
        frequency_mhz=params['freq_mhz'],
        harmonic=params['harmonic']
    )

    # Added 07/08/26 the D_KL and H(X,Y) metrics to compare gaussian components if we have 2 or more 
    #if gmm_model.n_components >= 2:
    #    print("\n--- Computing Cluster Comparison Metrics (KL Divergence & Joint Entropy) ---")
    #    comp_labels = [str(i + 1) for i in range(gmm_model.n_components)]
    #    GMM.compute_cluster_comparison_metrics(
    #        gmm_model=gmm_model,
    #        output_path=params['output_folder'],
    #        analysis_name="Aggregated_GMM_Fit",
    #        comp_labels=comp_labels
    #    )


def run_gmm_application_workflow(**params): 
    """Applies a pre-defined GMM model to each file to generate overlays."""
    print("--- Building GMM model from GUI parameters ---")
    try:
        n_components = len(params['manual_gmm_params']['weights'])
        gmm_model = GaussianMixture(n_components=n_components, covariance_type='full')
        gmm_model.weights_ = params['manual_gmm_params']['weights']
        gmm_model.means_ = params['manual_gmm_params']['means']
        gmm_model.covariances_ = params['manual_gmm_params']['covariances']
        gmm_model.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm_model.covariances_))
        print(f"Successfully built a {n_components}-component GMM model.")
    except Exception as e:
        raise ValueError(f"Failed to build GMM model from parameters.\nDetails: {e}")


   # comp_labels = [str(i + 1) for i in range(gmm_model.n_components)] #
   # kl_matrix, entropy_matrix = None, None 
   # if gmm_model.n_components >= 2:
    #    print("\n--- Computing Cluster Comparison Metrics (KL Divergence & Joint Entropy) ---")
    #    kl_matrix, entropy_matrix = GMM.compute_cluster_comparison_metrics(
     #       gmm_model=gmm_model,
     #       output_path=params['output_folder'],
     #       analysis_name="Applied_GMM_Model",
      #      comp_labels=comp_labels
       # ) # we save these for downstream use in pdf report.... 

    # === AB 01/26/26 adding the net image ===
    all_g_filtered = []
    all_s_filtered = []
    all_i_filtered = []
    all_agg_dfs = []
    per_file_images = [] # for the PDF report 

    phi_rad, m_val = get_calibration_parameters(params) 
    print(f"\nCalibration determined: Phi={phi_rad:.4f} radians, M={m_val:.4f}\n")
    
    num_files = len(params["filepaths"])
    for i, fpath in enumerate(params["filepaths"]):
        print(f"--- Applying GMM to file {i+1}/{num_files}: {os.path.basename(fpath)} ---")
        base_name = os.path.splitext(os.path.basename(fpath))[0]
        
        calibrated_basename, g_filt, s_filt, i_filt = _preprocess_file(fpath, base_name, phi_rad, m_val, params)
        
        if calibrated_basename is None:
            continue

        # === AB 01/26/26 adding the net image ===
        if g_filt is not None and g_filt.size > 0:
            all_g_filtered.append(g_filt)
            all_s_filtered.append(s_filt)
            all_i_filtered.append(i_filt)
        # === End of Mod ===

        g_signal_path = f"{calibrated_basename}_G_signal.tiff"
        s_signal_path = f"{calibrated_basename}_S_signal.tiff"
        temp_norm_path = os.path.join(params['output_folder'], f"{base_name}_binned_normalized.tiff")
        
        norm_intensity_data = tiff.imread(temp_norm_path)
        I_summed = np.sum(norm_intensity_data, axis=0) if norm_intensity_data.ndim == 3 else norm_intensity_data
        
        G_image = tiff.imread(g_signal_path)
        S_image = tiff.imread(s_signal_path)
        
        print("\nGenerating GMM Overlay Plots...")
        GMM.plot_gmm_hard_cluster_overlay(
            gmm_model=gmm_model, G_image=G_image, S_image=S_image, intensity_img=I_summed,
            output_path=params['output_folder'], analysis_name=f"{base_name}_HardCluster",
            min_thresh=params['min_threshold'], max_thresh=params['max_threshold']
        )
        GMM.plot_gmm_aggregate_overlay(
            gmm_model=gmm_model, G_image=G_image, S_image=S_image, intensity_img=I_summed,
            output_path=params['output_folder'], analysis_name=f"{base_name}_Aggregate",
            min_thresh=params['min_threshold'], max_thresh=params['max_threshold']
        )
        X_filtered_data = np.vstack((g_filt, s_filt)).T
        # --- Save pixel-level CSV ---
        pixel_df = GMM.extract_pixel_data(
            gmm_model=gmm_model, G_image=G_image, S_image=S_image,
            intensity_img=I_summed, file_name=base_name,
            min_thresh=params['min_threshold'], max_thresh=params['max_threshold']
        )
        if not pixel_df.empty:
            pixel_csv_path = os.path.join(params['output_folder'], f"{base_name}_pixel_data.csv")
            pixel_df.to_csv(pixel_csv_path, index=False)
            print(f"Pixel-level data saved to: {pixel_csv_path}")

        # --- Collect aggregate-level data ---
        agg_df = GMM.extract_aggregate_data(
            gmm_model=gmm_model, G_image=G_image, S_image=S_image,
            intensity_img=I_summed, file_name=base_name,
            min_thresh=params['min_threshold'], max_thresh=params['max_threshold']
        )
        if not agg_df.empty:
            all_agg_dfs.append(agg_df)

        GMM.plot_gmm_phasor_fit(
            gmm=gmm_model,
            X_filtered=X_filtered_data,
            intensities=i_filt,
            title=f"GMM Phasor Fit for {base_name}",
            output_path=params['output_folder'],
            analysis_name=f"{base_name}_PhasorFit",
            frequency_mhz=params['freq_mhz'],
            harmonic=params['harmonic']
        )

        # Saves the per image filepath for the pdf report 
        per_file_images.append({
            "base_name": base_name,
            "intensity_phasor": os.path.join(params['output_folder'], f"gmm_phasor_fit_{base_name}_PhasorFit_Intensity.png"),
            "uncertainty_phasor": os.path.join(params['output_folder'], f"gmm_phasor_fit_{base_name}_PhasorFit_StandardGMM.png"),
            "hard_cluster": os.path.join(params['output_folder'], f"gmm_overlay_hard_cluster_{base_name}_HardCluster.png"),
        })
        print(f"Finished processing {base_name}")

    # === AB 01/26/26 adding the net image ===
    net_plot_path = None  
    if all_g_filtered:
        print("\n--- Generating Net Aggregated GMM Fit (All Files) ---")
        try:
            # Stack all collected arrays
            total_g = np.concatenate(all_g_filtered)
            total_s = np.concatenate(all_s_filtered)
            total_i = np.concatenate(all_i_filtered)

            # Create the data structure expected by the plotter (columns: G, S)
            X_total_filtered = np.vstack((total_g, total_s)).T

            GMM.plot_gmm_phasor_fit(
                gmm=gmm_model,
                X_filtered=X_total_filtered,
                intensities=total_i,
                title="Net Aggregated GMM Fit (All Files)",
                output_path=params['output_folder'],
                analysis_name="Net_Aggregated_GMM_Fit",
                frequency_mhz=params['freq_mhz'],
                harmonic=params['harmonic']
            )
            net_plot_path = os.path.join(params['output_folder'], "gmm_phasor_fit_Net_Aggregated_GMM_Fit_Intensity.png")
            print("Net aggregated plot saved successfully.")
        except Exception as e:
            print(f"Error generating net aggregated plot: {e}")
    else:
        print("No valid data collected for net aggregation.")
    # === End of Mod ===

    # --- Save combined aggregate CSV and clean up intermediates ---
    combined_agg_df = None  
    if all_agg_dfs:
        combined_agg_df = pd.concat(all_agg_dfs, ignore_index=True)
        combined_agg_path = os.path.join(params['output_folder'], "combined_aggregate_data.csv")
        combined_agg_df.to_csv(combined_agg_path, index=False)
        print(f"Combined aggregate data ({len(combined_agg_df)} aggregates across {len(all_agg_dfs)} file(s)) saved to: {combined_agg_path}")
    else:
        print("No aggregate data collected — combined CSV not saved.")

    
    if params.get("delete_intermediates", False):
        for fpath in params["filepaths"]:
            cleanup_intermediate_files(os.path.splitext(os.path.basename(fpath))[0], params['output_folder'])