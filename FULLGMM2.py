import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm, ListedColormap, Normalize, to_rgba
from matplotlib.patches import Ellipse, Patch
from scipy import stats
import itertools
import os
import csv
import json
from pathlib import Path
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.stats import multivariate_normal
from skimage import measure
import matplotlib.cm as cm
import pandas as pd

CONFIG_JSON_PATH = Path(__file__).parent / "config.json" # this is the file that the WEBGUI will modify for global parameters (at some point)

# Define color palette: Green, Red, Blue, Fuchsia, Purple (Left to Right)
USER_PALETTE = ['#009e73','#ff0000','#0000ff', '#ff00ff', '#CC79A7']

#Definte Global Phasor plot parameters
G_AXIS_RANGE = [0, 1.0]
S_AXIS_RANGE = [0, 0.65]
SHOW_GRID = False 
SHOW_LIFETIME_MARKERS = True  # Toggle for lifetime markers
LIFETIME_MARKERS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]  # ns;
try:
    with open(CONFIG_JSON_PATH, "r") as f:
        config = json.load(f)
        FREQ = float(config.get("Laser_rep_rate_MHz", 80.0))
except Exception as e:
    print(f"Error occured in loading JSON: {e}")
    print("Setting FREQ=80.0 MHz by default")
    FREQ = 80.0 

try:
    with open(CONFIG_JSON_PATH, "r") as f:
        config = json.load(f)
        HARMONIC = float(config.get("Harmonic", 1.0))
except Exception as e:
    print(f"Error occured in loading JSON: {e}")
    print("Setting Harmonic=1.0 by default")
    HARMONIC = 1.0
USER_PALETTE_MAX_COMPONENTS = len(USER_PALETTE) # To ensure we have enough colors 

def get_sorted_color_mapping(gmm_model):
    """
    Returns:
        component_color_map: dict {component_index: hex_color}
        sorted_indices: array of component indices sorted by Mean G (low to high)

    Uses USER_PALETTE (applied to all overlay plots, since they all call this function)
    for up to USER_PALETTE_MAX_COMPONENTS components. Beyond that, reverts to
    matplotlib's default 'tab20' colormap, which scales to more components without the
    stark repeated-hue collisions a cycled 4-color palette would produce.
    """
    # Sort components by Mean G (dim 0)
    sorted_indices = np.argsort(gmm_model.means_[:, 0])
    n_components = len(sorted_indices)

    component_color_map = {}
    if n_components > USER_PALETTE_MAX_COMPONENTS:
        default_colors = cm.get_cmap('tab20')(np.linspace(0, 1, n_components))
        for rank, comp_idx in enumerate(sorted_indices):
            component_color_map[comp_idx] = default_colors[rank]
    else:
        palette_length = len(USER_PALETTE)
        for rank, comp_idx in enumerate(sorted_indices):
            # Use modulo if components > 4, though unlikely
            color = USER_PALETTE[rank % palette_length]
            component_color_map[comp_idx] = color

    return component_color_map, sorted_indices

def plot_gmm_aggregate_overlay(gmm_model, G_image, S_image, intensity_img, output_path, analysis_name, min_thresh, max_thresh):
    """
    Creates an aggregate-based GMM overlay. Each entire aggregate is assigned to a single
    Gaussian component. Colors are ordered by G-position (Left=Blue -> Right=Purple).
    """
    if not gmm_model:
        print(f"No GMM model provided for aggregate overlay: {analysis_name}")
        return
    
    # --- Sorting Logic ---
    comp_color_map, sorted_indices = get_sorted_color_mapping(gmm_model)

    # Apply thresholding to intensity image to identify regions of interest
    intensity_mask = (intensity_img >= min_thresh) & (intensity_img <= max_thresh)
    binary_mask = intensity_mask.astype(np.uint8)
    
    # Label the connected components (aggregates)
    labeled_img = measure.label(binary_mask)
    regions = measure.regionprops(labeled_img)
    
    print(f"Found {len(regions)} aggregates for GMM classification.")
    
    # Create output image for the overlay
    rgba_overlay = np.zeros((G_image.shape[0], G_image.shape[1], 4))
    
    # Process each aggregate
    for region in regions:
        # Skip very small regions (likely noise)
        if region.area < 10:
            continue
            
        # Create mask for this aggregate
        aggregate_mask = (labeled_img == region.label)
        
        # Get G and S values for this aggregate
        G_values = G_image[aggregate_mask]
        S_values = S_image[aggregate_mask]
        
        # Combine into feature array
        X_aggregate = np.vstack((G_values, S_values)).T
        
        # Ensure there are points to classify
        if X_aggregate.shape[0] == 0:
            continue

        # Calculate the total log-likelihood of all points in the aggregate under each component.
        total_log_likelihoods = []
        for i in range(gmm_model.n_components):
            mean_i = gmm_model.means_[i]
            cov_i = gmm_model.covariances_[i]
            log_probs = multivariate_normal.logpdf(X_aggregate, mean=mean_i, cov=cov_i)
            total_log_likelihood = np.sum(log_probs)
            total_log_likelihoods.append(total_log_likelihood)
            
        most_likely_component = np.argmax(total_log_likelihoods)
        
        # Assign the entire aggregate to this component using the SORTED color map
        color_hex = comp_color_map[most_likely_component]
        color_rgb = to_rgba(color_hex)[:3] # Convert hex to RGB 
        
        rgba_overlay[aggregate_mask, :3] = color_rgb
        rgba_overlay[aggregate_mask, 3] = 1.0
        
        # Calculate mean G/S for printing purposes
        mean_G = np.mean(G_values)
        mean_S = np.mean(S_values)
        
        # Print information about the classification
        print(f"Aggregate {region.label}: size={region.area}, " +
              f"mean G/S=({mean_G:.3f}, {mean_S:.3f}), assigned to component {most_likely_component+1}")
    
    # Create the figure and plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(intensity_img, cmap='gray')
    ax.imshow(rgba_overlay)
    
    # Add legend (Sorted visually: Blue -> Orange -> Green -> Purple)
    legend_elements = []
    for rank, comp_idx in enumerate(sorted_indices):
        color = comp_color_map[comp_idx]
        legend_elements.append(Patch(facecolor=color, label=f'Component {comp_idx+1}'))

    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title(f"GMM Aggregate-Based Overlay ({analysis_name})")
    ax.axis('off')
    
    # Add scale bar
    ax.add_artist(ScaleBar(1.0, "um", color='w', location='lower right', frameon=False))
    
    # Save the figure
    output_filename = os.path.join(output_path, f"gmm_overlay_aggregate_based_{analysis_name}.pdf")
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
    output_filename_eps = os.path.join(output_path, f"gmm_overlay_aggregate_based_{analysis_name}.eps")
    plt.savefig(output_filename_eps, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"GMM aggregate-based overlay saved to {output_filename}")
    plt.close(fig)


def plot_gmm_hard_cluster_overlay(gmm_model, G_image, S_image, intensity_img, output_path, analysis_name, min_thresh, max_thresh):
    """
    Creates a 'hard clustered' GMM overlay. Colors ordered by G-position.
    """
    if not gmm_model:
        print(f"No GMM model provided for hard cluster overlay: {analysis_name}")
        return

    comp_color_map, sorted_indices = get_sorted_color_mapping(gmm_model)

    X_raw = np.vstack((G_image.flatten(), S_image.flatten())).T
    initial_labels_flat = gmm_model.predict(X_raw)
    initial_probs_flat = gmm_model.predict_proba(X_raw).max(axis=1)
    
    initial_label_img = initial_labels_flat.reshape(G_image.shape)
    initial_prob_img = initial_probs_flat.reshape(G_image.shape)

    hard_cluster_label_img = initial_label_img.copy()
    low_confidence_mask = (initial_prob_img < 0.6)
    
    label_pad = np.pad(initial_label_img, pad_width=1, mode='edge')
    prob_pad = np.pad(initial_prob_img, pad_width=1, mode='edge')

    print(f"Performing hard clustering on {np.sum(low_confidence_mask)} low-confidence pixels...")
    
    # Checks neighbors to see what their values are near 
    for r, c in np.argwhere(low_confidence_mask):
        neighbors_labels = label_pad[r:r+3, c:c+3].flatten()
        neighbors_probs = prob_pad[r:r+3, c:c+3].flatten()
        
        confident_neighbors_labels = neighbors_labels[neighbors_probs >= 0.6]
        
        if confident_neighbors_labels.size > 0:
            majority_label = stats.mode(confident_neighbors_labels, keepdims=False)[0]
            hard_cluster_label_img[r, c] = majority_label

    intensity_mask = (intensity_img >= min_thresh) & (intensity_img <= max_thresh)
    rgba_overlay = np.zeros((G_image.shape[0], G_image.shape[1], 4))

    for i in range(gmm_model.n_components):
        final_mask = (hard_cluster_label_img == i) & intensity_mask
        
        # Get color based on component index i
        color_hex = comp_color_map[i]
        color_rgb = to_rgba(color_hex)[:3]
        
        rgba_overlay[final_mask, :3] = color_rgb
        rgba_overlay[final_mask, 3] = 1.0

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(intensity_img, cmap='gray')
    ax.imshow(rgba_overlay)

    # Sorted Legend
    legend_elements = []
    for rank, comp_idx in enumerate(sorted_indices):
        color = comp_color_map[comp_idx]
        legend_elements.append(Patch(facecolor=color, label=f'Component {comp_idx+1}'))

    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title(f"GMM Hard Cluster Overlay ({analysis_name})")
    ax.axis('off')

    # Scale bar
    ax.add_artist(ScaleBar(1.0, "um", color='w', location='lower right', frameon=False))

    output_filename = os.path.join(output_path, f"gmm_overlay_hard_cluster_{analysis_name}.pdf")
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
    output_filename_eps = os.path.join(output_path, f"gmm_overlay_hard_cluster_{analysis_name}.eps")
    plt.savefig(output_filename_eps, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"GMM hard cluster overlay saved to {output_filename}")
    plt.close(fig)


def plot_gmm_overlay(gmm_model, G_image, S_image, intensity_img, output_path, analysis_name, min_thresh, max_thresh):
    """
    Creates and saves an image overlay of the GMM results within a specified intensity range.
    Colors ordered by G-position.
    """
    if not gmm_model:
        print(f"No GMM model provided for overlay plot: {analysis_name}")
        return

    # --- Sorting Logic ---
    comp_color_map, sorted_indices = get_sorted_color_mapping(gmm_model)

    X_raw = np.vstack((G_image.flatten(), S_image.flatten())).T
    labels_full = gmm_model.predict(X_raw)
    probs_full = gmm_model.predict_proba(X_raw)
    max_probs_full = np.max(probs_full, axis=1)

    label_img = labels_full.reshape(G_image.shape)
    prob_img = max_probs_full.reshape(G_image.shape)

    intensity_mask = (intensity_img >= min_thresh) & (intensity_img <= max_thresh)
    rgba_overlay = np.zeros((G_image.shape[0], G_image.shape[1], 4))

    for i in range(gmm_model.n_components):
        final_mask = (label_img == i) & intensity_mask
        
        # Get color based on component index i
        color_hex = comp_color_map[i]
        color_rgb = to_rgba(color_hex)[:3]
        
        rgba_overlay[final_mask, :3] = color_rgb
        rgba_overlay[final_mask, 3] = prob_img[final_mask]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(intensity_img, cmap='gray')
    ax.imshow(rgba_overlay)

    # Sorted Legend
    legend_elements = []
    for rank, comp_idx in enumerate(sorted_indices):
        color = comp_color_map[comp_idx]
        legend_elements.append(Patch(facecolor=color, label=f'Component {comp_idx+1}'))

    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title(f"GMM Component Overlay ({analysis_name})")
    ax.axis('off')

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)

    norm = Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap=cm.gray, norm=norm)
    sm.set_array([]) 
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label('Component Assignment Certainty', rotation=270, labelpad=20)

    output_filename = os.path.join(output_path, f"gmm_overlay_{analysis_name}.pdf")
    plt.savefig(output_filename,  bbox_inches='tight', pad_inches=0.1)
    output_filename_eps = os.path.join(output_path, f"gmm_overlay_{analysis_name}.eps")
    plt.savefig(output_filename_eps, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"GMM overlay saved to {output_filename}")
    plt.close(fig)


def save_gmm_results(gmm_model, filtered_data, output_path, analysis_name, source_filename, is_first_file=True):
    """
    Saves GMM results to a combined CSV file, appending if the file exists.
    """
    if not gmm_model:
        print(f"No GMM model to save for {analysis_name}.")
        return

    overall_centroid_g = np.mean(filtered_data[:, 0])
    overall_centroid_s = np.mean(filtered_data[:, 1])

    output_filename = os.path.join(output_path, f"gmm_results_{analysis_name}_combined.csv")
    mode = 'w' if is_first_file else 'a'

    try:
        with open(output_filename, mode, newline='') as csvfile:
            writer = csv.writer(csvfile)

            if is_first_file:
                writer.writerow(['Filename', 'Component', 'Mean_G', 'Mean_S', 'Weight', 'Covariance_GG', 'Covariance_GS', 'Covariance_SS'])

            writer.writerow([source_filename, 'Overall_Centroid', f'{overall_centroid_g:.6f}', f'{overall_centroid_s:.6f}', '', '', '', ''])

            for i in range(gmm_model.n_components):
                mean = gmm_model.means_[i]
                weight = gmm_model.weights_[i]
                cov = gmm_model.covariances_[i]
                writer.writerow([
                    source_filename, f'Component_{i+1}', f'{mean[0]:.6f}', f'{mean[1]:.6f}',
                    f'{weight:.6f}', f'{cov[0, 0]:.6f}', f'{cov[0, 1]:.6f}', f'{cov[1, 1]:.6f}'
                ])
        print(f"GMM results for '{source_filename}' saved to {output_filename}")

    except Exception as e:
        print(f"Error saving GMM results to CSV: {e}")

def save_empty_gmm_results(output_path, analysis_name, source_filename, is_first_file=True):
    """
    Saves placeholder data to a combined CSV file when GMM fails.
    """
    output_filename = os.path.join(output_path, f"gmm_results_{analysis_name}_combined.csv")
    mode = 'w' if is_first_file else 'a'
    print(f"No GMM model fit for {source_filename}. Saving placeholder file with zeros.")

    try:
        with open(output_filename, mode, newline='') as csvfile:
            writer = csv.writer(csvfile)
            if is_first_file:
                writer.writerow(['Filename', 'Component', 'Mean_G', 'Mean_S', 'Weight', 'Covariance_GG', 'Covariance_GS', 'Covariance_SS'])
            
            writer.writerow([source_filename, 'Overall_Centroid', '0.000000', '0.000000', '', '', '', ''])
            writer.writerow([source_filename, 'Component_1', '0.000000', '0.000000', '0.000000', '0.000000', '0.000000', '0.000000'])
        print(f"Placeholder results for '{source_filename}' saved to {output_filename}")
    except Exception as e:
        print(f"Error saving placeholder GMM results to CSV: {e}")

def Phasor_GMM(G_full, S_full, n_components, initial_means, min_mag=0.01):
    """
    Applies a Gaussian Mixture Model (GMM) to phasor plot data. 
    """
    X_raw = np.vstack((G_full.flatten(), S_full.flatten())).T
    magnitude = np.linalg.norm(X_raw, axis=1)
    mask = magnitude > min_mag
    X_filtered = X_raw[mask]
    original_indices_of_filtered_points = np.where(mask)[0]

    if X_filtered.shape[0] < n_components:
        print(f"Warning: Not enough data points ({X_filtered.shape[0]}) to fit GMM with {n_components} components.")
        return None, None, None, None

    gmm = GaussianMixture(n_components=n_components, means_init=initial_means, random_state=42, covariance_type='full')
    gmm.fit(X_filtered)

    labels_filtered = gmm.predict(X_filtered)
    probs_filtered = gmm.predict_proba(X_filtered)
    max_probs_filtered = np.max(probs_filtered, axis=1)

    final_labels = np.full(X_raw.shape[0], -1, dtype=int)
    final_labels[original_indices_of_filtered_points] = labels_filtered
    final_probabilities = np.zeros(X_raw.shape[0])
    final_probabilities[original_indices_of_filtered_points] = max_probs_filtered

    return gmm, final_labels, final_probabilities, X_filtered

def plot_phasor_density(g_tiff_path, s_tiff_path):
    """
    Loads G and S TIFF files and generates a 2D density phasor plot.
    """
    try:
        G = tiff.imread(g_tiff_path)
        S = tiff.imread(s_tiff_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    phasor_mask = (G > 0) & (S > 0)
    fig, ax = plt.subplots(figsize=(8, 8))
    phasor_plot = ax.hist2d(G[phasor_mask], S[phasor_mask], bins=150, cmap='jet', norm=LogNorm(), range=[G_AXIS_RANGE, S_AXIS_RANGE])
    cbar = fig.colorbar(phasor_plot[3], ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Density')
    if SHOW_LIFETIME_MARKERS:
                omega = 2 * np.pi * (FREQ / 1000) * HARMONIC 
                
                for tau in LIFETIME_MARKERS:
                    g_marker = 1 / (1 + (omega * tau)**2)
                    s_marker = (omega * tau) / (1 + (omega * tau)**2)
                    
                    ax.plot(g_marker, s_marker, marker='s', markersize=5, 
                                markerfacecolor='red', markeredgecolor='black', zorder=10)
                    
                    ax.text(g_marker + 0.015, s_marker + 0.015, f"{tau} ns", 
                                fontsize=8, color='black', fontweight='bold', zorder=10)

    ax.set_xlim(G_AXIS_RANGE); ax.set_ylim(S_AXIS_RANGE)
    ax.set_xlabel('G'); ax.set_ylabel('S')
    ax.set_aspect('equal'); ax.set_title("Phasor Plot with Density")
    
    if SHOW_GRID:
        ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout(); plt.show(); plt.close('all')

def check_ellipse_overlap(gmm, num_points=100):
    """
    Checks if any 1-SD ellipses of the GMM components overlap.
    """
    if gmm.n_components < 2: return False
    c = 1
    component_pairs = list(itertools.combinations(range(gmm.n_components), 2))

    for i, j in component_pairs:
        mu1, mu2 = gmm.means_[i], gmm.means_[j]
        Sigma1, Sigma2 = gmm.covariances_[i], gmm.covariances_[j]
        try:
            inv_Sigma1 = np.linalg.inv(Sigma1)
            inv_Sigma2 = np.linalg.inv(Sigma2)
            eigvals1, eigvecs1 = np.linalg.eigh(Sigma1)
            eigvals2, eigvecs2 = np.linalg.eigh(Sigma2)
            L1 = eigvecs1 @ np.diag(np.sqrt(eigvals1 * c))
            L2 = eigvecs2 @ np.diag(np.sqrt(eigvals2 * c))
        except np.linalg.LinAlgError:
            print(f"Warning: Could not process covariance matrix for components {i} and {j}. Skipping overlap check.")
            continue

        thetas = np.linspace(0, 2 * np.pi, num_points)
        for theta in thetas:
            u = np.array([np.cos(theta), np.sin(theta)])
            x1 = mu1 + L1 @ u
            if (x1 - mu2).T @ inv_Sigma2 @ (x1 - mu2) <= c: return True
            x2 = mu2 + L2 @ u
            if (x2 - mu1).T @ inv_Sigma1 @ (x2 - mu1) <= c: return True
    return False

def analyze_and_plot_optimal_gmm(g_tiff_path, s_tiff_path, min_mag=0.01, max_components=5, output_path=None, analysis_name=None, autoclose_plots=True):
    """
    Determines the optimal GMM components, fits the model, and plots the final result.
    """
    try:
        G_full = tiff.imread(g_tiff_path)
        S_full = tiff.imread(s_tiff_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None

    print(f"Analyzing for optimal components (up to {max_components})...")
    initial_k, _, _, _ = find_optimal_gmm_by_elbow(G_full, S_full, min_mag=min_mag, max_components=max_components)
    if initial_k is None:
        print("Could not determine optimal number of components.")
        return None
    print(f"BIC Elbow method initially suggests: {initial_k} components.")

    X_raw = np.vstack((G_full.flatten(), S_full.flatten())).T
    magnitude = np.linalg.norm(X_raw, axis=1)
    mask = magnitude > min_mag
    X_filtered = X_raw[mask]

    if X_filtered.shape[0] < 2:
        print("Not enough data points to proceed with GMM fitting.")
        return None

    optimal_k = initial_k; gmm_model = None; X_filtered_data = None
    while optimal_k > 0:
        print(f"Testing k={optimal_k} for 1-SD ellipse overlap...")
        temp_kmeans = KMeans(n_clusters=optimal_k, n_init='auto', random_state=42).fit(X_filtered)
        candidate_gmm, _, _, candidate_X_filtered = Phasor_GMM(G_full, S_full, optimal_k, temp_kmeans.cluster_centers_, min_mag)
        if optimal_k == 1 or (candidate_gmm and not check_ellipse_overlap(candidate_gmm)):
            print(f"No significant overlap detected at k={optimal_k}. This is the optimal model.")
            gmm_model = candidate_gmm; X_filtered_data = candidate_X_filtered
            break
        else:
            print(f"Overlap detected at k={optimal_k}. Trying with k={optimal_k - 1}.")
            optimal_k -= 1

    if gmm_model:
        print("Plotting final GMM fit...")
        plot_gmm_phasor_fit(
            gmm=gmm_model, X_filtered=X_filtered_data, intensities=None,
            title=f"Optimal GMM Fit ({analysis_name})",
            output_path=output_path,
            analysis_name=analysis_name,
            autoclose_plots=autoclose_plots
        )
        return gmm_model, X_filtered_data

    print("Could not find a GMM with non-overlapping 1-SD ellipses.")
    return None, None

def plot_gmm_phasor_fit(gmm, X_filtered, intensities=None, title="GMM Phasor Fit", output_path=None, analysis_name=None, autoclose_plots=True, frequency_mhz=None, harmonic=None):
    """
    Visualizes the GMM fit. Ellipses are colored Left-to-Right (Blue->Purple).
    """

    # Common Data Setup
    probabilities = gmm.predict_proba(X_filtered).max(axis=1)

    # --- Sorting Logic for Colors ---
    comp_color_map, sorted_indices = get_sorted_color_mapping(gmm)

    _freq_mhz = frequency_mhz if frequency_mhz is not None else FREQ
    _harmonic = harmonic if harmonic is not None else HARMONIC

    # Helper to draw GMM components (reused for all plots)
    def draw_gmm_components(ax_obj):
        # We loop through sorted_indices to draw them in logical order, though it doesn't matter for drawing
        # But we MUST use the color map we created.

        for i in range(gmm.n_components):
            mean = gmm.means_[i]
            cov = gmm.covariances_[i]

            # Use the color mapped to this component index
            color = comp_color_map[i]

            ax_obj.scatter(mean[0], mean[1], marker='x', s=120, color=color, linewidths=2)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            for n_std in [1, 2]:
                width, height = 2 * n_std * np.sqrt(eigenvalues)
                ell = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor=color, facecolor='none', linewidth=2, linestyle='-')
                ax_obj.add_artist(ell)

        theta_uc = np.linspace(0, np.pi, 300)
        ax_obj.plot(0.5 + 0.5 * np.cos(theta_uc), 0.5 * np.sin(theta_uc), 'k--', linewidth=1.5, alpha=0.8)

        if SHOW_LIFETIME_MARKERS:
            omega = 2 * np.pi * (_freq_mhz / 1000) * _harmonic
            for tau in LIFETIME_MARKERS:
                g_marker = 1 / (1 + (omega * tau)**2)
                s_marker = (omega * tau) / (1 + (omega * tau)**2)
                ax_obj.scatter(g_marker, s_marker, color='black', marker='x', s=40, zorder=-10)
                ax_obj.annotate(f"{tau}ns", (g_marker, s_marker), textcoords="offset points",
                                 xytext=(-5, 6), ha='right', fontsize=9, color='black', zorder=10)

        s_ylim = (S_AXIS_RANGE[0], S_AXIS_RANGE[1] + 0.03) if SHOW_LIFETIME_MARKERS else S_AXIS_RANGE
        ax_obj.set_xlim(G_AXIS_RANGE); ax_obj.set_ylim(s_ylim); ax_obj.set_aspect('equal')
        ax_obj.set_xlabel("G"); ax_obj.set_ylabel("S")
        
        if SHOW_GRID:
            ax_obj.grid(True, linestyle=':', alpha=0.5)
        else:
            ax_obj.grid(False)

    # --- PLOT 1: ALPHA-WEIGHTED (Prob + Intensity Alpha) ---
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    cmap1 = plt.cm.viridis
    colors1 = cmap1(probabilities)
    
    if intensities is not None:
        vmax = np.percentile(intensities, 95)
        vmin = np.min(intensities)
        if vmax > vmin:
            norm_intensities = np.clip((intensities - vmin) / (vmax - vmin), 0, 1)
        else:
            norm_intensities = np.ones_like(intensities)
        colors1[:, 3] = (norm_intensities ** 3) * 1.0         
        ax1.text(0.02, 0.98, "Alpha = Intensity (3rd power scale)", transform=ax1.transAxes,
                verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        colors1[:, 3] = 0.6 

    ax1.scatter(X_filtered[:, 0], X_filtered[:, 1], s=5, c=colors1,rasterized = True, zorder = -10)
    ax1.set_rasterization_zorder(0)
    sm1 = plt.cm.ScalarMappable(cmap=cmap1, norm=plt.Normalize(vmin=0, vmax=1)); sm1.set_array([])
    fig1.colorbar(sm1, ax=ax1, fraction=0.046, pad=0.04).set_label('Cluster Assignment Probability')
    draw_gmm_components(ax1)
    ax1.set_title(f"{title} (Alpha-Weighted)")
    
    if output_path and analysis_name:
        out_name1 = os.path.join(output_path, f"gmm_phasor_fit_{analysis_name}_AlphaWeighted.png")
        plt.savefig(out_name1,dpi=300, bbox_inches='tight')
        out_name1_eps = os.path.join(output_path, f"gmm_phasor_fit_{analysis_name}_AlphaWeighted.eps")
        plt.savefig(out_name1_eps, dpi=300, bbox_inches='tight')
        print(f"Saved: {out_name1}")

    # Intensity only plot 
    if intensities is not None:
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        cmap2 = plt.cm.plasma
        # For this plot, use linear normalization for color so we can see the range
        sc2 = ax2.scatter(X_filtered[:, 0], X_filtered[:, 1], s=5, c=intensities, cmap=cmap2, alpha=0.6, vmin=vmin, vmax=vmax, rasterized = True, zorder = -10)
        ax2.set_rasterization_zorder(0)
        fig2.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04).set_label('Intensity (Photon Count)')
        draw_gmm_components(ax2)
        ax2.set_title(f"{title} (Intensity Scale)")
        
        if output_path and analysis_name:
            out_name2 = os.path.join(output_path, f"gmm_phasor_fit_{analysis_name}_Intensity.png")
            plt.savefig(out_name2, dpi =300, bbox_inches='tight')
            out_name2_eps = os.path.join(output_path, f"gmm_phasor_fit_{analysis_name}_Intensity.eps")
            plt.savefig(out_name2_eps, dpi=300, bbox_inches='tight')
            print(f"Saved: {out_name2}")
        if autoclose_plots: plt.close(fig2)

    # Standard GMM view
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    sc3 = ax3.scatter(X_filtered[:, 0], X_filtered[:, 1], s=5, c=probabilities, cmap='viridis', alpha=0.6, vmin=0, vmax=1, rasterized = True, zorder = -10)
    ax3.set_rasterization_zorder(0)
    fig3.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.04).set_label('Cluster Assignment Probability')
    draw_gmm_components(ax3)
    ax3.set_title(f"{title} (Standard GMM View)")

    if output_path and analysis_name:
        out_name3 = os.path.join(output_path, f"gmm_phasor_fit_{analysis_name}_StandardGMM.png")
        plt.savefig(out_name3, dpi = 300, bbox_inches='tight')
        print(f"Saved: {out_name3}")
        out_name3_eps = os.path.join(output_path, f"gmm_phasor_fit_{analysis_name}_StandardGMM.eps")
        plt.savefig(out_name3_eps, dpi=300, bbox_inches='tight')
    
    if autoclose_plots:
        plt.close('all')
    else:
        plt.show()

def find_optimal_gmm_by_elbow(G_full, S_full, min_mag=0.01, max_components=10): # Add this to GUI in next version (it is better way)
    """
    Finds the optimal number of GMM components by locating the "elbow" in the BIC/AIC score curves.
    """
    if max_components < 3:
        print("Error: max_components must be at least 3 to find the elbow.")
        return None, None, [], []
    
    _, aic_scores, bic_scores = find_optimal_gmm_components(G_full, S_full, min_mag=min_mag, max_components=max_components)
    if not bic_scores: return None, None, [], []
    
    aic_derivative = np.diff(aic_scores); bic_derivative = np.diff(bic_scores)
    aic_second_derivative = np.diff(aic_derivative); bic_second_derivative = np.diff(bic_derivative)
    
    optimal_k_aic = np.argmax(aic_second_derivative) + 2
    optimal_k_bic = np.argmax(bic_second_derivative) + 2
        


    return optimal_k_bic, optimal_k_aic, aic_scores, bic_scores

def find_optimal_gmm_components(G_full, S_full, min_mag=0.01, max_components=10):
    """
    Determines the optimal number of GMM components using AIC and BIC.
    """
    X_raw = np.vstack((G_full.flatten(), S_full.flatten())).T
    magnitude = np.linalg.norm(X_raw, axis=1)
    mask = magnitude > min_mag
    X_filtered = X_raw[mask]

    if X_filtered.shape[0] < 2:
        print("Warning: Not enough data points after filtering to run GMM.")
        return 1, [], []
    
    if X_filtered.shape[0] < max_components:
        print(f"Warning: Number of data points ({X_filtered.shape[0]}) is less than max_components. Adjusting max_components.")
        max_components = X_filtered.shape[0]

    n_components_range = range(1, max_components + 1)
    aic_scores = []; bic_scores = []

    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=42, covariance_type='full')
        gmm.fit(X_filtered)
        aic_scores.append(gmm.aic(X_filtered)); bic_scores.append(gmm.bic(X_filtered))

    optimal_k = np.argmin(bic_scores) + 1
    return optimal_k, aic_scores, bic_scores

def extract_pixel_data(gmm_model, G_image, S_image, intensity_img, file_name, min_thresh, max_thresh):
    import pandas as pd
    from skimage import measure
    from scipy.stats import multivariate_normal

    rows = []
    intensity_mask = (intensity_img >= min_thresh) & (intensity_img <= max_thresh)
    ys, xs = np.where(intensity_mask)

    if len(xs) == 0:
        return pd.DataFrame()

    G_vals = G_image[ys, xs]
    S_vals = S_image[ys, xs]
    X = np.vstack((G_vals, S_vals)).T

    all_probs = gmm_model.predict_proba(X)
    hard_assigns = gmm_model.predict(X)
    soft_assigns = (all_probs * np.arange(gmm_model.n_components)).sum(axis=1)

    # --- Build a (y, x) -> aggregate_cluster lookup ---
    binary_mask = intensity_mask.astype(np.uint8)
    labeled_img = measure.label(binary_mask)
    regions = measure.regionprops(labeled_img)

    pixel_to_agg_cluster = {}
    for region in regions:
        if region.area < 10:
            continue
        agg_mask = (labeled_img == region.label)
        G_agg = G_image[agg_mask]
        S_agg = S_image[agg_mask]
        X_agg = np.vstack((G_agg, S_agg)).T
        if X_agg.shape[0] == 0:
            continue
        total_log_likelihoods = []
        for c in range(gmm_model.n_components):
            log_probs = multivariate_normal.logpdf(X_agg, mean=gmm_model.means_[c], cov=gmm_model.covariances_[c])
            total_log_likelihoods.append(np.sum(log_probs))
        most_likely = np.argmax(total_log_likelihoods) + 1  # 1-indexed
        for (py, px) in zip(*np.where(agg_mask)):
                pixel_to_agg_cluster[(py, px)] = (region.label, most_likely)
    # --- End lookup build ---

    for i in range(len(xs)):
        row = {
            'file_name': file_name,
            'x_position': xs[i],
            'y_position': ys[i],
            'G_position': G_vals[i],
            'S_position': S_vals[i],
        }
        for c in range(gmm_model.n_components):
            row[f'prob_component_{c+1}'] = all_probs[i, c]
        row['hard_assign_gaussian'] = hard_assigns[i] + 1
        row['soft_assign_gaussian'] = soft_assigns[i] + 1
        agg_lookup = pixel_to_agg_cluster.get((ys[i], xs[i]), (None, None))
        row['aggregate_id'] = agg_lookup[0]
        row['aggregate_cluster_assign'] = agg_lookup[1]        
        rows.append(row)

    return pd.DataFrame(rows)

def extract_aggregate_data(gmm_model, G_image, S_image, intensity_img, file_name, min_thresh, max_thresh):
    """
    Extracts per-aggregate GMM assignment data for saving to CSV.
    Returns a list of dicts:
      [file_name, aggregate_id, aggregate_pixel_size,  gmm_cluster_aggregate_assign]
    """
    import pandas as pd
    from skimage import measure
    from scipy.stats import multivariate_normal

    intensity_mask = (intensity_img >= min_thresh) & (intensity_img <= max_thresh)
    binary_mask = intensity_mask.astype(np.uint8)
    labeled_img = measure.label(binary_mask)
    regions = measure.regionprops(labeled_img)

    rows = []
    for region in regions:
       # if region.area < 10:
       #     continue
        agg_mask = (labeled_img == region.label)
        G_vals = G_image[agg_mask]
        S_vals = S_image[agg_mask]
        X_agg = np.vstack((G_vals, S_vals)).T

        if X_agg.shape[0] == 0:
            continue

        total_log_likelihoods = []
        for c in range(gmm_model.n_components):
            log_probs = multivariate_normal.logpdf(X_agg, mean=gmm_model.means_[c], cov=gmm_model.covariances_[c])
            total_log_likelihoods.append(np.sum(log_probs))

        most_likely = np.argmax(total_log_likelihoods) + 1  # 1-indexed

        rows.append({
            'file_name': file_name,
            'aggregate_id': region.label,
            'aggregate_pixel_size': region.area,
            'gmm_cluster_aggregate_assign': most_likely,
    
            'aggregate_photon_count': float(np.sum(intensity_img[agg_mask])),
            'aggregate_mean_G': float(np.mean(G_vals)),
            'aggregate_mean_S': float(np.mean(S_vals)),
        })

    return pd.DataFrame(rows)


