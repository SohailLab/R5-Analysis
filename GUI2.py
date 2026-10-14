import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import os
import Logic2 as Logic
import sv_ttk
import numpy as np
import csv
#=== ADDED BY CLAUDE ===
import json
from pathlib import Path

#=== ADDED BY CLAUDE ===
# config.json's filepath -- a plain module-level var so it's easy to point elsewhere in
# a later iteration. Path(__file__).parent assumes config.json sits next to this file.
CONFIG_JSON_PATH = Path(__file__).parent / "config.json"


def _load_config():
    """Loads config.json for this GUI's initial values. Returns {} on any read/parse
    error so the GUI just falls back to its hardcoded defaults."""
    try:
        with open(CONFIG_JSON_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


_CONFIG_PLACEHOLDER_PREFIX = "usr/local/data"


def _is_placeholder_path(value):
    """Config filepaths starting with /usr/local/data (with or without the leading
    slash) are machine-specific placeholders, not real paths -- treat as unset."""
    return isinstance(value, str) and value.lstrip("/").startswith(_CONFIG_PLACEHOLDER_PREFIX)


def _config_path(config, key, fallback=""):
    """Filepath-flavored config lookup: blanks out /usr/local/data placeholder paths."""
    value = config.get(key, fallback)
    return fallback if _is_placeholder_path(value) else value


def _config_int_str(config, key, fallback):
    """Formats a config value as an int-parseable string (e.g. 1.0 -> '1'), since
    several fields are later read with int(var.get())."""
    value = config.get(key, None)
    if value is None:
        return fallback
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return fallback


def _config_float_str(config, key, fallback):
    """Formats a config value as a float-parseable string, since several fields are
    later read with float(var.get())."""
    value = config.get(key, None)
    if value is None:
        return fallback
    try:
        return str(float(value))
    except (TypeError, ValueError):
        return fallback


_config = _load_config()

# Front end for LOGIC2.py
# Note this version does intensities the old doesn't
class StdoutRedirector:
    """A class to redirect stdout to a tkinter Text widget for logging."""
    def __init__(self, text_widget):
        self.text_space = text_widget

    def write(self, string):
        self.text_space.insert(tk.END, string)
        self.text_space.see(tk.END)
        try:
            self.text_space.update_idletasks()
        except tk.TclError:
            pass

    def flush(self):
        pass
    
script_dir = os.path.dirname(os.path.abspath(__file__))
#=== ADDED BY CLAUDE ===
# Working_Dir from config.json seeds the file-dialog starting directory (still subject
# to the /usr/local/data placeholder blanking rule, via _config_path).
last_dir = _config_path(_config, "Working_Dir", os.path.abspath(os.path.join(script_dir, "..", "..")))

def select_input_files():
    """Opens a dialog to select analysis TIFF files."""
    global last_dir
    filepaths = filedialog.askopenfilenames(
        initialdir=last_dir,
        title="Select Input TIFF Files",
        filetypes=(("TIFF files", ("*.tiff","*.tif")), ("All files", "*.*"))
    )
    if filepaths:
        last_dir = os.path.dirname(filepaths[0])
        file_listbox.delete(0, tk.END)
        for f in filepaths:
            file_listbox.insert(tk.END, f)

def select_calibration_file():
    """Opens a dialog for the calibration TIFF file."""
    global last_dir
    filepath = filedialog.askopenfilename(
        initialdir=last_dir,
        title="Select Calibration TIFF File",
        filetypes=(("TIFF files", ("*.tiff","*.tif")), ("All files", "*.*"))
    )
    if filepath:
        last_dir = os.path.dirname(filepath)
        calibration_file_var.set(filepath)

#=== ADDED BY CLAUDE ===
last_dir_out = _config_path(_config, "Working_Dir", os.path.abspath(os.path.join(script_dir, "..", "..")))

def select_output_folder():
    """Opens a dialog for the output folder."""
    global last_dir_out
    folderpath = filedialog.askdirectory(
        initialdir=last_dir_out, 
        title="Select Output Folder"
    )
    if last_dir_out:
        last_dir_out = folderpath
        output_folder_var.set(folderpath)

def toggle_manual_calibration():
    """Enables/disables calibration widgets based on the manual checkbox."""
    if manual_cal_var.get():
        cal_file_button.config(state='disabled')
        cal_file_entry.config(state='disabled')
        tau_ref_entry.config(state='disabled')
        manual_phi_entry.config(state='normal')
        manual_m_entry.config(state='normal')
    else:
        cal_file_button.config(state='normal')
        cal_file_entry.config(state='readonly')
        tau_ref_entry.config(state='normal')
        manual_phi_entry.config(state='disabled')
        manual_m_entry.config(state='disabled')

def update_gmm_table(*args):
    """Shows or hides GMM parameter entry rows based on dropdown selection."""
    try:
        num_components = gmm_n_components_var.get()
        for i, row in enumerate(manual_gmm_rows):
            if i < num_components:
                for widget in row:
                    widget.grid()
            else:
                for widget in row:
                    widget.grid_remove()
    except (tk.TclError, ValueError):
        pass # Handle cases where the var is not yet an integer

def toggle_analysis_mode():
    """Shows/hides GUI frames based on the selected analysis mode."""
    mode = analysis_mode_var.get()
    if mode == "fit":
        fit_gmm_frame.grid()
        apply_gmm_frame.grid_remove()
    elif mode == "apply":
        fit_gmm_frame.grid_remove()
        apply_gmm_frame.grid()


def import_gmm_from_csv():
    """Opens a file dialog to import GMM parameters from a CSV file."""
    global last_dir_out
    filepath = filedialog.askopenfilename(
        title="Import GMM Parameters from CSV",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        initialdir=last_dir_out
    )
    if not filepath:
        return
    last_dir_out = os.path.dirname(filepath)
    try:
        with open(filepath, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = [h.strip() for h in next(reader)]
            expected_header_1 = ["Component", "Weight", "Mean_G", "Mean_S", "Covariance_GG", "Covariance_GS", "Covariance_SG", "Covariance_SS"]
            expected_header_2 = ['Filename', 'Component', 'Mean_G', 'Mean_S', 'Weight', 'Covariance_GG', 'Covariance_GS', 'Covariance_SS']

            if header not in [expected_header_1, expected_header_2]:
                messagebox.showwarning("CSV Format Error", "The CSV header does not match the expected format.")
                return

            params = [row for row in reader if not row[1].startswith('Overall')]
            num_components = len(params)
            
            if not (1 <= num_components <= 12):
                messagebox.showerror("CSV Error", f"The CSV file must contain between 1 and 12 components, but {num_components} were found.")
                return

            gmm_n_components_var.set(num_components)
            for i, row_data in enumerate(params):
                if header == expected_header_1:
                    gmm_param_vars[i]['weight'].set(row_data[1])
                    gmm_param_vars[i]['mean_g'].set(row_data[2])
                    gmm_param_vars[i]['mean_s'].set(row_data[3])
                    gmm_param_vars[i]['cov_gg'].set(row_data[4])
                    gmm_param_vars[i]['cov_gs'].set(row_data[5])
                    gmm_param_vars[i]['cov_sg'].set(row_data[6])
                    gmm_param_vars[i]['cov_ss'].set(row_data[7])
                else:
                    gmm_param_vars[i]['mean_g'].set(row_data[2])
                    gmm_param_vars[i]['mean_s'].set(row_data[3])
                    gmm_param_vars[i]['weight'].set(row_data[4])
                    gmm_param_vars[i]['cov_gg'].set(row_data[5])
                    gmm_param_vars[i]['cov_gs'].set(row_data[6])
                    gmm_param_vars[i]['cov_ss'].set(row_data[7])

            messagebox.showinfo("Success", f"Successfully imported {num_components} GMM component(s).")
    except Exception as e:
        messagebox.showerror("CSV Read Error", f"An error occurred while reading the CSV file: {e}")

def run_analysis_workflow():
    """Validates GUI inputs and calls the master logic function."""
    try:
        params = {
            "analysis_mode": analysis_mode_var.get(),
            "filepaths": file_listbox.get(0, tk.END),
            "output_folder": output_folder_var.get(),
            # Analysis Params
            "bin_size": int(bin_size_var.get()),
            "min_threshold": float(min_threshold_var.get()),
            "max_threshold": float(max_threshold_var.get()),
            # Calibration
            "calibrate_manually": manual_cal_var.get(),
            "bin_width_ns": float(bin_width_var.get()),
            "freq_mhz": float(freq_var.get()),
            "harmonic": int(harmonic_var.get()),
            "calibration_filepath": calibration_file_var.get(),
            "tau_ref": float(tau_ref_var.get()),
            "manual_phi": float(manual_phi_var.get()),
            "manual_m": float(manual_m_var.get()),
            # General Options
            "autoclose_plots": autoclose_plots_var.get(),
            "delete_intermediates": delete_intermediates_var.get(),
            "export_pdf": export_pdf_var.get(), # new task of exporting pdf report... 
            # GMM Params
            "gmm_components": int(gmm_n_components_fit_var.get()),
            "gmm_auto_find": gmm_auto_find_var.get(),
            "manual_gmm_params": None,
            "model_type": model_type_var.get(),
        }

        if params["analysis_mode"] == "apply":
            num_components = gmm_n_components_var.get()
            weights, means, covariances = [], [], []
            for i in range(num_components):
                weights.append(float(gmm_param_vars[i]['weight'].get()))
                means.append([float(gmm_param_vars[i]['mean_g'].get()), float(gmm_param_vars[i]['mean_s'].get())])
                covariances.append([
                    [float(gmm_param_vars[i]['cov_gg'].get()), float(gmm_param_vars[i]['cov_gs'].get())],
                    [float(gmm_param_vars[i]['cov_sg'].get()), float(gmm_param_vars[i]['cov_ss'].get())]
                ])
            if not np.isclose(sum(weights), 1.0):
                if not messagebox.askokcancel("Input Warning", f"GMM weights should sum to 1.0, but the current sum is {sum(weights):.4f}.\n\nDo you want to continue anyway?"):
                    return
            params["manual_gmm_params"] = {"weights": np.array(weights), "means": np.array(means), "covariances": np.array(covariances)}

    except ValueError as e:
        messagebox.showerror("Input Error", f"Invalid number in one of the fields.\nDetails: {e}")
        return

    if not params["filepaths"]:
        messagebox.showerror("Input Error", "Please select one or more input TIFF files.")
        return
    if not params["output_folder"]:
        messagebox.showerror("Input Error", "Please select an output folder.")
        return
    if not params["calibrate_manually"] and not params["calibration_filepath"]:
        messagebox.showerror("Input Error", "Please select a calibration file or choose manual calibration.")
        return

    print(f"--- Starting Analysis Workflow (Mode: {params['analysis_mode'].upper()}) ---")
    try:
        Logic.run_master_workflow(**params)
        print("\n--- Analysis Finished Successfully! ---")
        messagebox.showinfo("Success", "Analysis completed successfully!")
    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED IN THE WORKFLOW ---")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("Workflow Error", f"An error occurred during processing:\n\n{e}")

# --- GUI Setup ---
root = tk.Tk()
root.title("Phasor FLIM-GMM Analysis")
sv_ttk.set_theme("light")



model_type_var = tk.StringVar(value="BIC+SD")


# ADDED Scrollable Wrapper Setup (should fix scaling issue)
# https://www.geeksforgeeks.org/python/python-tkinter-scrollbar/
# Makes container to hold the Canvas and Scrollbar
container = ttk.Frame(root)
container.pack(fill="both", expand=True)

# Makes Canvas
canvas = tk.Canvas(container)
canvas.pack(side="left", fill="both", expand=True)

# Add the Scrollbar linked to the Canvas
scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
scrollbar.pack(side="right", fill="y")
canvas.configure(yscrollcommand=scrollbar.set)

# Makes mainframe 
main_frame = ttk.Frame(canvas, padding="10")
canvas_frame = canvas.create_window((0, 0), window=main_frame, anchor="nw")

# Scrolling and resize logic 
def on_frame_configure(event):
    """Reset the scroll region to encompass the inner frame"""
    canvas.configure(scrollregion=canvas.bbox("all"))

def on_canvas_configure(event):
    """Resize the inner frame to match the canvas width"""
    canvas.itemconfig(canvas_frame, width=event.width)
def on_mousewheel(event):
    """Enable mousewheel scrolling"""
    # This was an AI suggestion but imma assume its fine (see linux support below)
    # Windows/MacOS: event.delta 
    # Linux: Button-4/Button-5
    if event.delta:
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")

main_frame.bind("<Configure>", on_frame_configure)
canvas.bind("<Configure>", on_canvas_configure)
canvas.bind_all("<MouseWheel>", on_mousewheel)
# Linux support
canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

analysis_mode_var = tk.StringVar(value="fit")
#=== ADDED BY CLAUDE ===
# calibration_file_var seeded from config.json's TIFF_Cal_Filepath (blanked if it's the
# /usr/local/data placeholder). No dedicated "output_folder" config key exists, so that
# one keeps its blank default.
calibration_file_var = tk.StringVar(value=_config_path(_config, "TIFF_Cal_Filepath"))
output_folder_var = tk.StringVar()
#=== ADDED BY CLAUDE ===
bin_size_var = tk.StringVar(value="1")  # no config key for this yet
min_threshold_var = tk.StringVar(value=_config_float_str(_config, "Min_Intensity_Threshold", "0.1"))
max_threshold_var = tk.StringVar(value=_config_float_str(_config, "Max_Intensity_Threshold", "1.0"))
manual_cal_var, autoclose_plots_var = tk.BooleanVar(value=False), tk.BooleanVar(value=True)
delete_intermediates_var = tk.BooleanVar(value=True)
export_pdf_var = tk.BooleanVar(value=True)
#=== ADDED BY CLAUDE ===
bin_width_var = tk.StringVar(value=_config_float_str(_config, "Bin_Width_ns", "0.0244"))
freq_var = tk.StringVar(value=_config_float_str(_config, "Laser_rep_rate_MHz", "80.0"))
harmonic_var = tk.StringVar(value=_config_int_str(_config, "Harmonic", "1"))
tau_ref_var = tk.StringVar(value="2.5")  # no config key for this yet
manual_phi_var = tk.StringVar(value=_config_float_str(_config, "phi_cal", "0.0"))
manual_m_var = tk.StringVar(value=_config_float_str(_config, "m_cal", "1.0"))
gmm_n_components_fit_var = tk.StringVar(value="3")
gmm_auto_find_var = tk.BooleanVar(value=True)
gmm_n_components_var = tk.IntVar(value=2)
gmm_param_vars = []
#=== ADDED BY CLAUDE ===
# Seed the first components' manual mean_g/mean_s from config.json's G_centers/S_centers.
_config_g_centers = _config.get("G_centers", [])
_config_s_centers = _config.get("S_centers", [])
for _i in range(12):
    _default_mean_g = str(_config_g_centers[_i]) if _i < len(_config_g_centers) else '0.5'
    _default_mean_s = str(_config_s_centers[_i]) if _i < len(_config_s_centers) else '0.5'
    gmm_param_vars.append({
        'weight': tk.StringVar(value='0.5'), 'mean_g': tk.StringVar(value=_default_mean_g), 'mean_s': tk.StringVar(value=_default_mean_s),
        'cov_gg': tk.StringVar(value='0.01'), 'cov_gs': tk.StringVar(value='0.0'),
        'cov_sg': tk.StringVar(value='0.0'), 'cov_ss': tk.StringVar(value='0.01')
    })
gmm_n_components_var.trace_add("write", update_gmm_table)

# --- GUI Layout ---
# REMOVED: Redefinition of main_frame and root.grid() calls that caused the error.

files_frame = ttk.LabelFrame(main_frame, text="File I/O", padding=10)
files_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
files_frame.columnconfigure(1, weight=1)
ttk.Label(files_frame, text="Input TIFF File(s):").grid(row=0, column=0, sticky="nw", padx=5)
file_list_frame = ttk.Frame(files_frame); file_list_frame.grid(row=0, column=1, sticky="nsew"); file_list_frame.columnconfigure(0, weight=1)
file_listbox = tk.Listbox(file_list_frame, height=4, selectmode=tk.EXTENDED); file_listbox.grid(row=0, column=0, sticky="nsew")
scrollbar_list = ttk.Scrollbar(file_list_frame, orient=tk.VERTICAL, command=file_listbox.yview); file_listbox['yscrollcommand'] = scrollbar_list.set; scrollbar_list.grid(row=0, column=1, sticky="ns")
ttk.Button(files_frame, text="Browse...", command=select_input_files).grid(row=0, column=2, padx=5, sticky="n")
cal_file_button = ttk.Button(files_frame, text="Select Calibration TIFF...", command=select_calibration_file); cal_file_button.grid(row=1, column=0, sticky="w", padx=5, pady=5)
cal_file_entry = ttk.Entry(files_frame, textvariable=calibration_file_var, state="readonly"); cal_file_entry.grid(row=1, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
ttk.Button(files_frame, text="Select Output Folder...", command=select_output_folder).grid(row=2, column=0, sticky="w", padx=5, pady=5)
ttk.Entry(files_frame, textvariable=output_folder_var, state="readonly").grid(row=2, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

left_col_frame = ttk.Frame(main_frame); left_col_frame.grid(row=1, column=0, sticky="nsew")
right_col_frame = ttk.Frame(main_frame); right_col_frame.grid(row=1, column=1, sticky="nsew")
main_frame.columnconfigure(0, weight=1); main_frame.columnconfigure(1, weight=1)

params_frame = ttk.LabelFrame(left_col_frame, text="Analysis Parameters", padding=10)
params_frame.pack(fill="x", expand=True, padx=5, pady=5)
ttk.Label(params_frame, text="Bin Size:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(params_frame, textvariable=bin_size_var, width=10).grid(row=0, column=1, sticky="w", padx=5, pady=2)
ttk.Label(params_frame, text="Min Intensity Threshold:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(params_frame, textvariable=min_threshold_var, width=10).grid(row=1, column=1, sticky="w", padx=5, pady=2)
ttk.Label(params_frame, text="Max Intensity Threshold:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(params_frame, textvariable=max_threshold_var, width=10).grid(row=2, column=1, sticky="w", padx=5, pady=2)
ttk.Checkbutton(params_frame, text="Automatically close plots", variable=autoclose_plots_var).grid(row=4, column=0, columnspan=2, sticky="w", padx=5, pady=5)
ttk.Checkbutton(params_frame, text="Delete G/S files after use", variable=delete_intermediates_var).grid(row=5, column=0, columnspan=2, sticky="w", padx=5, pady=5)
ttk.Checkbutton(params_frame, text="Export PDF Summary Report", variable=export_pdf_var).grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=5)
cal_frame = ttk.LabelFrame(right_col_frame, text="Calibration Settings", padding=10)
cal_frame.pack(fill="both", expand=True, padx=5, pady=5)
ttk.Checkbutton(cal_frame, text="Calibrate Manually", variable=manual_cal_var, command=toggle_manual_calibration).grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)
ttk.Label(cal_frame, text="Manual Phi:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
manual_phi_entry = ttk.Entry(cal_frame, textvariable=manual_phi_var, width=10); manual_phi_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)
ttk.Label(cal_frame, text="Manual M:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
manual_m_entry = ttk.Entry(cal_frame, textvariable=manual_m_var, width=10); manual_m_entry.grid(row=2, column=1, sticky="w", padx=5, pady=2)
ttk.Label(cal_frame, text="Time Bin Width (ns):").grid(row=3, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(cal_frame, textvariable=bin_width_var, width=10).grid(row=3, column=1, sticky="w", padx=5, pady=2)
ttk.Label(cal_frame, text="Frequency (MHz):").grid(row=4, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(cal_frame, textvariable=freq_var, width=10).grid(row=4, column=1, sticky="w", padx=5, pady=2)
ttk.Label(cal_frame, text="Harmonic:").grid(row=5, column=0, sticky="w", padx=5, pady=2)
ttk.Entry(cal_frame, textvariable=harmonic_var, width=10).grid(row=5, column=1, sticky="w", padx=5, pady=2)
ttk.Label(cal_frame, text="Reference Lifetime (ns):").grid(row=6, column=0, sticky="w", padx=5, pady=2)
tau_ref_entry = ttk.Entry(cal_frame, textvariable=tau_ref_var, width=10); tau_ref_entry.grid(row=6, column=1, sticky="w", padx=5, pady=2)
mode_frame = ttk.LabelFrame(main_frame, text="Workflow", padding=10)
mode_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
ttk.Radiobutton(mode_frame, text="Step 1: Fit New GMM Model from Aggregated Data", variable=analysis_mode_var, value="fit", command=toggle_analysis_mode).pack(anchor="w", padx=5)
ttk.Radiobutton(mode_frame, text="Step 2: Apply Existing GMM Model to Files", variable=analysis_mode_var, value="apply", command=toggle_analysis_mode).pack(anchor="w", padx=5)

fit_gmm_frame = ttk.LabelFrame(main_frame, text="GMM Fitting Parameters", padding=10)
fit_gmm_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
ttk.Label(fit_gmm_frame, text="Segmentation Model:").grid(row=0, column=0, sticky="w", padx=5)
ttk.Combobox(fit_gmm_frame, textvariable=model_type_var, values=["BIC+SD", "AIC", "Manual"], state="readonly", width=8).grid(row=0, column=1, sticky="w", padx=5)
ttk.Label(fit_gmm_frame, text="Number/Max Components:").grid(row=0, column=2, sticky="e", padx=5)
ttk.Entry(fit_gmm_frame, textvariable=gmm_n_components_fit_var, width=7).grid(row=0, column=3, sticky="w", padx=5)

apply_gmm_frame = ttk.LabelFrame(main_frame, text="GMM Model Parameters for Overlay", padding=10)
apply_gmm_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
manual_gmm_control_frame = ttk.Frame(apply_gmm_frame); manual_gmm_control_frame.grid(row=0, column=0, columnspan=8, sticky='ew', pady=(5,5))
ttk.Label(manual_gmm_control_frame, text="Components:").pack(side='left', padx=5)
ttk.Spinbox(manual_gmm_control_frame, from_=1, to=5, textvariable=gmm_n_components_var, width=5, state='readonly', wrap=True).pack(side='left')
ttk.Button(manual_gmm_control_frame, text="Import from CSV", command=import_gmm_from_csv).pack(side='left', padx=10)
manual_gmm_table_frame = ttk.Frame(apply_gmm_frame); manual_gmm_table_frame.grid(row=1, column=0, columnspan=8, sticky='ew')
headers = ["Comp", "Weight", "Mean_G", "Mean_S", "Cov_GG", "Cov_GS", "Cov_SG", "Cov_SS"]
for j, header in enumerate(headers):
    ttk.Label(manual_gmm_table_frame, text=header, font=('Helvetica', 9, 'bold')).grid(row=0, column=j, padx=2, pady=2, sticky='w')
manual_gmm_rows = []
for i in range(5):
    param_vars = gmm_param_vars[i]
    row_widgets = [ttk.Label(manual_gmm_table_frame, text=str(i+1)),
                   ttk.Entry(manual_gmm_table_frame, textvariable=param_vars['weight'], width=8),
                   ttk.Entry(manual_gmm_table_frame, textvariable=param_vars['mean_g'], width=8),
                   ttk.Entry(manual_gmm_table_frame, textvariable=param_vars['mean_s'], width=8),
                   ttk.Entry(manual_gmm_table_frame, textvariable=param_vars['cov_gg'], width=8),
                   ttk.Entry(manual_gmm_table_frame, textvariable=param_vars['cov_gs'], width=8),
                   ttk.Entry(manual_gmm_table_frame, textvariable=param_vars['cov_sg'], width=8),
                   ttk.Entry(manual_gmm_table_frame, textvariable=param_vars['cov_ss'], width=8)]
    for j, widget in enumerate(row_widgets):
        widget.grid(row=i+1, column=j, padx=2, pady=2, sticky="ew")
        widget.grid_remove()
    manual_gmm_rows.append(row_widgets)

action_frame = ttk.Frame(main_frame); action_frame.grid(row=4, column=0, columnspan=2, pady=10)
ttk.Button(action_frame, text="RUN ANALYSIS", command=run_analysis_workflow, style="Accent.TButton").pack()
log_frame = ttk.LabelFrame(main_frame, text="Log", padding=10); log_frame.grid(row=5, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
main_frame.rowconfigure(5, weight=1)
log_text = tk.Text(log_frame, height=15); log_text.pack(fill="both", expand=True)
sys.stdout = StdoutRedirector(log_text)

style = ttk.Style()
style.configure("Accent.TButton", font=("Helvetica", 12, "bold"))
toggle_manual_calibration()
update_gmm_table()
toggle_analysis_mode()

#=== ADDED BY CLAUDE ===
# Seed the input-files listbox from config.json's TIFF_Filepaths_GMM, dropping any
# /usr/local/data placeholder paths (left blank if none are real paths).
for _fp in _config.get("TIFF_Filepaths_GMM", []):
    if not _is_placeholder_path(_fp):
        file_listbox.insert(tk.END, _fp)

root.mainloop()
