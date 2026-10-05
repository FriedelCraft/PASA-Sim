import os
import sys
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Suppress runtime and matplotlib layout warnings for a clean console
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# =========================================================
# CONFIGURATION & STYLING (PUBLICATION QUALITY)
# =========================================================
FILE_CLEAR = "clear.npz"
FILE_POLL = "pollution.npz"

# New output directory as requested
FIGURE_DIR = "final_figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# Publication-quality formatting suitable for ACRS
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.autolayout': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 600  # High DPI for publication
})

COLOR_CLEAR = '#2c3e50'
COLOR_POLL = '#e74c3c'
COLOR_LOSS = '#f39c12'

# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def safe_load(filepath):
    if not os.path.exists(filepath):
        print(f"[ERROR] Missing file: {filepath}")
        return None
    return np.load(filepath)

def drop_nans(arr):
    arr = np.asarray(arr).flatten()
    return arr[~np.isnan(arr)]

def get_time_axis(arr, expected_len):
    if arr.ndim < 2: return 0 if len(arr) == expected_len else None
    if arr.shape[1] == expected_len: return 1
    elif arr.shape[0] == expected_len: return 0
    return None

def extract_hourly_timeseries(hourly_array, time_axis):
    face_axis = 1 - time_axis
    return np.nanmean(hourly_array, axis=face_axis)

# =========================================================
# FIGURE GENERATION
# =========================================================

def generate_fig1_histogram(c_data, p_data):
    print("Generating Figure 1: Annual Irradiance Distribution...")
    ann_c = drop_nans(c_data.get('annual_irradiance_wh_m2', [])) / 1000.0
    ann_p = drop_nans(p_data.get('annual_irradiance_wh_m2', [])) / 1000.0
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ann_c, bins='auto', alpha=0.6, label='Clear Sky', color=COLOR_CLEAR, density=True)
    ax.hist(ann_p, bins='auto', alpha=0.6, label='Pollution', color=COLOR_POLL, density=True)
    
    ax.set_xlabel('Annual Integrated Irradiance (kWh m⁻²)')
    ax.set_ylabel('Density')
    ax.set_title('Annual Integrated Irradiance Distribution')
    ax.legend(frameon=True, edgecolor='black')
    
    fig.savefig(os.path.join(FIGURE_DIR, "Fig1_Annual_Irradiance_Distribution.png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def generate_fig2_monthly_energy(c_data, p_data):
    print("Generating Figure 2: Monthly Energy Comparison...")
    h_c, h_p = c_data.get('hourly_irradiance_w_m2'), p_data.get('hourly_irradiance_w_m2')
    t_unix = c_data.get('time_unix_s')
    t_axis = get_time_axis(h_c, len(t_unix))
    
    ts_c = extract_hourly_timeseries(h_c, t_axis) / 1000.0
    ts_p = extract_hourly_timeseries(h_p, t_axis) / 1000.0
    dt_idx = pd.to_datetime(t_unix, unit='s')
    df = pd.DataFrame({'Clear': ts_c, 'Pollution': ts_p}, index=dt_idx)
    
    # Fix month ordering bug by explicitly grouping by integer month (1-12)
    df['Month'] = df.index.month
    monthly = df.groupby('Month').sum().reindex(range(1, 13))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        monthly['Loss_Pct'] = np.where(monthly['Clear'] > 0, ((monthly['Clear'] - monthly['Pollution']) / monthly['Clear']) * 100.0, 0.0)
    
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x_labels = [calendar.month_abbr[m] for m in range(1, 13)] # Strict Jan-Dec list
    x = np.arange(len(x_labels))
    width = 0.35
    
    ax1.bar(x - width/2, monthly['Clear'], width, label='Clear', color=COLOR_CLEAR)
    ax1.bar(x + width/2, monthly['Pollution'], width, label='Polluted', color=COLOR_POLL)
    ax1.set_ylabel('Monthly Integrated Irradiance (kWh m⁻²)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    
    ax2 = ax1.twinx()
    ax2.plot(x, monthly['Loss_Pct'], color=COLOR_LOSS, marker='o', lw=2.5, markersize=7, label='Loss %')
    ax2.set_ylabel('Energy Loss (%)', color=COLOR_LOSS)
    
    # Clean combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    
    fig.savefig(os.path.join(FIGURE_DIR, "Fig2_Monthly_Energy_Comparison.png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def generate_fig3_scatter(c_data, p_data):
    print("Generating Figure 3: Clear vs Polluted Scatter...")
    ann_c = drop_nans(c_data.get('annual_irradiance_wh_m2', [])) / 1000.0
    ann_p = drop_nans(p_data.get('annual_irradiance_wh_m2', [])) / 1000.0
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Pure scatter with high transparency for publication density mapping
    ax.scatter(ann_c, ann_p, s=0.5, alpha=0.15, color=COLOR_POLL, rasterized=True)
        
    ax.plot([0, max(ann_c)], [0, max(ann_c)], 'k--', lw=1.5, label='1:1 Reference Line')
    ax.set_xlabel('Clear-Sky Annual Irradiance (kWh m⁻²)')
    ax.set_ylabel('Pollution-Attenuated Irradiance (kWh m⁻²)')
    ax.set_title('Annual Integrated Irradiance per Mesh Face:\nClear-Sky vs. Pollution-Attenuated', pad=15)
    
    # Create a custom legend handle for the scatter points so it looks clean
    import matplotlib.lines as mlines
    scatter_proxy = mlines.Line2D([], [], color=COLOR_POLL, marker='o', linestyle='None',
                                  markersize=5, alpha=0.6, label='Mesh Face')
    line_proxy = mlines.Line2D([], [], color='k', linestyle='--', lw=1.5, label='1:1 Reference Line')
    ax.legend(handles=[scatter_proxy, line_proxy], loc='upper left', frameon=True)
    
    fig.savefig(os.path.join(FIGURE_DIR, "Fig3_Clear_vs_Polluted_Scatter.png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def generate_fig4_total_loss(c_data, p_data):
    print("Generating Figure 4: Total Energy Loss Info-graphic...")
    ann_c_raw = c_data.get('annual_irradiance_wh_m2', np.array([]))
    ann_p_raw = p_data.get('annual_irradiance_wh_m2', np.array([]))
    areas_raw = c_data.get('area_m2', np.array([]))
    
    if len(ann_c_raw) == 0 or len(ann_p_raw) == 0 or len(areas_raw) == 0: 
        return
    
    valid_mask = ~np.isnan(ann_c_raw) & ~np.isnan(ann_p_raw) & ~np.isnan(areas_raw)
    
    ann_c, ann_p, areas = ann_c_raw[valid_mask], ann_p_raw[valid_mask], areas_raw[valid_mask]
    energy_c = (ann_c * areas) / 1000.0  
    energy_p = (ann_p * areas) / 1000.0  
    total_loss_kwh = np.sum(energy_c) - np.sum(energy_p)
    
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Clean, centralized infographic text without the redundancy
    ax.text(0.5, 0.75, "Total Annual Energy Loss (Civil Lines Study Area)", fontsize=16, fontweight='bold', ha='center', va='center')
    
    text_vals = (
        f"{total_loss_kwh:,.2f} kWh\n"
        f"{total_loss_kwh / 1e3:,.2f} MWh\n"
        f"{total_loss_kwh / 1e6:,.2f} GWh"
    )
    ax.text(0.5, 0.35, text_vals, fontsize=14, ha='center', va='center', linespacing=1.8, color=COLOR_LOSS, fontweight='bold')
    
    fig.savefig(os.path.join(FIGURE_DIR, "Fig4_Total_Energy_Loss.png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def generate_fig5_aod_transmission(p_data):
    print("Generating Figure 5: Monthly AOD and Transmission...")
    aod = p_data.get('aod_group_value')          
    trans = p_data.get('aod_group_transmission') 
    t_unix = p_data.get('time_unix_s')
    
    t_axis = get_time_axis(aod, len(t_unix))
    if t_axis is None: return
    g_axis = 1 - t_axis
    
    mean_aod = np.nanmean(aod, axis=g_axis)
    mean_trans = np.nanmean(trans, axis=g_axis)
        
    dt_idx = pd.to_datetime(t_unix, unit='s')
    df = pd.DataFrame({'AOD': mean_aod, 'Trans': mean_trans}, index=dt_idx)
    
    # Fix month ordering
    df['Month'] = df.index.month
    monthly = df.groupby('Month').mean().reindex(range(1, 13))
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    months_str = [calendar.month_abbr[m] for m in range(1, 13)]
    
    # AOD Line
    line1 = ax1.plot(months_str, monthly['AOD'], color=COLOR_POLL, marker='o', lw=2, markersize=7, label='Aerosol Optical Depth (AOD)')
    ax1.set_ylabel('Aerosol Optical Depth (AOD)', color=COLOR_POLL)
    
    # Transmission Line
    ax2 = ax1.twinx()
    line2 = ax2.plot(months_str, monthly['Trans'], color=COLOR_CLEAR, marker='s', lw=2, markersize=7, label='Atmospheric Transmission')
    ax2.set_ylabel('Atmospheric Transmission', color=COLOR_CLEAR)
    
    # Unified Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
    
    fig.savefig(os.path.join(FIGURE_DIR, "Fig5_Monthly_AOD_Transmission.png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def generate_captions_file():
    print("Generating figure_captions.txt...")
    captions = (
        "Figure 1: Distribution of simulated annual face irradiance under clear-sky and pollution-attenuated conditions across the Civil Lines study area.\n\n"
        "Figure 2: Monthly integrated irradiance comparison between clear-sky and pollution-attenuated scenarios, overlaying the percentage energy loss. The seasonal variance tracks local pollution cycles.\n\n"
        "Figure 3: Relationship between clear-sky and pollution-attenuated annual integrated irradiance. Each point represents one mesh face. The deviation from the 1:1 reference line highlights the geometric dependency of atmospheric attenuation.\n\n"
        "Figure 4: Total annual energy deficit across the Civil Lines Study Area due to atmospheric pollution.\n\n"
        "Figure 5: Monthly mean Aerosol Optical Depth (AOD) and the corresponding atmospheric transmission coefficient, illustrating the physical driver behind the observed seasonal energy losses."
    )
    with open(os.path.join(FIGURE_DIR, "figure_captions.txt"), "w", encoding='utf-8') as f:
        f.write(captions)

# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    print("==================================================")
    print("PASA-Sim: ACRS 2026 Publication Figure Generator")
    print("==================================================")
    
    c_data, p_data = safe_load(FILE_CLEAR), safe_load(FILE_POLL)
    if c_data is None or p_data is None:
        print("[CRITICAL] Cannot proceed without both NPZ files.")
        sys.exit(1)
        
    generate_fig1_histogram(c_data, p_data)
    generate_fig2_monthly_energy(c_data, p_data)
    generate_fig3_scatter(c_data, p_data)
    generate_fig4_total_loss(c_data, p_data)
    generate_fig5_aod_transmission(p_data)
    generate_captions_file()
    
    print(f"\n[SUCCESS] Pipeline complete. Publication figures saved to: {FIGURE_DIR}/")

if __name__ == "__main__":
    main()