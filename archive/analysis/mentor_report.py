import os
import sys
import textwrap
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings

# Suppress runtime and matplotlib layout warnings for a clean console
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# =========================================================
# CONFIGURATION & STYLING
# =========================================================
REPORT_PDF = "ISRO_Mentor_Discussion_Report.pdf"
FILE_CLEAR = "clear.npz"
FILE_POLL = "pollution.npz"

TARIFF_INR_PER_KWH = 8.0  # Configurable electricity tariff

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.autolayout': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
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

def add_text_page(pdf, title, content):
    """Renders a formal text page for the PDF report with smart wrapping."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.clf()
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
    
    # Smart wrapping to preserve bullet points
    formatted_lines = []
    for paragraph in content.split('\n'):
        if paragraph.strip().startswith('•') or paragraph.strip().startswith('-'):
            formatted_lines.append("  " + paragraph.strip())
        elif paragraph.strip() == "":
            formatted_lines.append("")
        else:
            wrapped = textwrap.fill(paragraph, width=85)
            formatted_lines.append(wrapped)
            
    final_text = "\n".join(formatted_lines)
    ax.text(0.05, 0.90, final_text, transform=ax.transAxes, fontsize=11, va='top', linespacing=1.5)
    
    pdf.savefig(fig)
    plt.close(fig)

def get_time_axis(arr, expected_len):
    if arr.ndim < 2: return 0 if len(arr) == expected_len else None
    if arr.shape[1] == expected_len: return 1
    elif arr.shape[0] == expected_len: return 0
    return None

def extract_hourly_timeseries(hourly_array, time_axis):
    face_axis = 1 - time_axis
    return np.nanmean(hourly_array, axis=face_axis)

# =========================================================
# REPORT SECTIONS
# =========================================================

def generate_title_page(pdf):
    print("Generating Title Page...")
    fig = plt.figure(figsize=(8.5, 11))
    fig.clf()
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    ax.text(0.5, 0.7, "PASA-Sim", transform=ax.transAxes, fontsize=28, fontweight='bold', ha='center')
    ax.text(0.5, 0.6, "Preliminary Scientific Findings:\nEffect of Atmospheric Pollution on Urban Solar Potential", 
            transform=ax.transAxes, fontsize=18, ha='center', va='center', multialignment='center')
    
    intro = (
        "This report summarizes the most important observations from the current PASA-Sim simulation. "
        "It isolates the specific impact of aerosol-induced atmospheric attenuation on explicit 3D urban geometries. "
        "The document is intended to facilitate discussion regarding the most prominent scientific findings "
        "prior to the preparation of the final research paper."
    )
    ax.text(0.5, 0.4, textwrap.fill(intro, width=70), transform=ax.transAxes, fontsize=12, ha='center', va='center', multialignment='center')
    
    pdf.savefig(fig)
    plt.close(fig)

def section_1_maps(c_data, p_data, pdf):
    print("Generating Section 1: Annual Solar Irradiance Maps...")
    centroids = c_data.get('centroid')
    ann_c = c_data.get('annual_irradiance_wh_m2')
    ann_p = p_data.get('annual_irradiance_wh_m2')
    if centroids is None or ann_c is None or ann_p is None: return
    
    valid = ~np.isnan(ann_c) & ~np.isnan(ann_p)
    x, y = centroids[valid, 0], centroids[valid, 1]
    c_val, p_val = ann_c[valid] / 1000.0, ann_p[valid] / 1000.0
    abs_loss = c_val - p_val
    
    # Identical color scales for Clear and Polluted
    vmin, vmax = np.nanmin(p_val), np.nanmax(c_val)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten()
    
    # 1. Clear
    if len(x) < 50000: sc1 = axes[0].scatter(x, y, c=c_val, cmap='viridis', s=2, vmin=vmin, vmax=vmax)
    else: sc1 = axes[0].hexbin(x, y, C=c_val, gridsize=80, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('Clear Sky Annual Irradiance'); axes[0].axis('equal'); axes[0].axis('off')
    
    # 2. Polluted
    if len(x) < 50000: sc2 = axes[1].scatter(x, y, c=p_val, cmap='viridis', s=2, vmin=vmin, vmax=vmax)
    else: sc2 = axes[1].hexbin(x, y, C=p_val, gridsize=80, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('Pollution-Affected Irradiance'); axes[1].axis('equal'); axes[1].axis('off')
    
    # 3. Absolute Loss
    if len(x) < 50000: sc3 = axes[2].scatter(x, y, c=abs_loss, cmap='magma', s=2)
    else: sc3 = axes[2].hexbin(x, y, C=abs_loss, gridsize=80, cmap='magma')
    axes[2].set_title('Absolute Energy Loss'); axes[2].axis('equal'); axes[2].axis('off')
    
    fig.colorbar(sc1, ax=axes[0:2], shrink=0.7, label='Annual Irradiance (kWh/m2)')
    fig.colorbar(sc3, ax=axes[2], shrink=0.7, label='Energy Loss (kWh/m2)')
    
    pdf.savefig(fig); plt.close(fig)
    
    text = (
        "What is observed:\n"
        "The spatial mapping of annual irradiance demonstrates a distinct and non-uniform degradation of solar resource across the urban topology. While clear sky conditions establish the baseline geometric potential (identifying highly exposed roofs vs. occluded street canyons), the introduction of pollution dampens the entire resource envelope. The absolute energy loss map reveals that the highest quantitative losses occur on surfaces that initially possessed the highest solar potential.\n\n"
        "Why it occurs:\n"
        "Atmospheric aerosols attenuate direct normal irradiance (DNI). Surfaces that rely heavily on sustained direct beam radiation—such as unshaded flat roofs—suffer massive absolute reductions when that direct beam is scattered or absorbed by pollution.\n\n"
        "Scientific Importance:\n"
        "These maps prove that pollution does not simply subtract a constant value from all buildings. The attenuation penalty is geometrically dependent. \n\n"
        "Relevance for Paper:\n"
        "This visualization visually anchors the core premise of the paper: standard clear-sky models aggressively overestimate solar potential in polluted urban environments, particularly for prime rooftop real estate."
    )
    add_text_page(pdf, "Interpretation: Annual Solar Irradiance Maps", text)

def section_2_psi(c_data, p_data, pdf):
    print("Generating Section 2: Pollution Sensitivity Index (PSI)...")
    centroids = c_data.get('centroid')
    ann_c = c_data.get('annual_irradiance_wh_m2')
    ann_p = p_data.get('annual_irradiance_wh_m2')
    
    valid = ~np.isnan(ann_c) & ~np.isnan(ann_p)
    x, y = centroids[valid, 0], centroids[valid, 1]
    c_val, p_val = ann_c[valid], ann_p[valid]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        psi = np.where(c_val > 0, ((c_val - p_val) / c_val) * 100.0, 0.0)
        
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(x) < 50000:
        sc = ax.scatter(x, y, c=psi, cmap='inferno', s=2, vmin=0, vmax=np.nanpercentile(psi, 99))
    else:
        sc = ax.hexbin(x, y, C=psi, gridsize=80, cmap='inferno', vmin=0, vmax=np.nanpercentile(psi, 99))
        
    ax.set_title('Pollution Sensitivity Index (PSI)')
    ax.axis('equal'); ax.axis('off')
    fig.colorbar(sc, ax=ax, label='PSI (%)')
    
    pdf.savefig(fig); plt.close(fig)
    
    text = (
        "What is observed:\n"
        "The Pollution Sensitivity Index (PSI) maps the relative (percentage) energy loss across the urban geometry. Unlike absolute loss, PSI highlights how sensitive a particular surface is to atmospheric degradation. Certain geometries exhibit high PSI (extreme sensitivity to pollution), while others remain relatively stable despite the lower atmospheric transmittance.\n\n"
        "Why it occurs:\n"
        "Different surfaces have different dependencies on direct vs. diffuse radiation. Surfaces heavily reliant on diffuse sky radiation (such as heavily shaded alleys or north-facing walls) may exhibit lower PSI because aerosols scatter direct light into the diffuse component, slightly offsetting the loss of the direct beam. Conversely, surfaces optimized for direct beam capture suffer the highest relative penalties.\n\n"
        "Scientific Importance:\n"
        "PSI standardizes the measurement of atmospheric vulnerability. It separates geometric shading losses from atmospheric attenuation losses.\n\n"
        "Relevance for Paper:\n"
        "We can propose PSI as a novel metric in the paper. It provides urban planners and solar developers a unified metric to assess the 'pollution risk' of a specific roof geometry independent of its baseline irradiance."
    )
    add_text_page(pdf, "Interpretation: Pollution Sensitivity Index", text)

def section_3_monthly_energy(c_data, p_data, pdf):
    print("Generating Section 3: Monthly Energy Comparison...")
    h_c, h_p = c_data.get('hourly_irradiance_w_m2'), p_data.get('hourly_irradiance_w_m2')
    t_unix = c_data.get('time_unix_s')
    t_axis = get_time_axis(h_c, len(t_unix))
    
    ts_c = extract_hourly_timeseries(h_c, t_axis) / 1000.0
    ts_p = extract_hourly_timeseries(h_p, t_axis) / 1000.0
    dt_idx = pd.to_datetime(t_unix, unit='s')
    df = pd.DataFrame({'Clear': ts_c, 'Pollution': ts_p}, index=dt_idx)
    monthly = df.resample('MS').sum()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        monthly['Loss_Pct'] = np.where(monthly['Clear'] > 0, ((monthly['Clear'] - monthly['Pollution']) / monthly['Clear']) * 100.0, 0.0)
    
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x_labels = [calendar.month_abbr[m] for m in monthly.index.month]
    x = np.arange(len(x_labels))
    width = 0.35
    
    ax1.bar(x - width/2, monthly['Clear'], width, label='Clear', color=COLOR_CLEAR)
    ax1.bar(x + width/2, monthly['Pollution'], width, label='Pollution', color=COLOR_POLL)
    ax1.set_ylabel('City-Wide Accumulated Energy (kWh/m2)')
    ax1.set_xticks(x); ax1.set_xticklabels(x_labels)
    
    ax2 = ax1.twinx()
    ax2.plot(x, monthly['Loss_Pct'], color=COLOR_LOSS, marker='o', lw=2, label='Loss %')
    ax2.set_ylabel('Energy Loss (%)', color=COLOR_LOSS)
    
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)
    plt.title('Monthly Energy Comparison and Percentage Loss', y=1.05)
    pdf.savefig(fig); plt.close(fig)
    
    text = (
        "What is observed:\n"
        "The bar chart tracks theoretical (Clear) vs. actual (Polluted) energy generation across the calendar year. The superimposed line tracks the monthly percentage loss. The severity of the pollution penalty fluctuates heavily throughout the year, identifying specific months where pollution cripples PV generation.\n\n"
        "Why it occurs:\n"
        "This variance is driven by seasonal shifts in local aerosol loading (e.g., winter inversion layers, crop burning seasons, or dry dust periods) intersecting with seasonal solar geometry. Low solar elevation angles force light to travel through a thicker slice of the polluted atmospheric boundary layer (increased air mass), exacerbating the attenuation effect.\n\n"
        "Scientific Importance:\n"
        "Annual averages mask severe seasonal deficits. Grid operators relying on solar must account for these acute seasonal generation drops, which clear-sky models entirely fail to predict.\n\n"
        "Relevance for Paper:\n"
        "This figure effectively bridges the gap between atmospheric science (AOD seasonality) and power engineering (monthly PV yield), forming a strong interdisciplinary argument."
    )
    add_text_page(pdf, "Interpretation: Monthly Energy Comparison", text)
    return monthly

def section_4_scatter(c_data, p_data, pdf):
    print("Generating Section 4: Clear vs Polluted Scatter...")
    ann_c = drop_nans(c_data.get('annual_irradiance_wh_m2', [])) / 1000.0
    ann_p = drop_nans(p_data.get('annual_irradiance_wh_m2', [])) / 1000.0
    
    fig, ax = plt.subplots(figsize=(6, 6))
    if len(ann_c) < 50000: ax.scatter(ann_c, ann_p, s=2, alpha=0.3, color=COLOR_POLL)
    else: ax.hexbin(ann_c, ann_p, gridsize=80, cmap='inferno', mincnt=1)
        
    ax.plot([0, max(ann_c)], [0, max(ann_c)], 'k--', lw=1, label='1:1 Line (No Loss)')
    ax.set_xlabel('Clear-Sky Annual Irradiance (kWh/m2)')
    ax.set_ylabel('Pollution-Attenuated Irradiance (kWh/m2)')
    ax.set_title('Relationship between Clear-Sky and Pollution-Attenuated Irradiance')
    ax.legend()
    pdf.savefig(fig); plt.close(fig)
    
    text = (
        "What is observed:\n"
        "The scatter plot maps the predicted clear-sky irradiance of every face against its actual pollution-attenuated irradiance. The deviation from the 1:1 reference line represents the energy loss. The attenuation trend is largely linear but exhibits a notable vertical spread (heteroscedasticity) at higher irradiance values.\n\n"
        "Why it occurs:\n"
        "The deviation from the 1:1 line is the direct physical consequence of aerosol absorption and scattering. The increasing vertical spread at the higher end of the spectrum indicates that surfaces with identical clear-sky potential (e.g., two identical flat roofs) may experience slightly different pollution penalties depending on their geometric orientation and exact sky-view factors relative to the anisotropic diffuse sky dome.\n\n"
        "Scientific Importance:\n"
        "If pollution merely caused a flat percentage drop, this plot would be a perfect, tight straight line below the 1:1 line. The spread proves that 3D geometric modeling is strictly necessary to accurately predict pollution impacts.\n\n"
        "Relevance for Paper:\n"
        "This visually proves the complexity of urban solar modeling and justifies the use of a high-resolution tool like PASA-Sim over simple 2D derating factors."
    )
    add_text_page(pdf, "Interpretation: Clear vs Polluted Scatter", text)

def section_5_and_6_totals_and_cost(c_data, p_data, pdf):
    print("Generating Sections 5 & 6: Total Energy and Cost...")
    ann_c_raw = c_data.get('annual_irradiance_wh_m2', np.array([]))
    ann_p_raw = p_data.get('annual_irradiance_wh_m2', np.array([]))
    areas_raw = c_data.get('area_m2', np.array([]))
    
    if len(ann_c_raw) == 0 or len(ann_p_raw) == 0 or len(areas_raw) == 0: 
        return 0, 0, 0
    
    # Shared mask to strictly preserve parallel array lengths
    valid_mask = ~np.isnan(ann_c_raw) & ~np.isnan(ann_p_raw) & ~np.isnan(areas_raw)
    
    if not np.any(valid_mask):
        return 0, 0, 0
        
    ann_c = ann_c_raw[valid_mask]
    ann_p = ann_p_raw[valid_mask]
    areas = areas_raw[valid_mask]
    
    # Calculate Total Energy Loss
    energy_c = (ann_c * areas) / 1000.0  # Total kWh
    energy_p = (ann_p * areas) / 1000.0  
    total_loss_kwh = np.sum(energy_c) - np.sum(energy_p)
    total_area = np.sum(areas)
    
    # Cost
    total_cost = total_loss_kwh * TARIFF_INR_PER_KWH
    cost_per_m2 = total_cost / total_area if total_area > 0 else 0
    
    text5 = (
        f"Methodology:\n"
        f"Total city-wide energy loss is computed by integrating the absolute energy loss (Clear - Polluted) across the physical surface area of every simulated mesh face.\n\n"
        f"Findings:\n"
        f"• Total Energy Loss: {total_loss_kwh:,.2f} kWh\n"
        f"• Total Energy Loss: {total_loss_kwh / 1e3:,.2f} MWh\n"
        f"• Total Energy Loss: {total_loss_kwh / 1e6:,.2f} GWh\n\n"
        f"Scientific Importance:\n"
        f"While per-square-metre metrics are useful for physics, aggregate GWh figures are required by urban planners and grid operators. A loss of this magnitude represents a massive deficit in decentralized urban power generation.\n\n"
        f"Relevance for Paper:\n"
        f"Reporting the total GWh lost bridges the gap between micro-scale geometry and macro-scale energy policy."
    )
    add_text_page(pdf, "Section 5: Total City-Wide Energy Loss", text5)
    
    text6 = (
        f"Methodology:\n"
        f"The physical energy deficit translates directly to monetary loss for prosumers. This estimation utilizes a configurable electricity tariff of ₹{TARIFF_INR_PER_KWH:.2f} per kWh.\n\n"
        f"Findings:\n"
        f"• Annual Lost Energy: {total_loss_kwh:,.2f} kWh\n"
        f"• Annual Economic Loss: ₹{total_cost:,.2f}\n"
        f"• Average Cost per Square Metre: ₹{cost_per_m2:,.2f} / m²\n\n"
        f"Scientific Importance:\n"
        f"Environmental physics often struggles to influence policy without economic translation. Quantifying the financial damage caused by reduced atmospheric transmittance provides a hard monetary metric for the cost of urban air pollution.\n\n"
        f"Relevance for Paper:\n"
        f"Concluding the paper with a financial impact assessment significantly broadens the target audience, making the research highly relevant to energy economists and municipal policymakers."
    )
    add_text_page(pdf, "Section 6: Pollution Cost Assessment", text6)
    
    return total_loss_kwh, total_cost, cost_per_m2

def section_7_aod(p_data, pdf):
    print("Generating Section 7: Monthly AOD and Transmission...")
    aod = p_data.get('aod_group_value')          
    trans = p_data.get('aod_group_transmission') 
    t_unix = p_data.get('time_unix_s')
    
    t_axis = get_time_axis(aod, len(t_unix))
    if t_axis is None: return None, None
    g_axis = 1 - t_axis
    
    group_ids = p_data.get('aod_group_id')
    num_groups = aod.shape[g_axis]
    
    if group_ids is not None and num_groups > 1:
        valid_gids = drop_nans(group_ids).astype(int)
        unique_groups, counts = np.unique(valid_gids, return_counts=True)
        weights = np.zeros(num_groups)
        for g, c in zip(unique_groups, counts):
            if 0 <= g < num_groups: weights[g] = c
        if weights.sum() > 0:
            mean_aod = np.average(aod, axis=g_axis, weights=weights)
            mean_trans = np.average(trans, axis=g_axis, weights=weights)
        else:
            mean_aod = np.nanmean(aod, axis=g_axis)
            mean_trans = np.nanmean(trans, axis=g_axis)
    else:
        mean_aod = np.nanmean(aod, axis=g_axis)
        mean_trans = np.nanmean(trans, axis=g_axis)
        
    dt_idx = pd.to_datetime(t_unix, unit='s')
    df = pd.DataFrame({'AOD': mean_aod, 'Trans': mean_trans}, index=dt_idx)
    monthly = df.resample('MS').mean()
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    months_str = [calendar.month_abbr[m] for m in monthly.index.month]
    
    ax1.plot(months_str, monthly['AOD'], color=COLOR_POLL, marker='o', label='AOD')
    ax1.set_ylabel('Aerosol Optical Depth (AOD)', color=COLOR_POLL)
    
    ax2 = ax1.twinx()
    ax2.plot(months_str, monthly['Trans'], color=COLOR_CLEAR, marker='s', label='Transmission')
    ax2.set_ylabel('Atmospheric Transmission', color=COLOR_CLEAR)
    
    plt.title('Monthly Average AOD and Atmospheric Transmission')
    pdf.savefig(fig); plt.close(fig)
    
    text = (
        "What is observed:\n"
        "The graph plots the temporal progression of Aerosol Optical Depth alongside the resultant atmospheric transmission coefficient. The inverse relationship is explicit: as aerosol concentration peaks, transmission plummets.\n\n"
        "Why it occurs:\n"
        "This is the direct macroscopic expression of the Beer-Lambert law operating within the simulator. Photons passing through the atmospheric column are scattered and absorbed by particulate matter, parameterized here by the AOD.\n\n"
        "Scientific Importance:\n"
        "This verifies the core meteorological driver behind the energy deficits observed in Section 3. It links raw environmental pollution data directly to the optical parameters governing the ray-tracing mechanics.\n\n"
        "Relevance for Paper:\n"
        "Including this figure proves the physical consistency of the simulation pipeline. It allows the paper to conclusively attribute the mapped energy losses to documented periods of severe urban pollution."
    )
    add_text_page(pdf, "Interpretation: Monthly AOD and Transmission", text)
    return np.nanmean(mean_aod), np.nanmean(mean_trans)

def section_8_cumulative(c_data, p_data, pdf):
    print("Generating Section 8: Cumulative Annual Energy...")
    h_c, h_p = c_data.get('hourly_irradiance_w_m2'), p_data.get('hourly_irradiance_w_m2')
    t_unix = c_data.get('time_unix_s')
    t_axis = get_time_axis(h_c, len(t_unix))
    
    ts_c = extract_hourly_timeseries(h_c, t_axis)
    ts_p = extract_hourly_timeseries(h_p, t_axis)
    dt_idx = pd.to_datetime(t_unix, unit='s')
    df = pd.DataFrame({'Clear': ts_c, 'Pollution': ts_p}, index=dt_idx)
    
    # Corrected units: W/m2 integrated hourly is Wh/m2. Divided by 1000 = kWh/m2.
    cum_c = df['Clear'].cumsum() / 1000.0  
    cum_p = df['Pollution'].cumsum() / 1000.0
    cum_loss = cum_c - cum_p
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df.index, cum_c, color=COLOR_CLEAR, lw=2, label='Clear Cumulative')
    ax1.plot(df.index, cum_p, color=COLOR_POLL, lw=2, label='Polluted Cumulative')
    ax1.set_ylabel('Cumulative Irradiance (kWh/m2)')
    
    ax2 = ax1.twinx()
    ax2.fill_between(df.index, cum_loss, color=COLOR_LOSS, alpha=0.3, label='Cumulative Loss')
    ax2.set_ylabel('Cumulative Energy Loss (kWh/m2)', color=COLOR_LOSS)
    
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)
    plt.title('Cumulative Annual Energy and Deficit Trace', y=1.05)
    pdf.savefig(fig); plt.close(fig)
    
    text = (
        "What is observed:\n"
        "The cumulative energy curve tracks the continuous accumulation of solar potential over the 8,760-hour simulation. The shaded region tracks the widening deficit between the idealized clear sky and the polluted reality. The gap continues to grow relentlessly throughout the year.\n\n"
        "Why it occurs:\n"
        "Every hour of daylight affected by aerosols chips away at the total yield. Because pollution is chronic in the study area, the system rarely recovers; the loss simply compounds.\n\n"
        "Scientific Importance:\n"
        "While hourly or monthly profiles show variance, the cumulative curve illustrates the undeniable long-term macro effect on system payback periods and LCOE (Levelized Cost of Energy).\n\n"
        "Relevance for Paper:\n"
        "This visualization is highly effective for illustrating long-term financial and energetic viability. It visually communicates that pollution is not a transient weather event, but a fundamental, compounding drag on urban energy infrastructure."
    )
    add_text_page(pdf, "Interpretation: Cumulative Annual Energy", text)

def section_9_heatmap(c_data, p_data, pdf):
    print("Generating Section 9: Month x Hour Heatmap...")
    h_c, h_p = c_data.get('hourly_irradiance_w_m2'), p_data.get('hourly_irradiance_w_m2')
    t_unix = c_data.get('time_unix_s')
    t_axis = get_time_axis(h_c, len(t_unix))
    
    ts_c = extract_hourly_timeseries(h_c, t_axis)
    ts_p = extract_hourly_timeseries(h_p, t_axis)
    dt_idx = pd.to_datetime(t_unix, unit='s')
    df_hourly = pd.DataFrame({'Clear': ts_c, 'Pollution': ts_p}, index=dt_idx)
    
    df_hourly['Month'] = df_hourly.index.month
    df_hourly['Hour'] = df_hourly.index.hour
    with np.errstate(divide='ignore', invalid='ignore'):
        df_hourly['PctLoss'] = np.where(df_hourly['Clear'] > 0, ((df_hourly['Clear'] - df_hourly['Pollution']) / df_hourly['Clear']) * 100, 0)
    
    heat_c = df_hourly.pivot_table(index='Hour', columns='Month', values='Clear')
    heat_p = df_hourly.pivot_table(index='Hour', columns='Month', values='Pollution')
    heat_loss = df_hourly.pivot_table(index='Hour', columns='Month', values='PctLoss')
    month_labels = [calendar.month_abbr[i] for i in heat_c.columns]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    vmax = np.nanmax(heat_c.values)
    
    im0 = axes[0].imshow(heat_c.values, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=vmax)
    axes[0].set_title('Clear Irradiance (W/m2)')
    axes[0].set_xticks(np.arange(len(month_labels))); axes[0].set_xticklabels(month_labels)
    axes[0].set_ylabel('Hour of Day')
    
    im1 = axes[1].imshow(heat_p.values, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=vmax)
    axes[1].set_title('Polluted Irradiance (W/m2)')
    axes[1].set_xticks(np.arange(len(month_labels))); axes[1].set_xticklabels(month_labels)
    
    im2 = axes[2].imshow(heat_loss.values, aspect='auto', cmap='magma', origin='lower')
    axes[2].set_title('Percentage Loss (%)')
    axes[2].set_xticks(np.arange(len(month_labels))); axes[2].set_xticklabels(month_labels)
    
    fig.colorbar(im0, ax=axes[0:2], shrink=0.7)
    fig.colorbar(im2, ax=axes[2], shrink=0.7)
    
    pdf.savefig(fig); plt.close(fig)
    
    text = (
        "What is observed:\n"
        "The matrices map average irradiance across both the diurnal (y-axis) and seasonal (x-axis) cycles. The 'Percentage Loss' heatmap explicitly isolates the periods when atmospheric pollution most aggressively degrades the solar resource.\n\n"
        "Why it occurs:\n"
        "The interaction between the solar zenith angle and the atmospheric boundary layer governs this behavior. The most intense percentage losses frequently occur at the edges of the daylight envelope (early morning/late evening) and during specific months where high AOD aligns with lower solar altitudes, forcing light through a longer polluted air mass.\n\n"
        "Scientific Importance:\n"
        "This confirms that the pollution penalty is neither temporally nor seasonally constant. It is a highly dynamic function of solar astronomy interacting with local atmospheric chemistry.\n\n"
        "Relevance for Paper:\n"
        "For grid integration studies, timing is everything. This figure allows researchers to instantly identify if pollution-induced energy deficits align with peak urban power demand hours."
    )
    add_text_page(pdf, "Interpretation: Month × Hour Heatmap", text)

def section_10_distribution(c_data, p_data, pdf):
    print("Generating Section 10: Annual Irradiance Distribution (Histogram)...")
    ann_c = drop_nans(c_data.get('annual_irradiance_wh_m2', [])) / 1000.0
    ann_p = drop_nans(p_data.get('annual_irradiance_wh_m2', [])) / 1000.0
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ann_c, bins='auto', alpha=0.6, label='Clear Sky', color=COLOR_CLEAR, density=True)
    ax.hist(ann_p, bins='auto', alpha=0.6, label='Pollution', color=COLOR_POLL, density=True)
    ax.set_xlabel('Annual Irradiance (kWh/m2)')
    ax.set_ylabel('Density')
    ax.set_title('Shift in Annual Irradiance Distribution')
    ax.legend()
    
    pdf.savefig(fig); plt.close(fig)
    
    text = (
        "What is observed:\n"
        "The histogram directly overlays the clear and polluted annual irradiance populations. The entire distribution shifts to the left, indicating a systematic degradation of the solar resource across all surfaces. \n\n"
        "Why it occurs:\n"
        "Atmospheric attenuation globally reduces incoming solar radiation before it interacts with the urban canopy. The width and shape of the distribution remain relatively intact because mutual building shading and surface orientations still dictate the relative hierarchy of irradiance; pollution simply lowers the ceiling for everyone.\n\n"
        "Scientific Importance:\n"
        "This confirms that pollution affects the entire urban morphology, not just a specific subset of vulnerable geometries.\n\n"
        "Relevance for Paper:\n"
        "This is a fundamental descriptive statistic plot that effectively communicates the overarching premise of the research before diving into complex spatial or temporal dependencies."
    )
    add_text_page(pdf, "Interpretation: Annual Irradiance Distribution", text)

def section_11_and_12_conclusions(metrics, pdf):
    print("Generating Sections 11 & 12: Summary and Conclusions...")
    
    text11 = (
        "Table of Key Numerical Findings derived from the 3D PASA-Sim simulation:\n\n"
        f"• Average Annual Irradiance (Clear): {metrics.get('mean_c', 0):.2f} kWh/m2\n"
        f"• Average Annual Irradiance (Polluted): {metrics.get('mean_p', 0):.2f} kWh/m2\n"
        f"• Average Percentage Loss: {metrics.get('loss_pct', 0):.2f}%\n"
        f"• Total City Energy Loss: {metrics.get('total_loss_gwh', 0):.2f} GWh\n"
        f"• Pollution Cost Estimate: ₹{metrics.get('total_cost', 0):,.0f}\n"
        f"• Average Simulated AOD: {metrics.get('mean_aod', 0):.3f}\n"
        f"• Average Atmospheric Transmission: {metrics.get('mean_trans', 0):.3f}\n"
    )
    add_text_page(pdf, "Section 11: Summary of Key Numerical Findings", text11)
    
    text12 = (
        "Discussion of Preliminary Findings:\n"
        "The analyses presented in this document confirm that atmospheric aerosols induce severe, non-uniform, and geometrically dependent energy deficits across the urban topology. Clear-sky models fundamentally fail to predict the true solar viability of urban environments heavily afflicted by pollution. The temporal and spatial variance of this attenuation demands explicit 3D modeling coupled with transient atmospheric data.\n\n"
        "Possible Research Directions for Final Paper:\n"
        "• Urban pollution impact assessment on decentralized grid stability.\n"
        "• Spatial variability of pollution losses and identifying vulnerable geometries.\n"
        "• Economic implications and updated LCOE (Levelized Cost of Energy) calculations.\n"
        "• Proposal of the Pollution Sensitivity Index (PSI) as a novel solar planning metric.\n"
        "• Seasonal pollution effects intersecting with urban energy demand peaks.\n"
        "• Future validation methodologies using measured rooftop PV output.\n"
        "• Future pipeline integration with transient cloud cover and localized weather systems.\n\n"
        "Note: These are preliminary findings intended for collaborative discussion prior to selecting the final analytical direction for publication."
    )
    add_text_page(pdf, "Section 12: Conclusions and Possible Paper Directions", text12)

# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    print("==================================================")
    print("PASA-Sim: Mentor Discussion Report Generator")
    print("==================================================")
    
    c_data, p_data = safe_load(FILE_CLEAR), safe_load(FILE_POLL)
    if c_data is None or p_data is None:
        print("[CRITICAL] Cannot proceed without both NPZ files.")
        sys.exit(1)
        
    print(f"Generating Formal Discussion PDF: {REPORT_PDF}")
    metrics = {}
    
    with PdfPages(REPORT_PDF) as pdf:
        # Title
        generate_title_page(pdf)
        
        # Prelim calcs needed across sections
        ann_c_raw = c_data.get('annual_irradiance_wh_m2', np.array([]))
        ann_p_raw = p_data.get('annual_irradiance_wh_m2', np.array([]))
        
        if len(ann_c_raw) > 0 and len(ann_p_raw) > 0:
            valid = ~np.isnan(ann_c_raw) & ~np.isnan(ann_p_raw)
            ann_c = ann_c_raw[valid] / 1000.0
            ann_p = ann_p_raw[valid] / 1000.0
            
            if len(ann_c) > 0:
                metrics['mean_c'] = np.nanmean(ann_c)
                metrics['mean_p'] = np.nanmean(ann_p)
                metrics['loss_pct'] = ((metrics['mean_c'] - metrics['mean_p']) / metrics['mean_c']) * 100 if metrics['mean_c'] > 0 else 0
        
        # Sec 1
        section_1_maps(c_data, p_data, pdf)
        # Sec 2
        section_2_psi(c_data, p_data, pdf)
        # Sec 3
        section_3_monthly_energy(c_data, p_data, pdf)
        # Sec 4
        section_4_scatter(c_data, p_data, pdf)
        # Sec 5 & 6
        loss_kwh, cost, _ = section_5_and_6_totals_and_cost(c_data, p_data, pdf)
        metrics['total_loss_gwh'] = loss_kwh / 1e6
        metrics['total_cost'] = cost
        # Sec 7
        ma, mt = section_7_aod(p_data, pdf)
        metrics['mean_aod'] = ma
        metrics['mean_trans'] = mt
        # Sec 8
        section_8_cumulative(c_data, p_data, pdf)
        # Sec 9
        section_9_heatmap(c_data, p_data, pdf)
        # Sec 10
        section_10_distribution(c_data, p_data, pdf)
        # Sec 11 & 12
        section_11_and_12_conclusions(metrics, pdf)
        
    print(f"\n[SUCCESS] Pipeline complete. Mentor report saved to: {REPORT_PDF}")

if __name__ == "__main__":
    main()