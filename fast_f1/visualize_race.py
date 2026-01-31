"""
Create comprehensive visualizations from FastF1 race data CSV files.

This script reads the exported CSV files and generates various plots
to analyze the race performance, strategies, and telemetry.
"""

import os
import glob
import warnings
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')

# F1 team colors (2024/2025 season)
TEAM_COLORS = {
    'Red Bull Racing': '#3671C6',
    'Ferrari': '#E8002D',
    'McLaren': '#FF8000',
    'Mercedes': '#27F4D2',
    'Aston Martin': '#229971',
    'Alpine': '#0093CC',
    'Williams': '#64C4FF',
    'RB': '#6692FF',
    'Kick Sauber': '#52E252',
    'Haas F1 Team': '#B6BABD',
    # Fallback
    'default': '#888888'
}

# Tire compound colors (official F1 colors, more vibrant)
COMPOUND_COLORS = {
    'SOFT': '#E8002D',      # Pirelli Red
    'MEDIUM': '#FFD700',    # Golden Yellow
    'HARD': '#F0F0F0',      # Light gray/white
    'INTERMEDIATE': '#39B54A',  # Green
    'WET': '#00AEEF',       # Blue
    'UNKNOWN': '#888888',
}

# For better visibility on plots, use these alternate colors
COMPOUND_PLOT_COLORS = {
    'SOFT': '#E8002D',      # Red
    'MEDIUM': '#F5A623',    # Orange-yellow (more visible than pure yellow)
    'HARD': '#4A90D9',      # Steel blue (much more visible than white/gray)
    'INTERMEDIATE': '#2ECC71',  # Emerald green
    'WET': '#3498DB',       # Bright blue
    'UNKNOWN': '#95A5A6',
}


def get_latest_data_dir():
    """Find the most recent data directory."""
    if not os.path.exists(DATA_DIR):
        return None
    
    data_dirs = [d for d in os.listdir(DATA_DIR) 
                 if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    if not data_dirs:
        return None
    
    # Sort by name (which includes date)
    data_dirs.sort(reverse=True)
    return os.path.join(DATA_DIR, data_dirs[0])


def load_csv(data_dir, filename):
    """Load a CSV file from the data directory."""
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None


def get_team_color(team_name):
    """Get the color for a team."""
    for team, color in TEAM_COLORS.items():
        if team.lower() in str(team_name).lower() or str(team_name).lower() in team.lower():
            return color
    return TEAM_COLORS['default']


def plot_01_race_summary(data_dir, plots_dir, event_info):
    """Create a race summary infographic."""
    results_df = load_csv(data_dir, 'session_results.csv')
    if results_df is None or results_df.empty:
        print("  ⚠️  No results data for race summary")
        return False
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    # Title
    race_name = event_info['EventName'].iloc[0] if event_info is not None else "Race"
    race_date = event_info['EventDate'].iloc[0] if event_info is not None else ""
    
    fig.suptitle(f"🏁 {race_name}", fontsize=24, fontweight='bold', y=0.98)
    ax.text(0.5, 0.95, f"Race Summary - {race_date}", ha='center', va='top', 
            fontsize=14, transform=ax.transAxes, color='gray')
    
    # Podium section
    ax.text(0.5, 0.85, "🏆 PODIUM", ha='center', va='top', fontsize=18, 
            fontweight='bold', transform=ax.transAxes)
    
    positions = ['Position', 'ClassifiedPosition']
    pos_col = None
    for col in positions:
        if col in results_df.columns:
            pos_col = col
            break
    
    if pos_col:
        results_df[pos_col] = pd.to_numeric(results_df[pos_col], errors='coerce')
        top_3 = results_df.nsmallest(3, pos_col)
        
        medals = ['🥇', '🥈', '🥉']
        for i, (_, row) in enumerate(top_3.iterrows()):
            driver = row.get('Abbreviation', row.get('Driver', f"P{i+1}"))
            team = row.get('TeamName', '')
            y_pos = 0.75 - i * 0.08
            ax.text(0.5, y_pos, f"{medals[i]} {driver} - {team}", 
                    ha='center', va='top', fontsize=14, transform=ax.transAxes)
    
    # Stats section
    laps_df = load_csv(data_dir, 'laps.csv')
    if laps_df is not None and not laps_df.empty:
        ax.text(0.25, 0.45, "📊 RACE STATS", ha='center', va='top', fontsize=16, 
                fontweight='bold', transform=ax.transAxes)
        
        # Fastest lap
        if 'LapTime_seconds' in laps_df.columns:
            valid_laps = laps_df[laps_df['LapTime_seconds'] > 0]
            if not valid_laps.empty:
                fastest_idx = valid_laps['LapTime_seconds'].idxmin()
                fastest_lap = valid_laps.loc[fastest_idx]
                driver = fastest_lap.get('Driver', 'Unknown')
                lap_time = fastest_lap['LapTime_seconds']
                mins = int(lap_time // 60)
                secs = lap_time % 60
                ax.text(0.25, 0.38, f"⚡ Fastest Lap: {driver}", 
                        ha='center', va='top', fontsize=12, transform=ax.transAxes)
                ax.text(0.25, 0.33, f"   {mins}:{secs:06.3f}", 
                        ha='center', va='top', fontsize=11, transform=ax.transAxes, color='gray')
        
        # Total laps
        total_laps = laps_df['LapNumber'].max() if 'LapNumber' in laps_df.columns else 'N/A'
        ax.text(0.25, 0.25, f"🔄 Total Laps: {total_laps}", 
                ha='center', va='top', fontsize=12, transform=ax.transAxes)
    
    # Weather section
    weather_df = load_csv(data_dir, 'weather.csv')
    if weather_df is not None and not weather_df.empty:
        ax.text(0.75, 0.45, "🌤️ WEATHER", ha='center', va='top', fontsize=16, 
                fontweight='bold', transform=ax.transAxes)
        
        if 'AirTemp' in weather_df.columns:
            avg_air_temp = weather_df['AirTemp'].mean()
            ax.text(0.75, 0.38, f"🌡️ Air Temp: {avg_air_temp:.1f}°C", 
                    ha='center', va='top', fontsize=12, transform=ax.transAxes)
        
        if 'TrackTemp' in weather_df.columns:
            avg_track_temp = weather_df['TrackTemp'].mean()
            ax.text(0.75, 0.33, f"🛣️ Track Temp: {avg_track_temp:.1f}°C", 
                    ha='center', va='top', fontsize=12, transform=ax.transAxes)
        
        if 'Rainfall' in weather_df.columns:
            had_rain = weather_df['Rainfall'].any()
            rain_text = "🌧️ Rain: Yes" if had_rain else "☀️ Rain: No"
            ax.text(0.75, 0.28, rain_text, 
                    ha='center', va='top', fontsize=12, transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '01_race_summary.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    return True


def plot_02_final_positions(data_dir, plots_dir):
    """Create a bar chart of final positions with team colors."""
    results_df = load_csv(data_dir, 'session_results.csv')
    
    if results_df is None or results_df.empty:
        print("  ⚠️  No results data for final positions")
        return False
    
    # Get position column
    pos_col = 'Position' if 'Position' in results_df.columns else 'ClassifiedPosition'
    if pos_col not in results_df.columns:
        print("  ⚠️  No position data")
        return False
    
    results_df[pos_col] = pd.to_numeric(results_df[pos_col], errors='coerce')
    results_df = results_df.dropna(subset=[pos_col])
    results_df = results_df.sort_values(pos_col)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get driver abbreviations - use Abbreviation column directly from results
    abbrev_col = 'Abbreviation' if 'Abbreviation' in results_df.columns else 'DriverNumber'
    drivers = results_df[abbrev_col].astype(str).values[:20]
    positions = results_df[pos_col].values[:20]
    teams = results_df['TeamName'].fillna('Unknown').values[:20] if 'TeamName' in results_df.columns else ['Unknown'] * len(drivers)
    
    colors = [get_team_color(team) for team in teams]
    
    y_pos = np.arange(len(drivers))
    bars = ax.barh(y_pos, 21 - positions, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"P{int(p)}: {d}" for p, d in zip(positions, drivers)])
    ax.invert_yaxis()
    ax.set_xlabel('Finishing Order (higher = better)', fontsize=12)
    ax.set_title('Race Classification', fontsize=16, fontweight='bold', pad=20)
    
    # Add team legend
    unique_teams = list(set(teams))
    legend_patches = [mpatches.Patch(color=get_team_color(t), label=t) for t in unique_teams[:10]]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '02_final_positions.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    return True


def plot_03_lap_times(data_dir, plots_dir):
    """Create a lap time comparison chart for top drivers."""
    laps_df = load_csv(data_dir, 'laps.csv')
    drivers_df = load_csv(data_dir, 'drivers.csv')
    
    if laps_df is None or laps_df.empty:
        print("  ⚠️  No lap data for lap times chart")
        return False
    
    if 'LapTime_seconds' not in laps_df.columns:
        print("  ⚠️  No lap time data")
        return False
    
    # Merge with drivers for team info
    if drivers_df is not None and 'Driver' in laps_df.columns:
        driver_map = dict(zip(drivers_df['DriverNumber'].astype(str), drivers_df['TeamName']))
        laps_df['TeamName'] = laps_df['Driver'].astype(str).map(driver_map)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Get top 8 drivers by average lap time
    valid_laps = laps_df[(laps_df['LapTime_seconds'] > 0) & (laps_df['LapTime_seconds'] < 200)]
    driver_avg = valid_laps.groupby('Driver')['LapTime_seconds'].mean().nsmallest(8)
    top_drivers = driver_avg.index.tolist()
    
    for driver in top_drivers:
        driver_laps = valid_laps[valid_laps['Driver'] == driver]
        if not driver_laps.empty:
            team = driver_laps['TeamName'].iloc[0] if 'TeamName' in driver_laps.columns else 'Unknown'
            color = get_team_color(team)
            ax.plot(driver_laps['LapNumber'], driver_laps['LapTime_seconds'], 
                    marker='o', markersize=3, label=str(driver), color=color, alpha=0.8)
    
    ax.set_xlabel('Lap Number', fontsize=12)
    ax.set_ylabel('Lap Time (seconds)', fontsize=12)
    ax.set_title('Lap Times Comparison - Top 8 Drivers', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits to exclude outliers
    median_time = valid_laps['LapTime_seconds'].median()
    ax.set_ylim(median_time - 10, median_time + 20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '03_lap_times.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    return True


def plot_04_sector_times_heatmap(data_dir, plots_dir):
    """Create a heatmap of sector times."""
    laps_df = load_csv(data_dir, 'laps.csv')
    
    if laps_df is None or laps_df.empty:
        print("  ⚠️  No lap data for sector times")
        return False
    
    sector_cols = ['Sector1Time_seconds', 'Sector2Time_seconds', 'Sector3Time_seconds']
    if not all(col in laps_df.columns for col in sector_cols):
        print("  ⚠️  No sector time data")
        return False
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get best sector times per driver
    sector_data = []
    for driver in laps_df['Driver'].unique():
        driver_laps = laps_df[laps_df['Driver'] == driver]
        valid_laps = driver_laps[(driver_laps['Sector1Time_seconds'] > 0) & 
                                  (driver_laps['Sector2Time_seconds'] > 0) & 
                                  (driver_laps['Sector3Time_seconds'] > 0)]
        if not valid_laps.empty:
            sector_data.append({
                'Driver': driver,
                'S1': valid_laps['Sector1Time_seconds'].min(),
                'S2': valid_laps['Sector2Time_seconds'].min(),
                'S3': valid_laps['Sector3Time_seconds'].min(),
            })
    
    if not sector_data:
        print("  ⚠️  No valid sector data")
        return False
    
    sector_df = pd.DataFrame(sector_data).set_index('Driver')
    
    # Normalize for heatmap (relative to best in each sector)
    sector_normalized = sector_df.copy()
    for col in sector_df.columns:
        min_val = sector_df[col].min()
        sector_normalized[col] = sector_df[col] - min_val
    
    # Sort by total time
    sector_normalized['Total'] = sector_normalized.sum(axis=1)
    sector_normalized = sector_normalized.sort_values('Total').drop('Total', axis=1)
    
    sns.heatmap(sector_normalized, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                ax=ax, cbar_kws={'label': 'Delta to Best (s)'})
    
    ax.set_title('Best Sector Times - Delta to Fastest', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Sector', fontsize=12)
    ax.set_ylabel('Driver', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '04_sector_times_heatmap.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    return True


def plot_05_tire_strategy(data_dir, plots_dir):
    """Create a tire strategy visualization."""
    stints_df = load_csv(data_dir, 'stints.csv')
    results_df = load_csv(data_dir, 'session_results.csv')
    
    if stints_df is None or stints_df.empty:
        print("  ⚠️  No stint data for tire strategy")
        return False
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Get drivers in finishing order if available
    drivers = stints_df['Driver'].unique()
    if results_df is not None and not results_df.empty:
        pos_col = 'Position' if 'Position' in results_df.columns else 'ClassifiedPosition'
        if pos_col in results_df.columns:
            results_df[pos_col] = pd.to_numeric(results_df[pos_col], errors='coerce')
            ordered_results = results_df.dropna(subset=[pos_col]).sort_values(pos_col)
            if 'Driver' in ordered_results.columns:
                ordered_drivers = ordered_results['Driver'].astype(str).tolist()
                drivers = [d for d in ordered_drivers if str(d) in [str(x) for x in stints_df['Driver'].unique()]]
    
    # Plot each driver's stints
    for i, driver in enumerate(drivers[:20]):  # Limit to 20 drivers
        driver_stints = stints_df[stints_df['Driver'].astype(str) == str(driver)]
        
        for _, stint in driver_stints.iterrows():
            compound = str(stint.get('Compound', 'UNKNOWN')).upper()
            color = COMPOUND_COLORS.get(compound, COMPOUND_COLORS['UNKNOWN'])
            
            lap_start = stint.get('LapStart', 0)
            lap_end = stint.get('LapEnd', lap_start + stint.get('NumLaps', 1))
            
            ax.barh(i, lap_end - lap_start, left=lap_start, 
                    color=color, edgecolor='black', linewidth=0.5, height=0.8)
    
    ax.set_yticks(range(len(drivers[:20])))
    ax.set_yticklabels([f"P{i+1}: {d}" for i, d in enumerate(drivers[:20])])
    ax.invert_yaxis()
    ax.set_xlabel('Lap Number', fontsize=12)
    ax.set_title('Tire Strategy', fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_patches = [mpatches.Patch(color=color, label=compound) 
                      for compound, color in COMPOUND_COLORS.items() if compound != 'UNKNOWN']
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '05_tire_strategy.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    return True


def plot_06_weather_evolution(data_dir, plots_dir):
    """Create a weather evolution chart."""
    weather_df = load_csv(data_dir, 'weather.csv')
    
    if weather_df is None or weather_df.empty:
        print("  ⚠️  No weather data")
        return False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x_col = 'Time' if 'Time' in weather_df.columns else weather_df.index
    if isinstance(x_col, str):
        x = pd.to_datetime(weather_df[x_col], errors='coerce')
        if x.isna().all():
            x = range(len(weather_df))
    else:
        x = range(len(weather_df))
    
    # Air Temperature
    if 'AirTemp' in weather_df.columns:
        axes[0, 0].plot(x, weather_df['AirTemp'], color='#FF6B6B', linewidth=2)
        axes[0, 0].set_title('Air Temperature', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('°C')
        axes[0, 0].fill_between(x, weather_df['AirTemp'], alpha=0.3, color='#FF6B6B')
    
    # Track Temperature
    if 'TrackTemp' in weather_df.columns:
        axes[0, 1].plot(x, weather_df['TrackTemp'], color='#4ECDC4', linewidth=2)
        axes[0, 1].set_title('Track Temperature', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('°C')
        axes[0, 1].fill_between(x, weather_df['TrackTemp'], alpha=0.3, color='#4ECDC4')
    
    # Humidity
    if 'Humidity' in weather_df.columns:
        axes[1, 0].plot(x, weather_df['Humidity'], color='#45B7D1', linewidth=2)
        axes[1, 0].set_title('Humidity', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('%')
        axes[1, 0].fill_between(x, weather_df['Humidity'], alpha=0.3, color='#45B7D1')
    
    # Wind Speed
    if 'WindSpeed' in weather_df.columns:
        axes[1, 1].plot(x, weather_df['WindSpeed'], color='#96CEB4', linewidth=2)
        axes[1, 1].set_title('Wind Speed', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('m/s')
        axes[1, 1].fill_between(x, weather_df['WindSpeed'], alpha=0.3, color='#96CEB4')
    
    fig.suptitle('Weather Evolution During Race', fontsize=16, fontweight='bold', y=1.02)
    
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '06_weather_evolution.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    return True


def plot_07_pace_distribution(data_dir, plots_dir):
    """Create a violin plot of lap time distributions."""
    laps_df = load_csv(data_dir, 'laps.csv')
    drivers_df = load_csv(data_dir, 'drivers.csv')
    
    if laps_df is None or laps_df.empty:
        print("  ⚠️  No lap data for pace distribution")
        return False
    
    if 'LapTime_seconds' not in laps_df.columns:
        print("  ⚠️  No lap time data")
        return False
    
    # Filter valid laps
    valid_laps = laps_df[(laps_df['LapTime_seconds'] > 0) & (laps_df['LapTime_seconds'] < 200)]
    
    # Exclude pit laps and first lap
    if 'IsPersonalBest' in valid_laps.columns:
        pass  # Use all laps
    
    # Remove outliers (pit laps typically)
    median_time = valid_laps['LapTime_seconds'].median()
    valid_laps = valid_laps[valid_laps['LapTime_seconds'] < median_time * 1.15]
    
    # Get top 10 drivers by median pace
    driver_median = valid_laps.groupby('Driver')['LapTime_seconds'].median().nsmallest(10)
    top_drivers = driver_median.index.tolist()
    
    plot_data = valid_laps[valid_laps['Driver'].isin(top_drivers)]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create violin plot
    driver_order = top_drivers
    sns.violinplot(data=plot_data, x='Driver', y='LapTime_seconds', 
                   order=driver_order, ax=ax, inner='box', palette='husl')
    
    ax.set_xlabel('Driver', fontsize=12)
    ax.set_ylabel('Lap Time (seconds)', fontsize=12)
    ax.set_title('Race Pace Distribution - Top 10 Drivers', fontsize=16, fontweight='bold', pad=20)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '07_pace_distribution.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    return True


def plot_08_telemetry_comparison(data_dir, plots_dir):
    """Create telemetry comparison chart for fastest laps."""
    telemetry_df = load_csv(data_dir, 'telemetry_fastest_laps.csv')
    
    if telemetry_df is None or telemetry_df.empty:
        print("  ⚠️  No telemetry data")
        return False
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    
    drivers = telemetry_df['Driver'].unique()[:4]  # Top 4 drivers
    colors = plt.cm.Set1(np.linspace(0, 1, len(drivers)))
    
    for i, driver in enumerate(drivers):
        driver_tel = telemetry_df[telemetry_df['Driver'] == driver]
        
        x = driver_tel.get('Distance', driver_tel.get('Time_seconds', range(len(driver_tel))))
        
        # Speed
        if 'Speed' in driver_tel.columns:
            axes[0].plot(x, driver_tel['Speed'], label=str(driver), color=colors[i], alpha=0.8)
        
        # Throttle
        if 'Throttle' in driver_tel.columns:
            axes[1].plot(x, driver_tel['Throttle'], label=str(driver), color=colors[i], alpha=0.8)
        
        # Brake
        if 'Brake' in driver_tel.columns:
            axes[2].plot(x, driver_tel['Brake'], label=str(driver), color=colors[i], alpha=0.8)
        
        # Gear
        if 'nGear' in driver_tel.columns:
            axes[3].plot(x, driver_tel['nGear'], label=str(driver), color=colors[i], alpha=0.8)
    
    axes[0].set_ylabel('Speed (km/h)')
    axes[0].set_title('Speed', fontsize=11)
    axes[0].legend(loc='upper right')
    
    axes[1].set_ylabel('Throttle (%)')
    axes[1].set_title('Throttle', fontsize=11)
    
    axes[2].set_ylabel('Brake')
    axes[2].set_title('Brake', fontsize=11)
    
    axes[3].set_ylabel('Gear')
    axes[3].set_title('Gear', fontsize=11)
    axes[3].set_xlabel('Distance (m)' if 'Distance' in telemetry_df.columns else 'Time (s)')
    
    fig.suptitle('Telemetry Comparison - Fastest Laps', fontsize=16, fontweight='bold', y=1.02)
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '08_telemetry_comparison.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    return True


def plot_09_lap_time_evolution(data_dir, plots_dir):
    """Create lap time evolution showing tire degradation with separate driver subplots."""
    laps_df = load_csv(data_dir, 'laps.csv')
    drivers_df = load_csv(data_dir, 'drivers.csv')
    results_df = load_csv(data_dir, 'session_results.csv')
    
    if laps_df is None or laps_df.empty:
        print("  ⚠️  No lap data for lap time evolution")
        return False
    
    if 'LapTime_seconds' not in laps_df.columns:
        print("  ⚠️  No lap time data")
        return False
    
    # Get top 6 drivers by finishing position or average pace
    valid_laps = laps_df[(laps_df['LapTime_seconds'] > 0) & (laps_df['LapTime_seconds'] < 200)]
    median_time = valid_laps['LapTime_seconds'].median()
    valid_laps = valid_laps[valid_laps['LapTime_seconds'] < median_time * 1.15].copy()
    
    # Get driver order (by finishing position if available)
    if results_df is not None and not results_df.empty:
        pos_col = 'Position' if 'Position' in results_df.columns else 'ClassifiedPosition'
        if pos_col in results_df.columns:
            results_df[pos_col] = pd.to_numeric(results_df[pos_col], errors='coerce')
            ordered = results_df.dropna(subset=[pos_col]).sort_values(pos_col)
            # Map driver number to abbreviation
            if 'Abbreviation' in ordered.columns:
                top_drivers = ordered['Abbreviation'].head(6).tolist()
            else:
                top_drivers = ordered['DriverNumber'].astype(str).head(6).tolist()
        else:
            driver_avg = valid_laps.groupby('Driver')['LapTime_seconds'].mean().nsmallest(6)
            top_drivers = driver_avg.index.tolist()
    else:
        driver_avg = valid_laps.groupby('Driver')['LapTime_seconds'].mean().nsmallest(6)
        top_drivers = driver_avg.index.tolist()
    
    # Create driver name mapping
    driver_names = {}
    driver_teams = {}
    if drivers_df is not None:
        for _, row in drivers_df.iterrows():
            driver_num = str(row.get('DriverNumber', ''))
            abbrev = row.get('Abbreviation', driver_num)
            driver_names[driver_num] = abbrev
            driver_names[abbrev] = abbrev
            driver_teams[abbrev] = row.get('TeamName', 'Unknown')
            driver_teams[driver_num] = row.get('TeamName', 'Unknown')
    
    # Create 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes = axes.flatten()
    
    # Calculate global y-axis limits
    all_times = []
    for driver in top_drivers:
        driver_laps = valid_laps[valid_laps['Driver'].astype(str) == str(driver)]
        if not driver_laps.empty:
            all_times.extend(driver_laps['LapTime_seconds'].tolist())
    
    if all_times:
        y_min = min(all_times) - 1
        y_max = max(all_times) + 1
    else:
        y_min, y_max = 85, 100
    
    for idx, driver in enumerate(top_drivers[:6]):
        ax = axes[idx]
        driver_laps = valid_laps[valid_laps['Driver'].astype(str) == str(driver)].sort_values('LapNumber')
        
        if driver_laps.empty:
            ax.set_visible(False)
            continue
        
        driver_name = driver_names.get(str(driver), str(driver))
        team_name = driver_teams.get(str(driver), 'Unknown')
        team_color = get_team_color(team_name)
        
        # Plot by stint/compound if available
        if 'Compound' in driver_laps.columns and 'Stint' in driver_laps.columns:
            for stint in sorted(driver_laps['Stint'].dropna().unique()):
                stint_laps = driver_laps[driver_laps['Stint'] == stint].sort_values('LapNumber')
                if stint_laps.empty:
                    continue
                
                compound = str(stint_laps['Compound'].iloc[0]).upper()
                compound_color = COMPOUND_PLOT_COLORS.get(compound, COMPOUND_PLOT_COLORS['UNKNOWN'])
                
                # Plot line with compound color
                ax.plot(stint_laps['LapNumber'], stint_laps['LapTime_seconds'],
                       color=compound_color, linewidth=2.5, alpha=0.9,
                       label=f"Stint {int(stint)} ({compound})")
                
                # Add markers at start and end of stint
                ax.scatter(stint_laps['LapNumber'].iloc[0], stint_laps['LapTime_seconds'].iloc[0],
                          color=compound_color, s=60, zorder=5, edgecolor='black', linewidth=0.5)
                ax.scatter(stint_laps['LapNumber'].iloc[-1], stint_laps['LapTime_seconds'].iloc[-1],
                          color=compound_color, s=60, zorder=5, edgecolor='black', linewidth=0.5)
        else:
            # Simple line plot with team color
            ax.plot(driver_laps['LapNumber'], driver_laps['LapTime_seconds'],
                   color=team_color, linewidth=2, alpha=0.8, label=driver_name)
        
        # Styling
        ax.set_title(f"P{idx+1}: {driver_name} ({team_name})", fontsize=12, fontweight='bold', 
                     color=team_color)
        ax.set_xlabel('Lap Number', fontsize=10)
        if idx % 3 == 0:
            ax.set_ylabel('Lap Time (seconds)', fontsize=10)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=8)
        
        # Add horizontal line for driver's median pace
        median_pace = driver_laps['LapTime_seconds'].median()
        ax.axhline(y=median_pace, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Hide unused subplots
    for idx in range(len(top_drivers), 6):
        axes[idx].set_visible(False)
    
    # Add compound legend at bottom
    compound_legend = [mpatches.Patch(color=color, label=compound) 
                       for compound, color in COMPOUND_PLOT_COLORS.items() 
                       if compound not in ['UNKNOWN']]
    fig.legend(handles=compound_legend, loc='lower center', ncol=5, fontsize=10,
               title='Tire Compounds', title_fontsize=11, 
               bbox_to_anchor=(0.5, -0.02))
    
    fig.suptitle('Lap Time Evolution by Driver', fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '09_lap_time_evolution.png'), dpi=150, 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return True


def plot_10_race_control_timeline(data_dir, plots_dir):
    """Create a clean swim-lane timeline of race control messages."""
    rc_df = load_csv(data_dir, 'race_control.csv')
    laps_df = load_csv(data_dir, 'laps.csv')
    event_info = load_csv(data_dir, 'event_info.csv')
    
    if rc_df is None or rc_df.empty:
        print("  ⚠️  No race control data")
        return False
    
    # Get total laps
    total_laps = int(laps_df['LapNumber'].max()) if laps_df is not None else 58
    
    # Categorize events into lanes
    lanes = {
        'Safety Car / VSC': {'keywords': ['SAFETY CAR', 'VSC', 'VIRTUAL'], 'color': '#FF8C00', 'events': []},
        'Yellow Flags': {'keywords': ['YELLOW', 'DOUBLE YELLOW'], 'color': '#FFD700', 'events': []},
        'Incidents': {'keywords': ['INCIDENT', 'COLLISION', 'NOTED'], 'color': '#E74C3C', 'events': []},
        'Investigations': {'keywords': ['INVESTIGATION', 'UNDER INVESTIGATION', 'REVIEWED'], 'color': '#9B59B6', 'events': []},
        'Penalties': {'keywords': ['PENALTY', 'TIME PENALTY', 'WARNING', 'BLACK AND WHITE', 'REPRIMAND'], 'color': '#C0392B', 'events': []},
        'DRS': {'keywords': ['DRS'], 'color': '#27AE60', 'events': []},
        'Track Limits': {'keywords': ['TRACK LIMITS', 'DELETED'], 'color': '#3498DB', 'events': []},
    }
    
    # Assign events to lanes
    for _, row in rc_df.iterrows():
        message = str(row.get('Message', '')).upper()
        lap = row.get('Lap', 1)
        if pd.isna(lap):
            lap = 1
        
        assigned = False
        for lane_name, lane_data in lanes.items():
            if any(kw in message for kw in lane_data['keywords']):
                lane_data['events'].append({'lap': int(lap), 'message': row.get('Message', '')})
                assigned = True
                break
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Title
    race_name = event_info['EventName'].iloc[0] if event_info is not None else "Race"
    ax.set_title(f'Race Control Timeline - {race_name}', fontsize=18, fontweight='bold', pad=20)
    
    # Draw lanes
    lane_names = list(lanes.keys())
    lane_height = 0.8
    
    for i, (lane_name, lane_data) in enumerate(lanes.items()):
        y_center = len(lanes) - i - 0.5
        
        # Draw lane background
        ax.axhspan(y_center - lane_height/2, y_center + lane_height/2, 
                   alpha=0.1, color=lane_data['color'])
        
        # Draw lane label
        ax.text(-2, y_center, lane_name, ha='right', va='center', fontsize=11, 
                fontweight='bold', color=lane_data['color'])
        
        # Draw events as markers
        event_laps = [e['lap'] for e in lane_data['events']]
        if event_laps:
            # Count events per lap for sizing
            lap_counts = {}
            for lap in event_laps:
                lap_counts[lap] = lap_counts.get(lap, 0) + 1
            
            unique_laps = list(lap_counts.keys())
            sizes = [50 + lap_counts[lap] * 30 for lap in unique_laps]
            
            ax.scatter(unique_laps, [y_center] * len(unique_laps), 
                      s=sizes, c=lane_data['color'], alpha=0.7, 
                      edgecolors='white', linewidths=1, zorder=3)
            
            # Add count labels for multiple events
            for lap, count in lap_counts.items():
                if count > 1:
                    ax.text(lap, y_center, str(count), ha='center', va='center',
                           fontsize=8, fontweight='bold', color='white', zorder=4)
    
    # Draw lap grid
    for lap in range(0, total_laps + 1, 5):
        ax.axvline(x=lap, color='gray', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.text(lap, -0.3, str(lap), ha='center', va='top', fontsize=9, color='gray')
    
    # Highlight key race phases
    # Start
    ax.axvline(x=1, color='green', alpha=0.5, linewidth=2, linestyle='-')
    ax.text(1, len(lanes) + 0.3, 'START', ha='center', va='bottom', fontsize=9, 
            color='green', fontweight='bold')
    
    # Finish
    ax.axvline(x=total_laps, color='#E8002D', alpha=0.5, linewidth=2, linestyle='-')
    ax.text(total_laps, len(lanes) + 0.3, 'FINISH', ha='center', va='bottom', fontsize=9,
            color='#E8002D', fontweight='bold')
    
    # Styling
    ax.set_xlim(-1, total_laps + 3)
    ax.set_ylim(-0.8, len(lanes) + 0.8)
    ax.set_xlabel('Lap Number', fontsize=12, labelpad=10)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add summary stats at bottom
    summary_text = []
    for lane_name, lane_data in lanes.items():
        count = len(lane_data['events'])
        if count > 0:
            summary_text.append(f"{lane_name}: {count}")
    
    ax.text(0.5, -0.12, ' | '.join(summary_text), transform=ax.transAxes,
            ha='center', va='top', fontsize=10, color='gray', style='italic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '10_race_control_timeline.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    return True


def plot_11_race_events_list(data_dir, plots_dir):
    """Create a clean list of key race events with lap numbers."""
    rc_df = load_csv(data_dir, 'race_control.csv')
    event_info = load_csv(data_dir, 'event_info.csv')
    
    if rc_df is None or rc_df.empty:
        print("  ⚠️  No race control data")
        return False
    
    # Filter for important events (exclude routine DRS enable/disable, track limits)
    important_keywords = [
        'INCIDENT', 'INVESTIGATION', 'PENALTY', 'SAFETY CAR', 'VSC', 
        'RED FLAG', 'YELLOW', 'COLLISION', 'RETIRED', 'DNF', 
        'BLACK AND WHITE', 'TIME PENALTY', 'DRIVE THROUGH', 'STOP AND GO',
        'WARNING', 'REPRIMAND', 'DISQUALIFIED'
    ]
    
    # Also include flag changes
    flag_events = rc_df[rc_df['Category'] == 'Flag'].copy()
    
    # Filter other events by keywords
    other_events = rc_df[rc_df['Category'] != 'Flag'].copy()
    mask = other_events['Message'].str.upper().apply(
        lambda x: any(kw in str(x).upper() for kw in important_keywords)
    )
    other_events = other_events[mask]
    
    # Combine and sort
    key_events = pd.concat([flag_events, other_events]).drop_duplicates()
    if 'Lap' in key_events.columns:
        key_events = key_events.sort_values('Lap')
    
    # Limit to most important 25 events
    key_events = key_events.head(25)
    
    if key_events.empty:
        # If no key events, just show first 20 events
        key_events = rc_df.head(20)
    
    # Create figure with table-like layout
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')
    
    # Title
    race_name = event_info['EventName'].iloc[0] if event_info is not None else "Race"
    fig.suptitle(f'Key Race Events - {race_name}', fontsize=18, fontweight='bold', y=0.98)
    
    # Event type colors
    event_colors = {
        'Flag': '#FFD700',
        'SafetyCar': '#FF8C00',
        'Drs': '#32CD32',
        'Other': '#4A90D9',
        'Penalty': '#E8002D',
    }
    
    # Draw events as a list
    y_start = 0.92
    y_step = 0.035
    
    for i, (_, row) in enumerate(key_events.iterrows()):
        y_pos = y_start - (i * y_step)
        if y_pos < 0.05:
            break
        
        lap = row.get('Lap', '?')
        if pd.isna(lap):
            lap = '?'
        else:
            lap = int(lap)
        
        category = row.get('Category', 'Other')
        message = str(row.get('Message', ''))[:80]  # Truncate long messages
        
        # Determine color based on content
        if 'PENALTY' in message.upper() or 'INVESTIGATION' in message.upper():
            color = event_colors['Penalty']
        else:
            color = event_colors.get(category, event_colors['Other'])
        
        # Draw lap number box
        lap_text = f"Lap {lap:>2}" if isinstance(lap, int) else f"Lap {lap}"
        ax.text(0.02, y_pos, lap_text, transform=ax.transAxes, fontsize=11, 
                fontweight='bold', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3, edgecolor=color))
        
        # Draw category badge
        cat_short = category[:4].upper()
        ax.text(0.12, y_pos, cat_short, transform=ax.transAxes, fontsize=9,
                fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))
        
        # Draw message
        ax.text(0.19, y_pos, message, transform=ax.transAxes, fontsize=10,
                fontfamily='sans-serif', va='center')
    
    # Add legend at bottom
    legend_y = 0.02
    legend_x = 0.1
    for cat, color in event_colors.items():
        ax.add_patch(mpatches.FancyBboxPatch((legend_x, legend_y), 0.08, 0.025, 
                                              boxstyle='round,pad=0.01',
                                              facecolor=color, alpha=0.7,
                                              transform=ax.transAxes))
        ax.text(legend_x + 0.04, legend_y + 0.012, cat, transform=ax.transAxes,
                fontsize=9, ha='center', va='center', fontweight='bold')
        legend_x += 0.12
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '11_race_events_list.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    return True


def plot_12_position_changes(data_dir, plots_dir):
    """Create a bump chart showing position changes throughout the race."""
    laps_df = load_csv(data_dir, 'laps.csv')
    results_df = load_csv(data_dir, 'session_results.csv')
    drivers_df = load_csv(data_dir, 'drivers.csv')
    
    if laps_df is None or laps_df.empty:
        print("  ⚠️  No lap data")
        return False
    
    if 'Position' not in laps_df.columns:
        print("  ⚠️  No position data in laps")
        return False
    
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Get driver info
    driver_teams = {}
    if drivers_df is not None:
        for _, row in drivers_df.iterrows():
            driver_teams[str(row.get('DriverNumber', ''))] = row.get('TeamName', 'Unknown')
            driver_teams[row.get('Abbreviation', '')] = row.get('TeamName', 'Unknown')
    
    # Get finishing order for legend
    if results_df is not None:
        pos_col = 'Position' if 'Position' in results_df.columns else 'ClassifiedPosition'
        results_df[pos_col] = pd.to_numeric(results_df[pos_col], errors='coerce')
        ordered = results_df.dropna(subset=[pos_col]).sort_values(pos_col)
        driver_order = ordered['Abbreviation'].tolist() if 'Abbreviation' in ordered.columns else []
    else:
        driver_order = laps_df['Driver'].unique().tolist()
    
    # Plot each driver's position over laps
    for driver in driver_order[:15]:  # Top 15
        driver_laps = laps_df[laps_df['Driver'] == driver].sort_values('LapNumber')
        if driver_laps.empty or 'Position' not in driver_laps.columns:
            continue
        
        positions = driver_laps['Position'].values
        lap_numbers = driver_laps['LapNumber'].values
        
        team = driver_teams.get(driver, 'Unknown')
        color = get_team_color(team)
        
        ax.plot(lap_numbers, positions, linewidth=2.5, label=driver, 
                color=color, alpha=0.85)
        
        # Add driver label at end
        if len(positions) > 0:
            ax.text(lap_numbers[-1] + 0.5, positions[-1], driver, 
                    fontsize=9, va='center', color=color, fontweight='bold')
    
    ax.set_xlabel('Lap Number', fontsize=12)
    ax.set_ylabel('Position', fontsize=12)
    ax.set_title('Race Position Changes', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylim(20.5, 0.5)  # Invert so P1 is at top
    ax.set_xlim(0, laps_df['LapNumber'].max() + 3)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yticks(range(1, 21))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '12_position_changes.png'), dpi=150, 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return True


def plot_13_grid_vs_finish(data_dir, plots_dir):
    """Create scatter plot comparing grid position to finish position."""
    results_df = load_csv(data_dir, 'session_results.csv')
    
    if results_df is None or results_df.empty:
        print("  ⚠️  No results data")
        return False
    
    if 'GridPosition' not in results_df.columns:
        print("  ⚠️  No grid position data")
        return False
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    pos_col = 'Position' if 'Position' in results_df.columns else 'ClassifiedPosition'
    results_df['Grid'] = pd.to_numeric(results_df['GridPosition'], errors='coerce')
    results_df['Finish'] = pd.to_numeric(results_df[pos_col], errors='coerce')
    
    valid = results_df.dropna(subset=['Grid', 'Finish'])
    
    # Draw diagonal line (no change)
    ax.plot([0, 21], [0, 21], 'k--', alpha=0.3, linewidth=2, label='No change')
    
    # Color regions
    ax.fill_between([0, 21], [0, 21], [21, 21], alpha=0.1, color='red', label='Lost positions')
    ax.fill_between([0, 21], [0, 0], [0, 21], alpha=0.1, color='green', label='Gained positions')
    
    # Plot each driver
    for _, row in valid.iterrows():
        team = row.get('TeamName', 'Unknown')
        color = get_team_color(team)
        driver = row.get('Abbreviation', str(row.get('DriverNumber', '')))
        
        gained = row['Grid'] - row['Finish']
        marker = '^' if gained > 0 else ('v' if gained < 0 else 'o')
        size = 150 + abs(gained) * 30
        
        ax.scatter(row['Grid'], row['Finish'], s=size, c=color, 
                   marker=marker, edgecolors='white', linewidths=1.5, zorder=3)
        ax.annotate(driver, (row['Grid'], row['Finish']), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Grid Position', fontsize=12)
    ax.set_ylabel('Finish Position', fontsize=12)
    ax.set_title('Grid Position vs Finish Position', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, 21)
    ax.set_ylim(21, 0)  # Invert Y
    ax.set_xticks(range(1, 21))
    ax.set_yticks(range(1, 21))
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    biggest_gain = valid.loc[(valid['Grid'] - valid['Finish']).idxmax()]
    biggest_loss = valid.loc[(valid['Grid'] - valid['Finish']).idxmin()]
    
    ax.text(0.02, 0.98, f"Biggest Gain: {biggest_gain['Abbreviation']} (+{int(biggest_gain['Grid'] - biggest_gain['Finish'])})", 
            transform=ax.transAxes, fontsize=10, va='top', color='green', fontweight='bold')
    ax.text(0.02, 0.93, f"Biggest Loss: {biggest_loss['Abbreviation']} ({int(biggest_loss['Grid'] - biggest_loss['Finish'])})", 
            transform=ax.transAxes, fontsize=10, va='top', color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '13_grid_vs_finish.png'), dpi=150, 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return True


def plot_14_compound_performance(data_dir, plots_dir):
    """Compare average lap times on different tire compounds."""
    laps_df = load_csv(data_dir, 'laps.csv')
    
    if laps_df is None or laps_df.empty:
        print("  ⚠️  No lap data")
        return False
    
    if 'Compound' not in laps_df.columns or 'LapTime_seconds' not in laps_df.columns:
        print("  ⚠️  No compound or lap time data")
        return False
    
    # Filter valid laps
    valid = laps_df[(laps_df['LapTime_seconds'] > 0) & (laps_df['LapTime_seconds'] < 200)].copy()
    median_time = valid['LapTime_seconds'].median()
    valid = valid[valid['LapTime_seconds'] < median_time * 1.1]  # Remove pit laps
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Box plot by compound
    compounds = ['SOFT', 'MEDIUM', 'HARD']
    compound_data = []
    compound_labels = []
    compound_colors = []
    
    for compound in compounds:
        data = valid[valid['Compound'].str.upper() == compound]['LapTime_seconds']
        if not data.empty:
            compound_data.append(data.values)
            compound_labels.append(compound)
            compound_colors.append(COMPOUND_PLOT_COLORS.get(compound, '#888888'))
    
    if compound_data:
        bp = axes[0].boxplot(compound_data, labels=compound_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], compound_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[0].set_ylabel('Lap Time (seconds)', fontsize=11)
        axes[0].set_xlabel('Tire Compound', fontsize=11)
        axes[0].set_title('Lap Time Distribution by Compound', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
    
    # Right: Average pace by compound per driver (top 5)
    driver_avg = valid.groupby('Driver')['LapTime_seconds'].mean().nsmallest(5)
    top_drivers = driver_avg.index.tolist()
    
    x = np.arange(len(top_drivers))
    width = 0.25
    
    for i, compound in enumerate(compounds):
        compound_times = []
        for driver in top_drivers:
            driver_compound = valid[(valid['Driver'] == driver) & 
                                    (valid['Compound'].str.upper() == compound)]
            if not driver_compound.empty:
                compound_times.append(driver_compound['LapTime_seconds'].mean())
            else:
                compound_times.append(np.nan)
        
        color = COMPOUND_PLOT_COLORS.get(compound, '#888888')
        bars = axes[1].bar(x + i * width, compound_times, width, 
                          label=compound, color=color, alpha=0.8)
    
    axes[1].set_ylabel('Average Lap Time (seconds)', fontsize=11)
    axes[1].set_xlabel('Driver', fontsize=11)
    axes[1].set_title('Average Pace by Compound (Top 5 Drivers)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(top_drivers)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Tire Compound Performance Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '14_compound_performance.png'), dpi=150, 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return True


def plot_15_driver_consistency(data_dir, plots_dir):
    """Compare driver consistency (standard deviation of lap times)."""
    laps_df = load_csv(data_dir, 'laps.csv')
    results_df = load_csv(data_dir, 'session_results.csv')
    drivers_df = load_csv(data_dir, 'drivers.csv')
    
    if laps_df is None or laps_df.empty:
        print("  ⚠️  No lap data")
        return False
    
    if 'LapTime_seconds' not in laps_df.columns:
        print("  ⚠️  No lap time data")
        return False
    
    # Filter valid laps
    valid = laps_df[(laps_df['LapTime_seconds'] > 0) & (laps_df['LapTime_seconds'] < 200)].copy()
    median_time = valid['LapTime_seconds'].median()
    valid = valid[valid['LapTime_seconds'] < median_time * 1.08]  # Remove outliers
    
    # Calculate stats per driver
    driver_stats = []
    for driver in valid['Driver'].unique():
        driver_laps = valid[valid['Driver'] == driver]['LapTime_seconds']
        if len(driver_laps) >= 10:  # Need enough laps
            driver_stats.append({
                'Driver': driver,
                'Mean': driver_laps.mean(),
                'Std': driver_laps.std(),
                'Min': driver_laps.min(),
                'Max': driver_laps.max(),
                'Count': len(driver_laps)
            })
    
    stats_df = pd.DataFrame(driver_stats)
    
    # Get team colors
    driver_teams = {}
    if drivers_df is not None:
        for _, row in drivers_df.iterrows():
            driver_teams[str(row.get('DriverNumber', ''))] = row.get('TeamName', 'Unknown')
            driver_teams[row.get('Abbreviation', '')] = row.get('TeamName', 'Unknown')
    
    # Sort by mean pace
    stats_df = stats_df.sort_values('Mean')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Consistency (std dev) - lower is better
    colors = [get_team_color(driver_teams.get(d, 'Unknown')) for d in stats_df['Driver']]
    bars = axes[0].barh(stats_df['Driver'], stats_df['Std'], color=colors, alpha=0.8)
    axes[0].set_xlabel('Lap Time Standard Deviation (seconds)', fontsize=11)
    axes[0].set_ylabel('Driver', fontsize=11)
    axes[0].set_title('Driver Consistency (Lower = More Consistent)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Add values
    for bar, val in zip(bars, stats_df['Std']):
        axes[0].text(val + 0.05, bar.get_y() + bar.get_height()/2, 
                    f'{val:.2f}s', va='center', fontsize=9)
    
    # Right: Pace vs Consistency scatter
    for _, row in stats_df.iterrows():
        team = driver_teams.get(row['Driver'], 'Unknown')
        color = get_team_color(team)
        axes[1].scatter(row['Mean'], row['Std'], s=150, c=color, 
                       edgecolors='white', linewidths=1.5, alpha=0.8)
        axes[1].annotate(row['Driver'], (row['Mean'], row['Std']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    axes[1].set_xlabel('Average Lap Time (seconds) - Faster →', fontsize=11)
    axes[1].set_ylabel('Standard Deviation (seconds) - More Consistent ↓', fontsize=11)
    axes[1].set_title('Pace vs Consistency', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Highlight ideal quadrant (fast + consistent)
    xlim = axes[1].get_xlim()
    ylim = axes[1].get_ylim()
    mid_x = (xlim[0] + xlim[1]) / 2
    mid_y = (ylim[0] + ylim[1]) / 2
    axes[1].axhline(y=mid_y, color='gray', linestyle='--', alpha=0.3)
    axes[1].axvline(x=mid_x, color='gray', linestyle='--', alpha=0.3)
    axes[1].text(xlim[0], ylim[0], 'FAST & CONSISTENT', fontsize=10, 
                color='green', alpha=0.7, fontweight='bold')
    
    plt.suptitle('Driver Consistency Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '15_driver_consistency.png'), dpi=150, 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return True


def plot_16_teammate_comparison(data_dir, plots_dir):
    """Compare teammates head-to-head."""
    laps_df = load_csv(data_dir, 'laps.csv')
    results_df = load_csv(data_dir, 'session_results.csv')
    drivers_df = load_csv(data_dir, 'drivers.csv')
    
    if results_df is None or results_df.empty or drivers_df is None:
        print("  ⚠️  No results or driver data")
        return False
    
    # Group drivers by team
    teams = {}
    for _, row in results_df.iterrows():
        team = row.get('TeamName', 'Unknown')
        if team not in teams:
            teams[team] = []
        teams[team].append({
            'Abbreviation': row.get('Abbreviation', ''),
            'Position': row.get('Position', row.get('ClassifiedPosition', 99)),
            'Points': row.get('Points', 0),
        })
    
    # Filter teams with 2 drivers
    valid_teams = {k: v for k, v in teams.items() if len(v) == 2}
    
    if not valid_teams:
        print("  ⚠️  No complete team pairs")
        return False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Position comparison
    ax = axes[0, 0]
    team_names = list(valid_teams.keys())
    x = np.arange(len(team_names))
    
    driver1_pos = []
    driver2_pos = []
    driver1_names = []
    driver2_names = []
    team_colors = []
    
    for team in team_names:
        drivers = sorted(valid_teams[team], key=lambda x: x['Position'])
        driver1_pos.append(drivers[0]['Position'])
        driver2_pos.append(drivers[1]['Position'])
        driver1_names.append(drivers[0]['Abbreviation'])
        driver2_names.append(drivers[1]['Abbreviation'])
        team_colors.append(get_team_color(team))
    
    for i, (pos1, pos2, name1, name2, color) in enumerate(zip(driver1_pos, driver2_pos, 
                                                               driver1_names, driver2_names, team_colors)):
        ax.plot([i, i], [pos1, pos2], color=color, linewidth=3, alpha=0.6)
        ax.scatter([i], [pos1], s=150, color=color, zorder=3, edgecolors='white', linewidths=2)
        ax.scatter([i], [pos2], s=150, color=color, zorder=3, edgecolors='white', linewidths=2, alpha=0.5)
        ax.text(i + 0.1, pos1, name1, fontsize=9, va='center', fontweight='bold')
        ax.text(i + 0.1, pos2, name2, fontsize=9, va='center', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels([t[:12] for t in team_names], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Finish Position')
    ax.set_ylim(21, 0)
    ax.set_title('Teammate Position Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Points comparison (if available)
    ax = axes[0, 1]
    driver1_pts = []
    driver2_pts = []
    
    for team in team_names:
        drivers = sorted(valid_teams[team], key=lambda x: -x.get('Points', 0))
        driver1_pts.append(drivers[0].get('Points', 0))
        driver2_pts.append(drivers[1].get('Points', 0))
    
    width = 0.35
    ax.bar(x - width/2, driver1_pts, width, color=team_colors, alpha=0.9, label='Leader')
    ax.bar(x + width/2, driver2_pts, width, color=team_colors, alpha=0.5, label='Other')
    ax.set_xticks(x)
    ax.set_xticklabels([t[:12] for t in team_names], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Points Scored')
    ax.set_title('Teammate Points Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Average pace comparison (if lap data available)
    ax = axes[1, 0]
    if laps_df is not None and 'LapTime_seconds' in laps_df.columns:
        valid_laps = laps_df[(laps_df['LapTime_seconds'] > 0) & (laps_df['LapTime_seconds'] < 200)]
        median_time = valid_laps['LapTime_seconds'].median()
        valid_laps = valid_laps[valid_laps['LapTime_seconds'] < median_time * 1.08]
        
        driver1_pace = []
        driver2_pace = []
        
        for team in team_names:
            drivers = valid_teams[team]
            pace = []
            for d in drivers:
                driver_laps = valid_laps[valid_laps['Driver'] == d['Abbreviation']]
                if not driver_laps.empty:
                    pace.append(driver_laps['LapTime_seconds'].mean())
                else:
                    pace.append(np.nan)
            
            pace.sort()  # Faster first
            driver1_pace.append(pace[0] if len(pace) > 0 else np.nan)
            driver2_pace.append(pace[1] if len(pace) > 1 else np.nan)
        
        # Calculate gap
        gaps = [p2 - p1 if not (np.isnan(p1) or np.isnan(p2)) else 0 for p1, p2 in zip(driver1_pace, driver2_pace)]
        
        bars = ax.bar(x, gaps, color=team_colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([t[:12] for t in team_names], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Pace Gap (seconds)')
        ax.set_title('Average Pace Gap Between Teammates', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linewidth=0.5)
    
    # 4. Summary text
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_lines = ['TEAM BATTLE SUMMARY\n']
    for team in team_names:
        drivers = sorted(valid_teams[team], key=lambda x: x['Position'])
        winner = drivers[0]['Abbreviation']
        loser = drivers[1]['Abbreviation']
        gap = int(drivers[1]['Position']) - int(drivers[0]['Position'])
        summary_lines.append(f"{team[:15]}: {winner} beat {loser} by {gap} places")
    
    ax.text(0.1, 0.9, '\n'.join(summary_lines), transform=ax.transAxes,
            fontsize=11, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.suptitle('Teammate Head-to-Head Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '16_teammate_comparison.png'), dpi=150, 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return True


def plot_17_stint_degradation(data_dir, plots_dir):
    """Analyze tire degradation within stints."""
    laps_df = load_csv(data_dir, 'laps.csv')
    stints_df = load_csv(data_dir, 'stints.csv')
    drivers_df = load_csv(data_dir, 'drivers.csv')
    
    if laps_df is None or laps_df.empty or stints_df is None:
        print("  ⚠️  No lap or stint data")
        return False
    
    if 'LapTime_seconds' not in laps_df.columns:
        print("  ⚠️  No lap time data")
        return False
    
    # Create driver number to abbreviation mapping
    driver_num_to_abbrev = {}
    if drivers_df is not None:
        for _, row in drivers_df.iterrows():
            driver_num_to_abbrev[str(row.get('DriverNumber', ''))] = row.get('Abbreviation', '')
    
    # Filter valid laps
    valid = laps_df[(laps_df['LapTime_seconds'] > 0) & (laps_df['LapTime_seconds'] < 200)].copy()
    median_time = valid['LapTime_seconds'].median()
    valid = valid[valid['LapTime_seconds'] < median_time * 1.1]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Degradation per lap for each compound
    ax = axes[0]
    
    compounds = ['MEDIUM', 'HARD']
    for compound in compounds:
        compound_stints = stints_df[stints_df['Compound'].str.upper() == compound]
        
        all_normalized_times = {}
        for _, stint in compound_stints.iterrows():
            driver_num = str(stint['Driver'])
            # Convert driver number to abbreviation for matching
            driver_abbrev = driver_num_to_abbrev.get(driver_num, driver_num)
            
            lap_start = stint.get('LapStart', 1)
            lap_end = stint.get('LapEnd', lap_start + 10)
            
            # Match by abbreviation (Driver column in laps) or DriverNumber
            stint_laps = valid[
                ((valid['Driver'] == driver_abbrev) | (valid['DriverNumber'].astype(str) == driver_num)) & 
                (valid['LapNumber'] >= lap_start) & 
                (valid['LapNumber'] <= lap_end)
            ].sort_values('LapNumber')
            
            if len(stint_laps) < 5:
                continue
            
            # Normalize lap times to stint start (skip first lap which is often slow)
            base_time = stint_laps['LapTime_seconds'].iloc[1] if len(stint_laps) > 1 else stint_laps['LapTime_seconds'].iloc[0]
            for i, (_, lap) in enumerate(stint_laps.iterrows()):
                if i == 0:  # Skip first lap of stint
                    continue
                tyre_age = i
                if tyre_age not in all_normalized_times:
                    all_normalized_times[tyre_age] = []
                all_normalized_times[tyre_age].append(lap['LapTime_seconds'] - base_time)
        
        if all_normalized_times:
            ages = sorted(all_normalized_times.keys())[:25]  # First 25 laps
            avg_degradation = [np.mean(all_normalized_times[age]) for age in ages]
            
            color = COMPOUND_PLOT_COLORS.get(compound, '#888888')
            ax.plot(ages, avg_degradation, marker='o', markersize=6, 
                   linewidth=2.5, label=compound, color=color, alpha=0.9)
            
            # Add fill
            ax.fill_between(ages, 0, avg_degradation, color=color, alpha=0.2)
    
    ax.set_xlabel('Tire Age (laps into stint)', fontsize=11)
    ax.set_ylabel('Lap Time Increase from Stint Start (seconds)', fontsize=11)
    ax.set_title('Average Tire Degradation by Compound', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    # Right: Stint length vs average pace
    ax = axes[1]
    
    stint_data = []
    for _, stint in stints_df.iterrows():
        driver_num = str(stint['Driver'])
        driver_abbrev = driver_num_to_abbrev.get(driver_num, driver_num)
        
        lap_start = stint.get('LapStart', 1)
        lap_end = stint.get('LapEnd', lap_start + 10)
        compound = stint.get('Compound', 'UNKNOWN')
        
        stint_laps = valid[
            ((valid['Driver'] == driver_abbrev) | (valid['DriverNumber'].astype(str) == driver_num)) & 
            (valid['LapNumber'] >= lap_start) & 
            (valid['LapNumber'] <= lap_end)
        ]
        
        if len(stint_laps) >= 3:
            stint_data.append({
                'Driver': driver_abbrev,
                'Length': len(stint_laps),
                'AvgPace': stint_laps['LapTime_seconds'].mean(),
                'Compound': compound.upper() if compound else 'UNKNOWN'
            })
    
    if stint_data:
        stint_plot_df = pd.DataFrame(stint_data)
        
        for compound in ['MEDIUM', 'HARD']:
            compound_data = stint_plot_df[stint_plot_df['Compound'] == compound]
            if not compound_data.empty:
                color = COMPOUND_PLOT_COLORS.get(compound, '#888888')
                ax.scatter(compound_data['Length'], compound_data['AvgPace'],
                          s=150, c=color, alpha=0.7, label=compound, 
                          edgecolors='white', linewidths=1.5)
                
                # Add driver labels
                for _, row in compound_data.iterrows():
                    ax.annotate(row['Driver'], (row['Length'], row['AvgPace']),
                               xytext=(5, 0), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Stint Length (laps)', fontsize=11)
    ax.set_ylabel('Average Lap Time (seconds)', fontsize=11)
    ax.set_title('Stint Length vs Average Pace', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Tire Stint Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '17_stint_degradation.png'), dpi=150, 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return True


def plot_18_sector_dominance(data_dir, plots_dir):
    """Show which drivers dominated which sectors - with delta to fastest."""
    laps_df = load_csv(data_dir, 'laps.csv')
    drivers_df = load_csv(data_dir, 'drivers.csv')
    
    if laps_df is None or laps_df.empty:
        print("  ⚠️  No lap data")
        return False
    
    sector_cols = ['Sector1Time_seconds', 'Sector2Time_seconds', 'Sector3Time_seconds']
    if not all(col in laps_df.columns for col in sector_cols):
        print("  ⚠️  No sector time data")
        return False
    
    # Get team colors
    driver_teams = {}
    if drivers_df is not None:
        for _, row in drivers_df.iterrows():
            driver_teams[str(row.get('DriverNumber', ''))] = row.get('TeamName', 'Unknown')
            driver_teams[row.get('Abbreviation', '')] = row.get('TeamName', 'Unknown')
    
    # Filter valid data
    valid = laps_df.copy()
    for col in sector_cols:
        valid = valid[valid[col] > 0]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    
    sector_names = ['Sector 1', 'Sector 2', 'Sector 3']
    
    for i, (col, name) in enumerate(zip(sector_cols, sector_names)):
        ax = axes[i]
        
        # Get best sector times per driver
        best_sectors = valid.groupby('Driver')[col].min().nsmallest(10)
        
        # Calculate delta to fastest
        fastest_time = best_sectors.min()
        deltas = best_sectors - fastest_time
        
        colors = [get_team_color(driver_teams.get(d, 'Unknown')) for d in best_sectors.index]
        
        # Create horizontal bars showing delta
        y_pos = range(len(best_sectors))
        bars = ax.barh(y_pos, deltas.values, color=colors, alpha=0.85, 
                      edgecolor='white', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(best_sectors.index, fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        
        # Add time labels on bars
        for j, (driver, delta, actual) in enumerate(zip(best_sectors.index, deltas.values, best_sectors.values)):
            # Show actual time at the end of bar
            if delta < 0.01:  # Fastest
                ax.text(0.01, j, f'{actual:.3f}s', va='center', ha='left', 
                       fontsize=10, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='green', alpha=0.8))
            else:
                ax.text(delta + 0.005, j, f'+{delta:.3f}s', va='center', ha='left', 
                       fontsize=9, color='#333333')
                # Show actual time inside bar if space
                if delta > 0.1:
                    ax.text(delta / 2, j, f'{actual:.3f}s', va='center', ha='center', 
                           fontsize=8, color='white', alpha=0.9)
        
        ax.set_xlabel('Delta to Fastest (seconds)', fontsize=11)
        ax.set_title(f'{name}\n🏆 {best_sectors.idxmin()} ({fastest_time:.3f}s)', 
                    fontsize=13, fontweight='bold', color=get_team_color(driver_teams.get(best_sectors.idxmin(), 'Unknown')))
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax.set_xlim(0, deltas.max() * 1.3)  # Add space for labels
        
        # Add vertical line at 0
        ax.axvline(x=0, color='green', linewidth=2, alpha=0.7)
    
    plt.suptitle('Sector Dominance - Gap to Fastest', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '18_sector_dominance.png'), dpi=150, 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return True


def plot_20_race_start_analysis(data_dir, plots_dir):
    """Analyze race starts - positions gained/lost and first lap performance."""
    laps_df = load_csv(data_dir, 'laps.csv')
    results_df = load_csv(data_dir, 'session_results.csv')
    drivers_df = load_csv(data_dir, 'drivers.csv')
    
    if laps_df is None or laps_df.empty or results_df is None:
        print("  ⚠️  No lap or results data")
        return False
    
    # Get driver info
    driver_teams = {}
    if drivers_df is not None:
        for _, row in drivers_df.iterrows():
            driver_teams[row.get('Abbreviation', '')] = row.get('TeamName', 'Unknown')
            driver_teams[str(row.get('DriverNumber', ''))] = row.get('TeamName', 'Unknown')
    
    # Get lap 1 data
    lap1 = laps_df[laps_df['LapNumber'] == 1].copy()
    
    if lap1.empty:
        print("  ⚠️  No lap 1 data")
        return False
    
    # Get grid positions and positions after lap 1
    pos_col = 'Position' if 'Position' in results_df.columns else 'ClassifiedPosition'
    results_df['GridPosition'] = pd.to_numeric(results_df['GridPosition'], errors='coerce')
    
    # Merge lap 1 data with grid positions
    start_data = []
    for _, row in lap1.iterrows():
        driver = row.get('Driver', '')
        driver_num = str(row.get('DriverNumber', ''))
        
        # Find grid position
        driver_result = results_df[
            (results_df['Abbreviation'] == driver) | 
            (results_df['DriverNumber'].astype(str) == driver_num)
        ]
        
        if driver_result.empty:
            continue
        
        grid_pos = driver_result['GridPosition'].iloc[0]
        
        # Position after lap 1 (from lap data if available, otherwise estimate)
        pos_after_lap1 = row.get('Position', grid_pos)
        
        # Lap 1 time
        lap1_time = row.get('LapTime_seconds', None)
        
        # Sector 1 time (the actual start)
        sector1_time = row.get('Sector1Time_seconds', None)
        
        team = driver_teams.get(driver, driver_teams.get(driver_num, 'Unknown'))
        
        if pd.notna(grid_pos):
            start_data.append({
                'Driver': driver if driver else driver_num,
                'GridPosition': int(grid_pos),
                'PositionAfterLap1': pos_after_lap1,
                'Lap1Time': lap1_time,
                'Sector1Time': sector1_time,
                'Team': team,
                'PositionsGained': int(grid_pos) - int(pos_after_lap1) if pd.notna(pos_after_lap1) else 0
            })
    
    if not start_data:
        print("  ⚠️  Could not build start data")
        return False
    
    start_df = pd.DataFrame(start_data)
    start_df = start_df.sort_values('GridPosition')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Positions Gained/Lost at Start (Top Left)
    ax = axes[0, 0]
    
    # Sort by positions gained
    sorted_by_gain = start_df.sort_values('PositionsGained', ascending=False)
    
    colors = []
    for _, row in sorted_by_gain.iterrows():
        if row['PositionsGained'] > 0:
            colors.append('#27AE60')  # Green for gained
        elif row['PositionsGained'] < 0:
            colors.append('#E74C3C')  # Red for lost
        else:
            colors.append('#95A5A6')  # Gray for no change
    
    bars = ax.barh(range(len(sorted_by_gain)), sorted_by_gain['PositionsGained'].values, 
                   color=colors, alpha=0.8, edgecolor='white')
    
    ax.set_yticks(range(len(sorted_by_gain)))
    ax.set_yticklabels([f"{row['Driver']} (P{row['GridPosition']})" 
                        for _, row in sorted_by_gain.iterrows()], fontsize=9)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Positions Gained/Lost', fontsize=11)
    ax.set_title('Positions Gained/Lost at Start', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (_, row) in enumerate(sorted_by_gain.iterrows()):
        val = row['PositionsGained']
        label = f"+{val}" if val > 0 else str(val)
        x_pos = val + 0.1 if val >= 0 else val - 0.1
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, i, label, va='center', ha=ha, fontsize=9, fontweight='bold')
    
    # 2. Lap 1 Times Comparison (Top Right)
    ax = axes[0, 1]
    
    valid_lap1 = start_df[start_df['Lap1Time'].notna()].sort_values('Lap1Time')
    
    if not valid_lap1.empty:
        fastest_lap1 = valid_lap1['Lap1Time'].min()
        deltas = valid_lap1['Lap1Time'] - fastest_lap1
        
        colors = [get_team_color(row['Team']) for _, row in valid_lap1.iterrows()]
        
        bars = ax.barh(range(len(valid_lap1)), deltas.values, color=colors, alpha=0.85)
        ax.set_yticks(range(len(valid_lap1)))
        ax.set_yticklabels(valid_lap1['Driver'].values, fontsize=9, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlabel('Delta to Fastest Lap 1 (seconds)', fontsize=11)
        ax.set_title(f'Lap 1 Times (Fastest: {valid_lap1.iloc[0]["Driver"]} - {fastest_lap1:.3f}s)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add time labels
        for i, (_, row) in enumerate(valid_lap1.iterrows()):
            delta = row['Lap1Time'] - fastest_lap1
            ax.text(delta + 0.05, i, f"+{delta:.2f}s" if delta > 0 else "FASTEST", 
                   va='center', fontsize=8)
    
    # 3. Grid vs Position After Lap 1 (Bottom Left)
    ax = axes[1, 0]
    
    # Draw diagonal (no change line)
    max_pos = 20
    ax.plot([0, max_pos+1], [0, max_pos+1], 'k--', alpha=0.3, linewidth=2, label='No change')
    
    for _, row in start_df.iterrows():
        color = get_team_color(row['Team'])
        gained = row['PositionsGained']
        
        if gained > 0:
            marker = '^'
            size = 150 + gained * 20
        elif gained < 0:
            marker = 'v'
            size = 150 + abs(gained) * 20
        else:
            marker = 'o'
            size = 100
        
        ax.scatter(row['GridPosition'], row['PositionAfterLap1'], 
                  s=size, c=color, marker=marker, edgecolors='white', 
                  linewidths=1.5, alpha=0.8, zorder=3)
        ax.annotate(row['Driver'], (row['GridPosition'], row['PositionAfterLap1']),
                   xytext=(3, 3), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Grid Position', fontsize=11)
    ax.set_ylabel('Position After Lap 1', fontsize=11)
    ax.set_title('Grid Position vs Position After Lap 1', fontsize=13, fontweight='bold')
    ax.set_xlim(0, max_pos + 1)
    ax.set_ylim(max_pos + 1, 0)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 4. Start Performance Summary (Bottom Right)
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate summary stats
    best_start = start_df.loc[start_df['PositionsGained'].idxmax()]
    worst_start = start_df.loc[start_df['PositionsGained'].idxmin()]
    
    summary_text = f"""
    RACE START SUMMARY
    {'='*40}
    
    🚀 BEST START:
       {best_start['Driver']} gained {best_start['PositionsGained']} positions
       (P{best_start['GridPosition']} → P{int(best_start['PositionAfterLap1'])})
    
    😓 WORST START:
       {worst_start['Driver']} lost {abs(worst_start['PositionsGained'])} positions
       (P{worst_start['GridPosition']} → P{int(worst_start['PositionAfterLap1'])})
    
    📊 STATISTICS:
       • Drivers who gained: {len(start_df[start_df['PositionsGained'] > 0])}
       • Drivers who lost: {len(start_df[start_df['PositionsGained'] < 0])}
       • No change: {len(start_df[start_df['PositionsGained'] == 0])}
       • Avg positions changed: {start_df['PositionsGained'].abs().mean():.1f}
    """
    
    # Add fastest lap 1 info if available
    if not valid_lap1.empty:
        fastest = valid_lap1.iloc[0]
        summary_text += f"""
    ⚡ FASTEST LAP 1:
       {fastest['Driver']} - {fastest['Lap1Time']:.3f}s
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))
    
    fig.suptitle('Race Start Analysis', fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '20_race_start_analysis.png'), dpi=150, 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return True


def plot_19_stint_degradation_by_driver(data_dir, plots_dir):
    """Show tire degradation for each driver in separate subplots."""
    laps_df = load_csv(data_dir, 'laps.csv')
    stints_df = load_csv(data_dir, 'stints.csv')
    drivers_df = load_csv(data_dir, 'drivers.csv')
    results_df = load_csv(data_dir, 'session_results.csv')
    
    if laps_df is None or laps_df.empty or stints_df is None:
        print("  ⚠️  No lap or stint data")
        return False
    
    if 'LapTime_seconds' not in laps_df.columns:
        print("  ⚠️  No lap time data")
        return False
    
    # Create driver number to abbreviation mapping
    driver_num_to_abbrev = {}
    driver_teams = {}
    if drivers_df is not None:
        for _, row in drivers_df.iterrows():
            num = str(row.get('DriverNumber', ''))
            abbrev = row.get('Abbreviation', num)
            driver_num_to_abbrev[num] = abbrev
            driver_teams[abbrev] = row.get('TeamName', 'Unknown')
            driver_teams[num] = row.get('TeamName', 'Unknown')
    
    # Get top 6 drivers by finishing position
    top_drivers = []
    if results_df is not None and not results_df.empty:
        pos_col = 'Position' if 'Position' in results_df.columns else 'ClassifiedPosition'
        if pos_col in results_df.columns:
            results_df[pos_col] = pd.to_numeric(results_df[pos_col], errors='coerce')
            ordered = results_df.dropna(subset=[pos_col]).sort_values(pos_col)
            if 'Abbreviation' in ordered.columns:
                top_drivers = ordered['Abbreviation'].head(6).tolist()
            else:
                top_drivers = ordered['DriverNumber'].astype(str).head(6).tolist()
    
    if not top_drivers:
        top_drivers = list(driver_num_to_abbrev.values())[:6]
    
    # Filter valid laps
    valid = laps_df[(laps_df['LapTime_seconds'] > 0) & (laps_df['LapTime_seconds'] < 200)].copy()
    median_time = valid['LapTime_seconds'].median()
    valid = valid[valid['LapTime_seconds'] < median_time * 1.15]
    
    # Create 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, driver_abbrev in enumerate(top_drivers[:6]):
        ax = axes[idx]
        
        # Find driver number for stint matching
        driver_num = None
        for num, abbrev in driver_num_to_abbrev.items():
            if abbrev == driver_abbrev:
                driver_num = num
                break
        
        if driver_num is None:
            driver_num = driver_abbrev
        
        # Get stints for this driver
        driver_stints = stints_df[stints_df['Driver'].astype(str) == driver_num]
        
        team = driver_teams.get(driver_abbrev, 'Unknown')
        team_color = get_team_color(team)
        
        has_data = False
        
        for _, stint in driver_stints.iterrows():
            lap_start = int(stint.get('LapStart', 1))
            lap_end = int(stint.get('LapEnd', lap_start + 10))
            compound = str(stint.get('Compound', 'UNKNOWN')).upper()
            stint_num = int(stint.get('Stint', 1))
            
            # Get laps for this stint
            stint_laps = valid[
                ((valid['Driver'] == driver_abbrev) | (valid['DriverNumber'].astype(str) == driver_num)) & 
                (valid['LapNumber'] >= lap_start) & 
                (valid['LapNumber'] <= lap_end)
            ].sort_values('LapNumber')
            
            if len(stint_laps) < 3:
                continue
            
            has_data = True
            
            # Calculate tire age (laps into stint)
            tire_ages = range(1, len(stint_laps) + 1)
            lap_times = stint_laps['LapTime_seconds'].values
            
            # Normalize to first valid lap (skip outliers at stint start)
            base_time = np.median(lap_times[:3]) if len(lap_times) >= 3 else lap_times[0]
            normalized_times = lap_times - base_time
            
            compound_color = COMPOUND_PLOT_COLORS.get(compound, '#888888')
            
            # Plot the degradation curve
            ax.plot(tire_ages, normalized_times, marker='o', markersize=5,
                   linewidth=2, color=compound_color, alpha=0.9,
                   label=f'Stint {stint_num} ({compound})')
            
            # Add fill under curve
            ax.fill_between(tire_ages, 0, normalized_times, color=compound_color, alpha=0.15)
        
        # Styling
        ax.set_title(f"P{idx+1}: {driver_abbrev} ({team})", fontsize=12, fontweight='bold',
                    color=team_color)
        ax.set_xlabel('Tire Age (laps)', fontsize=10)
        ax.set_ylabel('Δ Lap Time (s)', fontsize=10)
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=9)
        
        if has_data:
            # Set reasonable y-limits
            ax.set_ylim(-1, 3)
    
    # Add compound legend at bottom
    compound_legend = [mpatches.Patch(color=color, label=compound) 
                       for compound, color in COMPOUND_PLOT_COLORS.items() 
                       if compound not in ['UNKNOWN']]
    fig.legend(handles=compound_legend, loc='lower center', ncol=5, fontsize=10,
               title='Tire Compounds', title_fontsize=11, 
               bbox_to_anchor=(0.5, -0.02))
    
    fig.suptitle('Tire Degradation by Driver', fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '19_stint_degradation_by_driver.png'), dpi=150, 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return True


def main():
    print("=" * 60)
    print("🏎️  FastF1 Race Data Visualizer")
    print("=" * 60)
    
    # Find latest data directory
    data_dir = get_latest_data_dir()
    
    if data_dir is None:
        print("\n❌ No data directory found. Please run fetch_latest_race.py first.")
        return
    
    print(f"\n📁 Using data from: {data_dir}")
    
    # Create plots directory
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Load event info
    event_info = load_csv(data_dir, 'event_info.csv')
    if event_info is not None:
        print(f"🏁 Race: {event_info['EventName'].iloc[0]}")
    
    print("\n📊 Generating visualizations...")
    print("-" * 50)
    
    # List of all plot functions
    plots = [
        ("Race Summary", plot_01_race_summary),
        ("Final Positions", plot_02_final_positions),
        ("Lap Times", plot_03_lap_times),
        ("Sector Times Heatmap", plot_04_sector_times_heatmap),
        ("Tire Strategy", plot_05_tire_strategy),
        ("Weather Evolution", plot_06_weather_evolution),
        ("Pace Distribution", plot_07_pace_distribution),
        ("Telemetry Comparison", plot_08_telemetry_comparison),
        ("Lap Time Evolution", plot_09_lap_time_evolution),
        ("Race Control Timeline", plot_10_race_control_timeline),
        ("Race Events List", plot_11_race_events_list),
        ("Position Changes", plot_12_position_changes),
        ("Grid vs Finish", plot_13_grid_vs_finish),
        ("Compound Performance", plot_14_compound_performance),
        ("Driver Consistency", plot_15_driver_consistency),
        ("Teammate Comparison", plot_16_teammate_comparison),
        ("Stint Degradation", plot_17_stint_degradation),
        ("Sector Dominance", plot_18_sector_dominance),
        ("Stint Degradation by Driver", plot_19_stint_degradation_by_driver),
        ("Race Start Analysis", plot_20_race_start_analysis),
    ]
    
    results = {}
    for name, func in plots:
        try:
            print(f"  📈 {name}...", end=" ")
            if name == "Race Summary":
                success = func(data_dir, PLOTS_DIR, event_info)
            else:
                success = func(data_dir, PLOTS_DIR)
            
            if success:
                print("✅")
            else:
                print("⚠️")
            results[name] = success
        except Exception as e:
            print(f"❌ Error: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VISUALIZATION SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n✅ Successfully created: {successful}/{total} plots")
    
    if successful < total:
        failed = [k for k, v in results.items() if not v]
        print(f"⚠️  Failed/Skipped: {', '.join(failed)}")
    
    print(f"\n📁 All plots saved to: {PLOTS_DIR}/")
    
    # List all plots
    print("\n📄 Generated plots:")
    for filename in sorted(os.listdir(PLOTS_DIR)):
        if filename.endswith('.png'):
            filepath = os.path.join(PLOTS_DIR, filename)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"   - {filename} ({size_kb:.1f} KB)")
    
    print("\n🏁 Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
