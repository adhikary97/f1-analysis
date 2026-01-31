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

# Tire compound colors
COMPOUND_COLORS = {
    'SOFT': '#FF3333',
    'MEDIUM': '#FFF200',
    'HARD': '#EBEBEB',
    'INTERMEDIATE': '#43B02A',
    'WET': '#0067AD',
    'UNKNOWN': '#888888',
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
    """Create lap time evolution showing tire degradation."""
    laps_df = load_csv(data_dir, 'laps.csv')
    stints_df = load_csv(data_dir, 'stints.csv')
    
    if laps_df is None or laps_df.empty:
        print("  ⚠️  No lap data for lap time evolution")
        return False
    
    if 'LapTime_seconds' not in laps_df.columns:
        print("  ⚠️  No lap time data")
        return False
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Get top 5 drivers by average pace
    valid_laps = laps_df[(laps_df['LapTime_seconds'] > 0) & (laps_df['LapTime_seconds'] < 200)]
    median_time = valid_laps['LapTime_seconds'].median()
    valid_laps = valid_laps[valid_laps['LapTime_seconds'] < median_time * 1.15]
    
    driver_avg = valid_laps.groupby('Driver')['LapTime_seconds'].mean().nsmallest(5)
    top_drivers = driver_avg.index.tolist()
    
    for driver in top_drivers:
        driver_laps = valid_laps[valid_laps['Driver'] == driver].sort_values('LapNumber')
        
        # Color by compound if available
        if 'Compound' in driver_laps.columns:
            for compound in driver_laps['Compound'].unique():
                compound_laps = driver_laps[driver_laps['Compound'] == compound]
                color = COMPOUND_COLORS.get(str(compound).upper(), COMPOUND_COLORS['UNKNOWN'])
                ax.scatter(compound_laps['LapNumber'], compound_laps['LapTime_seconds'],
                          color=color, s=30, alpha=0.7, label=f"{driver} - {compound}")
        else:
            ax.plot(driver_laps['LapNumber'], driver_laps['LapTime_seconds'],
                   marker='o', markersize=4, label=str(driver), alpha=0.7)
    
    ax.set_xlabel('Lap Number', fontsize=12)
    ax.set_ylabel('Lap Time (seconds)', fontsize=12)
    ax.set_title('Lap Time Evolution (Tire Degradation)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '09_lap_time_evolution.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    return True


def plot_10_race_control_timeline(data_dir, plots_dir):
    """Create a timeline of race control messages."""
    rc_df = load_csv(data_dir, 'race_control.csv')
    laps_df = load_csv(data_dir, 'laps.csv')
    
    if rc_df is None or rc_df.empty:
        print("  ⚠️  No race control data")
        return False
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Get total laps
    total_laps = laps_df['LapNumber'].max() if laps_df is not None else 60
    
    # Color by category
    category_colors = {
        'Flag': '#FFFF00',
        'SafetyCar': '#FFA500',
        'Drs': '#00FF00',
        'CarEvent': '#FF0000',
        'Other': '#888888',
    }
    
    categories_seen = set()
    
    for i, (_, row) in enumerate(rc_df.iterrows()):
        category = row.get('Category', 'Other')
        message = row.get('Message', '')
        lap = row.get('Lap', i + 1)
        
        if pd.isna(lap):
            lap = i + 1
        
        color = category_colors.get(category, category_colors['Other'])
        categories_seen.add(category)
        
        ax.axvline(x=lap, color=color, alpha=0.5, linewidth=2)
        
        # Add text for important messages
        if category in ['Flag', 'SafetyCar']:
            ax.text(lap, 0.5 + (i % 5) * 0.1, str(message)[:40], 
                   rotation=90, fontsize=8, va='bottom', ha='right')
    
    ax.set_xlim(0, total_laps + 5)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Lap Number', fontsize=12)
    ax.set_title('Race Control Messages Timeline', fontsize=16, fontweight='bold', pad=20)
    ax.set_yticks([])
    
    # Add legend
    legend_patches = [mpatches.Patch(color=color, label=cat) 
                      for cat, color in category_colors.items() if cat in categories_seen]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '10_race_control_timeline.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
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
