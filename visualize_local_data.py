"""
F1 Race Analysis & Visualization - From Local CSV Data

Analyzes race data from the 2025 Abu Dhabi GP CSV files.
Generates comprehensive visualizations for race analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Set style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Data directory and output directory
DATA_DIR = "race_data_Race_2025-12-07"
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load all CSV files into DataFrames."""
    print("Loading data from CSV files...")
    
    data = {}
    
    # Load all available CSV files
    csv_files = [
        'drivers', 'laps', 'stints', 'pit_stops', 'weather',
        'positions', 'race_control', 'intervals', 'session_results',
        'sessions', 'meeting', 'overtakes', 'team_radio'
    ]
    
    for file in csv_files:
        filepath = os.path.join(DATA_DIR, f"{file}.csv")
        if os.path.exists(filepath):
            data[file] = pd.read_csv(filepath)
            print(f"  Loaded {file}.csv: {len(data[file])} rows")
        else:
            print(f"  Warning: {file}.csv not found")
            data[file] = pd.DataFrame()
    
    return data


def get_driver_info(data):
    """Create a mapping of driver number to driver info."""
    drivers_df = data['drivers']
    driver_map = {}
    
    for _, row in drivers_df.iterrows():
        driver_map[row['driver_number']] = {
            'name': row['name_acronym'],
            'full_name': row['full_name'],
            'team': row['team_name'],
            'color': f"#{row['team_colour']}" if pd.notna(row['team_colour']) else '#888888'
        }
    
    return driver_map


def get_team_colors():
    """Return team colors for consistent plotting."""
    return {
        'Red Bull Racing': '#4781D7',
        'McLaren': '#F47600',
        'Ferrari': '#ED1131',
        'Mercedes': '#00D7B6',
        'Aston Martin': '#229971',
        'Alpine': '#00A1E8',
        'Williams': '#1868DB',
        'Racing Bulls': '#6C98FF',
        'Kick Sauber': '#01C00E',
        'Haas F1 Team': '#9C9FA2'
    }


# =============================================================================
# RACE SUMMARY
# =============================================================================

def plot_race_summary(data, driver_map):
    """Create a comprehensive race summary dashboard."""
    print("\n📊 Generating race summary dashboard...")
    
    fig = plt.figure(figsize=(18, 12))
    
    # Get session info
    sessions = data['sessions']
    race_session = sessions[sessions['session_type'] == 'Race'].iloc[0]
    
    fig.suptitle(f"2025 Abu Dhabi Grand Prix - Race Analysis\n{race_session['location']}, {race_session['country_name']}", 
                fontsize=16, fontweight='bold', y=0.98)
    
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # 1. Final Classification
    ax1 = fig.add_subplot(gs[0, 0])
    results = data['session_results'].sort_values('position')
    
    positions = results['position'].values[:10]
    driver_names = [driver_map.get(dn, {}).get('name', str(dn)) for dn in results['driver_number'].values[:10]]
    points = results['points'].values[:10]
    
    team_colors = get_team_colors()
    colors = [team_colors.get(driver_map.get(dn, {}).get('team', ''), '#888888') 
              for dn in results['driver_number'].values[:10]]
    
    bars = ax1.barh(driver_names[::-1], points[::-1], color=colors[::-1])
    ax1.set_xlabel('Points')
    ax1.set_title('Final Classification (Top 10)', fontweight='bold')
    
    # Add point labels
    for bar, pt in zip(bars, points[::-1]):
        if pt > 0:
            ax1.text(pt + 0.3, bar.get_y() + bar.get_height()/2, 
                    f'{int(pt)}', va='center', fontsize=9)
    
    # 2. Gap to Winner
    ax2 = fig.add_subplot(gs[0, 1])
    
    finishers = results[results['gap_to_leader'] != '+1 LAP'].copy()
    finishers['gap_numeric'] = pd.to_numeric(finishers['gap_to_leader'], errors='coerce')
    finishers = finishers.dropna(subset=['gap_numeric']).head(10)
    
    driver_names_gap = [driver_map.get(dn, {}).get('name', str(dn)) for dn in finishers['driver_number'].values]
    gaps = finishers['gap_numeric'].values
    
    colors_gap = [team_colors.get(driver_map.get(dn, {}).get('team', ''), '#888888') 
                  for dn in finishers['driver_number'].values]
    
    ax2.barh(driver_names_gap[::-1], gaps[::-1], color=colors_gap[::-1])
    ax2.set_xlabel('Gap to Winner (seconds)')
    ax2.set_title('Gap to Race Winner', fontweight='bold')
    
    # 3. Weather Summary
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    weather = data['weather']
    if len(weather) > 0:
        first_w = weather.iloc[0]
        last_w = weather.iloc[-1]
        
        weather_text = f"""
WEATHER CONDITIONS

Race Start:
  Air Temp: {first_w['air_temperature']}°C
  Track Temp: {first_w['track_temperature']}°C
  Humidity: {first_w['humidity']}%
  
Race End:
  Air Temp: {last_w['air_temperature']}°C
  Track Temp: {last_w['track_temperature']}°C
  Humidity: {last_w['humidity']}%
  
Rain: {'Yes' if weather['rainfall'].sum() > 0 else 'No'}
"""
        ax3.text(0.1, 0.5, weather_text, transform=ax3.transAxes,
                fontsize=11, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax3.set_title('Weather Summary', fontweight='bold')
    
    # 4. Pit Stop Analysis
    ax4 = fig.add_subplot(gs[1, 0])
    
    pit_stops = data['pit_stops'].copy()
    if len(pit_stops) > 0:
        pit_counts = pit_stops.groupby('driver_number').size().reset_index(name='stops')
        pit_counts['driver_name'] = pit_counts['driver_number'].apply(
            lambda x: driver_map.get(x, {}).get('name', str(x)))
        pit_counts = pit_counts.sort_values('stops', ascending=False).head(10)
        
        colors_pit = [team_colors.get(driver_map.get(dn, {}).get('team', ''), '#888888') 
                      for dn in pit_counts['driver_number'].values]
        
        ax4.bar(pit_counts['driver_name'], pit_counts['stops'], color=colors_pit)
        ax4.set_ylabel('Number of Pit Stops')
        ax4.set_title('Pit Stop Count', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
    
    # 5. Fastest Pit Stops
    ax5 = fig.add_subplot(gs[1, 1])
    
    if len(pit_stops) > 0 and 'stop_duration' in pit_stops.columns:
        valid_stops = pit_stops.dropna(subset=['stop_duration'])
        valid_stops = valid_stops[valid_stops['stop_duration'] > 0].copy()
        
        if len(valid_stops) > 0:
            fastest_stops = valid_stops.nsmallest(10, 'stop_duration')
            fastest_stops['driver_name'] = fastest_stops['driver_number'].apply(
                lambda x: driver_map.get(x, {}).get('name', str(x)))
            
            colors_fast = [team_colors.get(driver_map.get(dn, {}).get('team', ''), '#888888') 
                          for dn in fastest_stops['driver_number'].values]
            
            ax5.barh(fastest_stops['driver_name'].values[::-1], 
                    fastest_stops['stop_duration'].values[::-1],
                    color=colors_fast[::-1])
            ax5.set_xlabel('Pit Stop Duration (seconds)')
            ax5.set_title('Fastest Pit Stops', fontweight='bold')
    
    # 6. Race Control Events Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    race_control = data['race_control']
    if len(race_control) > 0:
        event_counts = race_control['category'].value_counts()
        
        events_text = "RACE CONTROL EVENTS\n\n"
        for cat, count in event_counts.items():
            events_text += f"  {cat}: {count}\n"
        
        # Count penalties
        penalties = race_control[race_control['message'].str.contains('PENALTY', case=False, na=False)]
        events_text += f"\n  Total Penalties: {len(penalties)}"
        
        ax6.text(0.1, 0.5, events_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        ax6.set_title('Race Events', fontweight='bold')
    
    plt.savefig(f"{OUTPUT_DIR}/00_race_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/00_race_summary.png")


# =============================================================================
# LAP TIME ANALYSIS
# =============================================================================

def plot_lap_times_comparison(data, driver_map, top_n=8):
    """Plot lap times for top drivers throughout the race."""
    print("\n📊 Generating lap times comparison...")
    
    laps = data['laps'].copy()
    results = data['session_results'].sort_values('position')
    
    # Get top N finishers
    top_drivers = results['driver_number'].head(top_n).tolist()
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    team_colors = get_team_colors()
    
    for driver_num in top_drivers:
        driver_laps = laps[laps['driver_number'] == driver_num].copy()
        
        # Filter out invalid laps (pit laps, slow laps)
        driver_laps = driver_laps[driver_laps['lap_duration'] < 100]
        driver_laps = driver_laps[driver_laps['lap_duration'] > 80]
        
        if len(driver_laps) > 0:
            driver_info = driver_map.get(driver_num, {})
            color = team_colors.get(driver_info.get('team', ''), '#888888')
            label = f"{driver_info.get('name', str(driver_num))} ({driver_info.get('team', '')})"
            
            ax.plot(driver_laps['lap_number'], driver_laps['lap_duration'],
                   marker='o', markersize=3, linewidth=1.5,
                   label=label, color=color, alpha=0.8)
    
    ax.set_xlabel('Lap Number', fontsize=12)
    ax.set_ylabel('Lap Time (seconds)', fontsize=12)
    ax.set_title('Lap Time Evolution - Top 8 Finishers', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 60)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_lap_times_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/01_lap_times_comparison.png")


def plot_sector_times_heatmap(data, driver_map):
    """Create heatmap of average sector times for all drivers."""
    print("\n📊 Generating sector times heatmap...")
    
    laps = data['laps'].copy()
    results = data['session_results'].sort_values('position')
    
    sector_data = []
    
    for driver_num in results['driver_number'].values:
        driver_laps = laps[laps['driver_number'] == driver_num].copy()
        
        # Filter valid laps
        valid = driver_laps.dropna(subset=['duration_sector_1', 'duration_sector_2', 'duration_sector_3'])
        valid = valid[(valid['duration_sector_1'] > 15) & (valid['duration_sector_1'] < 25)]
        
        if len(valid) > 5:
            driver_info = driver_map.get(driver_num, {})
            sector_data.append({
                'Driver': driver_info.get('name', str(driver_num)),
                'Sector 1': valid['duration_sector_1'].mean(),
                'Sector 2': valid['duration_sector_2'].mean(),
                'Sector 3': valid['duration_sector_3'].mean()
            })
    
    if not sector_data:
        print("  No sector data available")
        return
    
    df = pd.DataFrame(sector_data)
    df = df.set_index('Driver')
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn_r',
                ax=ax, cbar_kws={'label': 'Time (seconds)'})
    
    ax.set_title('Average Sector Times by Driver\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_sector_times_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/02_sector_times_heatmap.png")


def plot_lap_consistency(data, driver_map):
    """Box plot showing lap time consistency (variance) for drivers."""
    print("\n📊 Generating lap consistency analysis...")
    
    laps = data['laps'].copy()
    results = data['session_results'].sort_values('position')
    
    # Get top 12 finishers
    top_drivers = results['driver_number'].head(12).tolist()
    
    lap_data = []
    for driver_num in top_drivers:
        driver_laps = laps[laps['driver_number'] == driver_num].copy()
        
        # Filter valid racing laps
        valid = driver_laps[(driver_laps['lap_duration'] > 85) & 
                           (driver_laps['lap_duration'] < 95) &
                           (driver_laps['lap_number'] > 1)]
        
        driver_info = driver_map.get(driver_num, {})
        for _, lap in valid.iterrows():
            lap_data.append({
                'Driver': driver_info.get('name', str(driver_num)),
                'Lap Time': lap['lap_duration'],
                'Team': driver_info.get('team', '')
            })
    
    if not lap_data:
        print("  No lap data available")
        return
    
    df = pd.DataFrame(lap_data)
    
    # Order drivers by finishing position
    driver_order = [driver_map.get(d, {}).get('name', str(d)) for d in top_drivers]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    team_colors = get_team_colors()
    palette = {driver_map.get(d, {}).get('name', str(d)): 
               team_colors.get(driver_map.get(d, {}).get('team', ''), '#888888')
               for d in top_drivers}
    
    sns.boxplot(data=df, x='Driver', y='Lap Time', ax=ax, 
                order=driver_order, hue='Driver', palette=palette, legend=False)
    
    ax.set_xlabel('Driver', fontsize=12)
    ax.set_ylabel('Lap Time (seconds)', fontsize=12)
    ax.set_title('Lap Time Consistency - Distribution Analysis\n(Smaller box = more consistent)', 
                fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_lap_consistency.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/03_lap_consistency.png")


def plot_pace_degradation(data, driver_map):
    """Plot pace degradation over stints (tyre wear analysis)."""
    print("\n📊 Generating pace degradation analysis...")
    
    laps = data['laps'].copy()
    stints = data['stints'].copy()
    results = data['session_results'].sort_values('position')
    
    # Get top 4 finishers
    top_drivers = results['driver_number'].head(4).tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    compound_colors = {'SOFT': '#FF3333', 'MEDIUM': '#FFDD00', 'HARD': '#FFFFFF'}
    compound_edge = {'SOFT': '#CC0000', 'MEDIUM': '#CC9900', 'HARD': '#888888'}
    
    for idx, driver_num in enumerate(top_drivers):
        ax = axes[idx]
        driver_info = driver_map.get(driver_num, {})
        
        driver_stints = stints[stints['driver_number'] == driver_num].copy()
        driver_laps = laps[laps['driver_number'] == driver_num].copy()
        
        # Filter valid laps
        valid_laps = driver_laps[(driver_laps['lap_duration'] > 85) & 
                                 (driver_laps['lap_duration'] < 95)]
        
        for _, stint in driver_stints.iterrows():
            stint_laps = valid_laps[(valid_laps['lap_number'] >= stint['lap_start']) &
                                    (valid_laps['lap_number'] <= stint['lap_end'])]
            
            if len(stint_laps) > 0:
                compound = stint['compound']
                color = compound_colors.get(compound, '#888888')
                edge = compound_edge.get(compound, '#444444')
                
                ax.scatter(stint_laps['lap_number'], stint_laps['lap_duration'],
                          c=color, edgecolors=edge, s=50, alpha=0.8, label=compound)
                
                # Add trend line
                if len(stint_laps) > 3:
                    z = np.polyfit(stint_laps['lap_number'], stint_laps['lap_duration'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(stint_laps['lap_number'].min(), 
                                        stint_laps['lap_number'].max(), 50)
                    ax.plot(x_line, p(x_line), '--', color=edge, alpha=0.6, linewidth=1.5)
        
        ax.set_xlabel('Lap Number', fontsize=10)
        ax.set_ylabel('Lap Time (seconds)', fontsize=10)
        ax.set_title(f"{driver_info.get('name', str(driver_num))} - {driver_info.get('team', '')}", 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 60)
        ax.set_ylim(86, 94)
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
    
    plt.suptitle('Pace Degradation by Stint (Tyre Wear Analysis)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_pace_degradation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/04_pace_degradation.png")


# =============================================================================
# TYRE STRATEGY
# =============================================================================

def plot_tyre_strategy(data, driver_map):
    """Visualize tyre strategy for all drivers."""
    print("\n📊 Generating tyre strategy visualization...")
    
    stints = data['stints'].copy()
    results = data['session_results'].sort_values('position')
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    compound_colors = {'SOFT': '#FF3333', 'MEDIUM': '#FFDD00', 'HARD': '#EEEEEE'}
    
    y_positions = {}
    for i, driver_num in enumerate(results['driver_number'].values):
        y_positions[driver_num] = len(results) - i
    
    for _, stint in stints.iterrows():
        driver_num = stint['driver_number']
        if driver_num not in y_positions:
            continue
            
        y = y_positions[driver_num]
        start = stint['lap_start']
        end = stint['lap_end']
        compound = stint['compound']
        color = compound_colors.get(compound, '#888888')
        
        ax.barh(y, end - start + 1, left=start, height=0.6,
               color=color, edgecolor='black', linewidth=0.5)
    
    # Set y-axis labels
    y_labels = [driver_map.get(dn, {}).get('name', str(dn)) 
                for dn in results['driver_number'].values]
    ax.set_yticks(range(len(y_labels), 0, -1))
    ax.set_yticklabels(y_labels)
    
    ax.set_xlabel('Lap Number', fontsize=12)
    ax.set_title('Tyre Strategy - All Drivers', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 60)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#FF3333', edgecolor='black', label='Soft'),
        mpatches.Patch(facecolor='#FFDD00', edgecolor='black', label='Medium'),
        mpatches.Patch(facecolor='#EEEEEE', edgecolor='black', label='Hard')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_tyre_strategy.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/05_tyre_strategy.png")


# =============================================================================
# POSITION CHANGES
# =============================================================================

def plot_position_changes(data, driver_map):
    """Plot position changes throughout the race."""
    print("\n📊 Generating position changes chart...")
    
    laps = data['laps'].copy()
    results = data['session_results'].sort_values('position')
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    team_colors = get_team_colors()
    
    # Create position by lap for each driver
    for driver_num in results['driver_number'].values:
        driver_laps = laps[laps['driver_number'] == driver_num].sort_values('lap_number')
        
        if len(driver_laps) > 0:
            # Calculate position based on cumulative time
            driver_info = driver_map.get(driver_num, {})
            color = team_colors.get(driver_info.get('team', ''), '#888888')
            
            # Use lap number as x, would need to calculate actual positions
            ax.plot(driver_laps['lap_number'], range(1, len(driver_laps) + 1),
                   color=color, alpha=0.6, linewidth=1)
    
    ax.set_xlabel('Lap Number', fontsize=12)
    ax.set_ylabel('Position', fontsize=12)
    ax.set_title('Race Progression', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0, 60)
    ax.set_ylim(20, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_position_changes.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/06_position_changes.png")


# =============================================================================
# INTERVALS ANALYSIS
# =============================================================================

def plot_gap_evolution(data, driver_map):
    """Plot gap to leader evolution."""
    print("\n📊 Generating gap evolution chart...")
    
    intervals = data['intervals'].copy()
    results = data['session_results'].sort_values('position')
    
    # Get top 6 finishers
    top_drivers = results['driver_number'].head(6).tolist()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    team_colors = get_team_colors()
    
    for driver_num in top_drivers:
        driver_intervals = intervals[intervals['driver_number'] == driver_num].copy()
        driver_intervals = driver_intervals.dropna(subset=['gap_to_leader'])
        driver_intervals['gap_numeric'] = pd.to_numeric(driver_intervals['gap_to_leader'], errors='coerce')
        driver_intervals = driver_intervals.dropna(subset=['gap_numeric'])
        
        if len(driver_intervals) > 10:
            driver_info = driver_map.get(driver_num, {})
            color = team_colors.get(driver_info.get('team', ''), '#888888')
            label = driver_info.get('name', str(driver_num))
            
            # Sample to reduce noise
            sampled = driver_intervals.iloc[::10]
            
            ax.plot(range(len(sampled)), sampled['gap_numeric'],
                   label=label, color=color, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Time (race progression)', fontsize=12)
    ax.set_ylabel('Gap to Leader (seconds)', fontsize=12)
    ax.set_title('Gap to Leader Evolution', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_gap_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/07_gap_evolution.png")


# =============================================================================
# WEATHER EVOLUTION
# =============================================================================

def plot_weather_evolution(data):
    """Plot weather conditions throughout the race."""
    print("\n📊 Generating weather evolution chart...")
    
    weather = data['weather'].copy()
    
    if len(weather) == 0:
        print("  No weather data available")
        return
    
    weather['date'] = pd.to_datetime(weather['date'])
    weather = weather.sort_values('date')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Air temperature
    axes[0, 0].plot(range(len(weather)), weather['air_temperature'], 
                   color='#E63946', linewidth=2)
    axes[0, 0].set_ylabel('Air Temperature (°C)')
    axes[0, 0].set_title('Air Temperature')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Track temperature
    axes[0, 1].plot(range(len(weather)), weather['track_temperature'], 
                   color='#457B9D', linewidth=2)
    axes[0, 1].set_ylabel('Track Temperature (°C)')
    axes[0, 1].set_title('Track Temperature')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Humidity
    axes[1, 0].plot(range(len(weather)), weather['humidity'], 
                   color='#2A9D8F', linewidth=2)
    axes[1, 0].set_ylabel('Humidity (%)')
    axes[1, 0].set_title('Humidity')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Wind speed
    axes[1, 1].plot(range(len(weather)), weather['wind_speed'], 
                   color='#E9C46A', linewidth=2)
    axes[1, 1].set_ylabel('Wind Speed (m/s)')
    axes[1, 1].set_title('Wind Speed')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Weather Evolution Throughout the Race', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/08_weather_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/08_weather_evolution.png")


# =============================================================================
# SPEED ANALYSIS
# =============================================================================

def plot_speed_comparison(data, driver_map):
    """Compare speed trap and intermediate speeds."""
    print("\n📊 Generating speed comparison chart...")
    
    laps = data['laps'].copy()
    results = data['session_results'].sort_values('position')
    
    speed_data = []
    
    for driver_num in results['driver_number'].values:
        driver_laps = laps[laps['driver_number'] == driver_num].copy()
        
        valid = driver_laps.dropna(subset=['st_speed', 'i1_speed', 'i2_speed'])
        
        if len(valid) > 0:
            driver_info = driver_map.get(driver_num, {})
            speed_data.append({
                'Driver': driver_info.get('name', str(driver_num)),
                'Speed Trap': valid['st_speed'].max(),
                'I1 Speed': valid['i1_speed'].max(),
                'I2 Speed': valid['i2_speed'].max(),
                'Team': driver_info.get('team', '')
            })
    
    if not speed_data:
        print("  No speed data available")
        return
    
    df = pd.DataFrame(speed_data)
    df = df.sort_values('Speed Trap', ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(df))
    width = 0.25
    
    bars1 = ax.bar(x - width, df['Speed Trap'], width, label='Speed Trap', color='#E63946')
    bars2 = ax.bar(x, df['I1 Speed'], width, label='Intermediate 1', color='#457B9D')
    bars3 = ax.bar(x + width, df['I2 Speed'], width, label='Intermediate 2', color='#2A9D8F')
    
    ax.set_xlabel('Driver', fontsize=12)
    ax.set_ylabel('Speed (km/h)', fontsize=12)
    ax.set_title('Maximum Speeds Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Driver'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/09_speed_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/09_speed_comparison.png")


# =============================================================================
# FASTEST LAPS
# =============================================================================

def plot_fastest_laps(data, driver_map):
    """Bar chart of fastest laps."""
    print("\n📊 Generating fastest laps chart...")
    
    laps = data['laps'].copy()
    results = data['session_results'].sort_values('position')
    
    fastest_laps = []
    
    for driver_num in results['driver_number'].values:
        driver_laps = laps[laps['driver_number'] == driver_num].copy()
        valid = driver_laps[(driver_laps['lap_duration'] > 85) & 
                           (driver_laps['lap_duration'] < 92)]
        
        if len(valid) > 0:
            fastest = valid.loc[valid['lap_duration'].idxmin()]
            driver_info = driver_map.get(driver_num, {})
            fastest_laps.append({
                'Driver': driver_info.get('name', str(driver_num)),
                'Lap Time': fastest['lap_duration'],
                'Lap Number': fastest['lap_number'],
                'Team': driver_info.get('team', '')
            })
    
    if not fastest_laps:
        print("  No fastest lap data available")
        return
    
    df = pd.DataFrame(fastest_laps)
    df = df.sort_values('Lap Time')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate gap to fastest
    fastest_time = df['Lap Time'].min()
    df['Gap'] = df['Lap Time'] - fastest_time
    
    team_colors = get_team_colors()
    colors = [team_colors.get(row['Team'], '#888888') for _, row in df.iterrows()]
    
    bars = ax.barh(df['Driver'].values[::-1], df['Gap'].values[::-1], color=colors[::-1])
    
    # Add time labels
    for i, (bar, row) in enumerate(zip(bars, df.iloc[::-1].iterrows())):
        _, row_data = row
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
               f"{row_data['Lap Time']:.3f}s (Lap {int(row_data['Lap Number'])})",
               va='center', fontsize=9)
    
    ax.set_xlabel('Gap to Fastest (seconds)', fontsize=12)
    ax.set_title('Fastest Lap Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/10_fastest_laps.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/10_fastest_laps.png")


# =============================================================================
# PENALTIES TIMELINE
# =============================================================================

def plot_penalties_timeline(data, driver_map):
    """Visualize penalties and incidents throughout the race."""
    print("\n📊 Generating penalties timeline...")
    
    race_control = data['race_control'].copy()
    
    # Filter for penalty-related messages
    penalties = race_control[
        race_control['message'].str.contains('PENALTY|INVESTIGATION|BLACK AND WHITE', 
                                             case=False, na=False)
    ].copy()
    
    if len(penalties) == 0:
        print("  No penalty data available")
        return
    
    penalties = penalties.sort_values('lap_number')
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Categorize penalties
    colors = {
        'PENALTY': '#E63946',
        'INVESTIGATION': '#F4A261',
        'BLACK AND WHITE': '#FFDD00',
        'OTHER': '#888888'
    }
    
    for i, (_, row) in enumerate(penalties.iterrows()):
        msg = row['message']
        if 'PENALTY' in msg.upper():
            color = colors['PENALTY']
            marker = 's'
        elif 'INVESTIGATION' in msg.upper():
            color = colors['INVESTIGATION']
            marker = '^'
        elif 'BLACK AND WHITE' in msg.upper():
            color = colors['BLACK AND WHITE']
            marker = 'o'
        else:
            color = colors['OTHER']
            marker = 'x'
        
        ax.scatter(row['lap_number'], i, c=color, marker=marker, s=100, zorder=5)
        
        # Truncate message for display
        short_msg = msg[:60] + '...' if len(msg) > 60 else msg
        ax.text(row['lap_number'] + 1, i, short_msg, fontsize=7, va='center')
    
    ax.set_xlabel('Lap Number', fontsize=12)
    ax.set_ylabel('Incident Index', fontsize=12)
    ax.set_title('Race Incidents & Penalties Timeline', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 65)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['PENALTY'], 
                   markersize=10, label='Penalty'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=colors['INVESTIGATION'], 
                   markersize=10, label='Investigation'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['BLACK AND WHITE'], 
                   markersize=10, label='Black & White Flag'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/11_penalties_timeline.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/11_penalties_timeline.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("🏎️  F1 RACE ANALYSIS - 2025 ABU DHABI GP")
    print("=" * 70)
    
    # Load data
    data = load_data()
    
    # Get driver info
    driver_map = get_driver_info(data)
    print(f"\nLoaded {len(driver_map)} drivers")
    
    # Print race results
    results = data['session_results'].sort_values('position')
    print("\n🏁 RACE RESULTS:")
    print("-" * 50)
    for i, (_, row) in enumerate(results.head(10).iterrows()):
        driver = driver_map.get(row['driver_number'], {})
        gap = row['gap_to_leader'] if row['gap_to_leader'] != 0 else 'WINNER'
        print(f"  P{int(row['position']):2} | {driver.get('name', '???'):3} | {driver.get('team', ''):15} | {gap}")
    
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # Generate all visualizations
    plot_race_summary(data, driver_map)
    plot_lap_times_comparison(data, driver_map)
    plot_sector_times_heatmap(data, driver_map)
    plot_lap_consistency(data, driver_map)
    plot_pace_degradation(data, driver_map)
    plot_tyre_strategy(data, driver_map)
    plot_gap_evolution(data, driver_map)
    plot_weather_evolution(data)
    plot_speed_comparison(data, driver_map)
    plot_fastest_laps(data, driver_map)
    plot_penalties_timeline(data, driver_map)
    
    print("\n" + "=" * 70)
    print(f"✅ All visualizations saved to '{OUTPUT_DIR}/' directory")
    print("=" * 70)
    
    # Print list of generated files
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            print(f"  - {OUTPUT_DIR}/{f}")


if __name__ == "__main__":
    main()
