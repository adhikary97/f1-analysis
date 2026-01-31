"""
F1 Race Analysis & Visualization Script

Pulls data from OpenF1 API endpoints:
- car_data: Driver inputs, braking, throttle, DRS
- laps: Lap times, sector times, consistency
- location: Racing lines, track positions

Generates multiple visualizations for race analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from openf1_client import OpenF1Client
from datetime import datetime
import os

# Set style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory for plots
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_latest_race_session(client, session_key=None):
    """Get the most recent race session with available data."""
    print("Fetching race session...")
    
    if session_key:
        # Use specified session
        sessions = client.sessions.list(session_key=session_key)
        if sessions:
            session = sessions[0]
            print(f"Using specified session: {session.session_name} at {session.location}")
            print(f"Session Key: {session.session_key}")
            return session
    
    sessions = client.sessions.list()
    race_sessions = [s for s in sessions if s.session_type == "Race"]
    
    if not race_sessions:
        raise ValueError("No race sessions found")
    
    # Try sessions from most recent until we find one with driver data
    for race in reversed(race_sessions[-10:]):
        drivers = client.drivers.list(session_key=race.session_key)
        if len(drivers) > 0:
            print(f"Found: {race.session_name} at {race.location}, {race.country_name}")
            print(f"Session Key: {race.session_key}")
            return race
    
    # Fallback to latest even if no drivers
    latest = race_sessions[-1]
    print(f"Found: {latest.session_name} at {latest.location}, {latest.country_name}")
    print(f"Session Key: {latest.session_key}")
    return latest


def get_drivers(client, session_key):
    """Get all drivers for a session."""
    drivers = client.drivers.list(session_key=session_key)
    driver_map = {d.driver_number: d for d in drivers}
    return drivers, driver_map


# =============================================================================
# LAP TIME ANALYSIS
# =============================================================================

def plot_lap_times_comparison(client, session_key, driver_map, top_n=5):
    """
    Plot lap times for top drivers throughout the race.
    Shows pace evolution and consistency.
    """
    print("\n📊 Generating lap times comparison...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get laps for all drivers and find top performers
    all_fastest = []
    for driver_num, driver in driver_map.items():
        fastest = client.laps.get_fastest_lap(
            session_key=session_key,
            driver_number=driver_num
        )
        if fastest and fastest.lap_duration:
            all_fastest.append((driver_num, driver.name_acronym, fastest.lap_duration))
    
    # Sort and get top N drivers
    all_fastest.sort(key=lambda x: x[2])
    top_drivers = [d[0] for d in all_fastest[:top_n]]
    
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))
    
    for idx, driver_num in enumerate(top_drivers):
        laps = client.laps.list(
            session_key=session_key,
            driver_number=driver_num
        )
        
        if not laps:
            continue
            
        # Filter out pit laps and invalid laps
        valid_laps = [l for l in laps if l.lap_duration and l.lap_duration < 200]
        
        if valid_laps:
            lap_numbers = [l.lap_number for l in valid_laps]
            lap_times = [l.lap_duration for l in valid_laps]
            
            driver = driver_map[driver_num]
            ax.plot(lap_numbers, lap_times, 
                   marker='o', markersize=3, linewidth=1.5,
                   label=f"{driver.name_acronym} ({driver.team_name})",
                   color=colors[idx], alpha=0.8)
    
    ax.set_xlabel('Lap Number', fontsize=12)
    ax.set_ylabel('Lap Time (seconds)', fontsize=12)
    ax.set_title('Lap Time Evolution - Top 5 Drivers', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_lap_times_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/01_lap_times_comparison.png")


def plot_sector_times_heatmap(client, session_key, driver_map, top_n=10):
    """
    Create heatmap of average sector times for top drivers.
    """
    print("\n📊 Generating sector times heatmap...")
    
    sector_data = []
    
    for driver_num, driver in list(driver_map.items())[:top_n]:
        laps = client.laps.list(
            session_key=session_key,
            driver_number=driver_num
        )
        
        valid_laps = [l for l in laps if l.duration_sector_1 and l.duration_sector_2 and l.duration_sector_3]
        
        if valid_laps:
            avg_s1 = np.mean([l.duration_sector_1 for l in valid_laps])
            avg_s2 = np.mean([l.duration_sector_2 for l in valid_laps])
            avg_s3 = np.mean([l.duration_sector_3 for l in valid_laps])
            
            sector_data.append({
                'Driver': driver.name_acronym,
                'Sector 1': avg_s1,
                'Sector 2': avg_s2,
                'Sector 3': avg_s3
            })
    
    if not sector_data:
        print("  No sector data available")
        return
        
    df = pd.DataFrame(sector_data)
    df = df.set_index('Driver')
    
    # Normalize for heatmap (lower is better, so invert colormap)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                ax=ax, cbar_kws={'label': 'Time (seconds)'})
    
    ax.set_title('Average Sector Times by Driver', fontsize=14, fontweight='bold')
    ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_sector_times_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/02_sector_times_heatmap.png")


def plot_lap_consistency(client, session_key, driver_map, top_n=8):
    """
    Box plot showing lap time consistency (variance) for top drivers.
    """
    print("\n📊 Generating lap consistency analysis...")
    
    lap_data = []
    
    for driver_num, driver in driver_map.items():
        laps = client.laps.list(
            session_key=session_key,
            driver_number=driver_num
        )
        
        # Filter valid racing laps (exclude pit laps, first lap, etc.)
        valid_laps = [l for l in laps 
                     if l.lap_duration and l.lap_duration < 150 and l.lap_number > 1]
        
        for lap in valid_laps:
            lap_data.append({
                'Driver': driver.name_acronym,
                'Lap Time': lap.lap_duration,
                'Team': driver.team_name
            })
    
    if not lap_data:
        print("  No lap data available")
        return
        
    df = pd.DataFrame(lap_data)
    
    # Get drivers with most laps for comparison
    top_drivers = df.groupby('Driver').size().nlargest(top_n).index.tolist()
    df_filtered = df[df['Driver'].isin(top_drivers)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.boxplot(data=df_filtered, x='Driver', y='Lap Time', ax=ax, palette='Set2')
    
    ax.set_xlabel('Driver', fontsize=12)
    ax.set_ylabel('Lap Time (seconds)', fontsize=12)
    ax.set_title('Lap Time Consistency - Distribution Analysis', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_lap_consistency.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/03_lap_consistency.png")


def plot_pace_degradation(client, session_key, driver_map, driver_nums=None):
    """
    Plot pace degradation over stints (tyre wear analysis).
    """
    print("\n📊 Generating pace degradation analysis...")
    
    if driver_nums is None:
        # Get top 3 drivers by fastest lap
        all_fastest = []
        for driver_num, driver in driver_map.items():
            fastest = client.laps.get_fastest_lap(
                session_key=session_key,
                driver_number=driver_num
            )
            if fastest and fastest.lap_duration:
                all_fastest.append((driver_num, fastest.lap_duration))
        all_fastest.sort(key=lambda x: x[1])
        driver_nums = [d[0] for d in all_fastest[:3]]
    
    if not driver_nums:
        print("  No driver data available for pace degradation")
        return
    
    fig, axes = plt.subplots(1, len(driver_nums), figsize=(5*len(driver_nums), 5), sharey=True)
    if len(driver_nums) == 1:
        axes = [axes]
    
    for idx, driver_num in enumerate(driver_nums):
        ax = axes[idx]
        driver = driver_map[driver_num]
        
        # Get stints
        stints = client.stints.list(
            session_key=session_key,
            driver_number=driver_num
        )
        
        # Get all laps
        laps = client.laps.list(
            session_key=session_key,
            driver_number=driver_num
        )
        
        valid_laps = [l for l in laps if l.lap_duration and l.lap_duration < 150]
        
        if not valid_laps or not stints:
            continue
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for stint_idx, stint in enumerate(stints):
            if not stint.lap_start or not stint.lap_end:
                continue
                
            stint_laps = [l for l in valid_laps 
                         if stint.lap_start <= l.lap_number <= stint.lap_end]
            
            if stint_laps:
                lap_nums = [l.lap_number for l in stint_laps]
                lap_times = [l.lap_duration for l in stint_laps]
                
                color = colors[stint_idx % len(colors)]
                compound = stint.compound if stint.compound else f"Stint {stint_idx+1}"
                
                ax.scatter(lap_nums, lap_times, c=color, s=30, alpha=0.7, label=compound)
                
                # Add trend line
                if len(lap_nums) > 2:
                    z = np.polyfit(lap_nums, lap_times, 1)
                    p = np.poly1d(z)
                    ax.plot(lap_nums, p(lap_nums), '--', color=color, alpha=0.5)
        
        ax.set_xlabel('Lap Number', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Lap Time (seconds)', fontsize=10)
        ax.set_title(f"{driver.name_acronym}", fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Pace Degradation by Stint (Tyre Wear)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_pace_degradation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/04_pace_degradation.png")


# =============================================================================
# CAR DATA (TELEMETRY) ANALYSIS
# =============================================================================

def plot_speed_trace(client, session_key, driver_map, driver_num, lap_number=None):
    """
    Plot speed trace for a specific lap.
    """
    print(f"\n📊 Generating speed trace for driver {driver_num}...")
    
    driver = driver_map.get(driver_num)
    if not driver:
        print(f"  Driver {driver_num} not found")
        return
    
    # If no lap specified, get the fastest lap
    if lap_number is None:
        fastest = client.laps.get_fastest_lap(
            session_key=session_key,
            driver_number=driver_num
        )
        if fastest:
            lap_number = fastest.lap_number
        else:
            print("  Could not find fastest lap")
            return
    
    # Get all laps to find the time range
    laps = client.laps.list(
        session_key=session_key,
        driver_number=driver_num
    )
    
    target_lap = next((l for l in laps if l.lap_number == lap_number), None)
    if not target_lap:
        print(f"  Lap {lap_number} not found")
        return
    
    # Get car data - sample to avoid too much data
    car_data = client.car_data.list(
        session_key=session_key,
        driver_number=driver_num
    )
    
    if not car_data:
        print("  No car data available")
        return
    
    # Sample the data (every 10th point)
    sampled_data = car_data[::10]
    
    # Create time axis
    times = range(len(sampled_data))
    speeds = [d.speed if d.speed else 0 for d in sampled_data]
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(times, speeds, color='#E63946', linewidth=1)
    ax.fill_between(times, speeds, alpha=0.3, color='#E63946')
    
    ax.set_xlabel('Time (samples)', fontsize=12)
    ax.set_ylabel('Speed (km/h)', fontsize=12)
    ax.set_title(f'Speed Trace - {driver.name_acronym} ({driver.team_name})', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add max speed annotation
    max_speed = max(speeds)
    max_idx = speeds.index(max_speed)
    ax.annotate(f'Max: {max_speed} km/h', 
               xy=(max_idx, max_speed), 
               xytext=(max_idx + 50, max_speed + 10),
               fontsize=10, 
               arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_speed_trace_{driver.name_acronym}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/05_speed_trace_{driver.name_acronym}.png")


def plot_throttle_brake_analysis(client, session_key, driver_map, driver_num):
    """
    Plot throttle and brake application patterns.
    """
    print(f"\n📊 Generating throttle/brake analysis for driver {driver_num}...")
    
    driver = driver_map.get(driver_num)
    if not driver:
        print(f"  Driver {driver_num} not found")
        return
    
    car_data = client.car_data.list(
        session_key=session_key,
        driver_number=driver_num
    )
    
    if not car_data:
        print("  No car data available")
        return
    
    # Sample data
    sampled = car_data[::20][:500]  # Take first 500 samples
    
    times = range(len(sampled))
    throttle = [d.throttle if d.throttle else 0 for d in sampled]
    brake = [d.brake if d.brake else 0 for d in sampled]
    speed = [d.speed if d.speed else 0 for d in sampled]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    
    # Speed
    axes[0].plot(times, speed, color='#2196F3', linewidth=1.5)
    axes[0].set_ylabel('Speed (km/h)', fontsize=10)
    axes[0].set_title(f'Driver Inputs Analysis - {driver.name_acronym}', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Throttle
    axes[1].fill_between(times, throttle, alpha=0.7, color='#4CAF50')
    axes[1].set_ylabel('Throttle %', fontsize=10)
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)
    
    # Brake
    axes[2].fill_between(times, brake, alpha=0.7, color='#F44336')
    axes[2].set_ylabel('Brake %', fontsize=10)
    axes[2].set_xlabel('Time (samples)', fontsize=10)
    axes[2].set_ylim(0, 105)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_throttle_brake_{driver.name_acronym}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/06_throttle_brake_{driver.name_acronym}.png")


def plot_drs_usage(client, session_key, driver_map, top_n=10):
    """
    Analyze DRS usage across drivers.
    """
    print("\n📊 Generating DRS usage analysis...")
    
    drs_stats = []
    
    for driver_num, driver in list(driver_map.items())[:top_n]:
        car_data = client.car_data.list(
            session_key=session_key,
            driver_number=driver_num
        )
        
        if car_data:
            # Count DRS activations (drs values > 10 typically mean DRS open)
            drs_on = sum(1 for d in car_data if d.drs and d.drs > 10)
            total = len(car_data)
            drs_percentage = (drs_on / total * 100) if total > 0 else 0
            
            drs_stats.append({
                'Driver': driver.name_acronym,
                'DRS %': drs_percentage,
                'Team': driver.team_name
            })
    
    if not drs_stats:
        print("  No DRS data available")
        return
    
    df = pd.DataFrame(drs_stats).sort_values('DRS %', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))
    bars = ax.barh(df['Driver'], df['DRS %'], color=colors)
    
    ax.set_xlabel('DRS Open Time (%)', fontsize=12)
    ax.set_ylabel('Driver', fontsize=12)
    ax.set_title('DRS Usage by Driver', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars, df['DRS %']):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
               f'{val:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_drs_usage.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/07_drs_usage.png")


def plot_gear_distribution(client, session_key, driver_map, driver_num):
    """
    Pie chart of gear usage distribution.
    """
    print(f"\n📊 Generating gear distribution for driver {driver_num}...")
    
    driver = driver_map.get(driver_num)
    if not driver:
        print(f"  Driver {driver_num} not found")
        return
    
    car_data = client.car_data.list(
        session_key=session_key,
        driver_number=driver_num
    )
    
    if not car_data:
        print("  No car data available")
        return
    
    # Count gear usage
    gear_counts = {}
    for d in car_data:
        if d.n_gear:
            gear = d.n_gear
            gear_counts[gear] = gear_counts.get(gear, 0) + 1
    
    if not gear_counts:
        print("  No gear data available")
        return
    
    # Sort gears
    gears = sorted(gear_counts.keys())
    counts = [gear_counts[g] for g in gears]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(gears)))
    wedges, texts, autotexts = ax.pie(counts, labels=[f'Gear {g}' for g in gears], 
                                       autopct='%1.1f%%', colors=colors,
                                       explode=[0.02]*len(gears))
    
    ax.set_title(f'Gear Usage Distribution - {driver.name_acronym}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/08_gear_distribution_{driver.name_acronym}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/08_gear_distribution_{driver.name_acronym}.png")


# =============================================================================
# LOCATION (TRACK POSITION) ANALYSIS
# =============================================================================

def plot_racing_line(client, session_key, driver_map, driver_num):
    """
    Plot the racing line (X/Y track position) for a driver.
    """
    print(f"\n📊 Generating racing line for driver {driver_num}...")
    
    driver = driver_map.get(driver_num)
    if not driver:
        print(f"  Driver {driver_num} not found")
        return
    
    location_data = client.location.list(
        session_key=session_key,
        driver_number=driver_num
    )
    
    if not location_data:
        print("  No location data available")
        return
    
    # Sample data (every 50th point)
    sampled = location_data[::50]
    
    x = [d.x for d in sampled if d.x is not None]
    y = [d.y for d in sampled if d.y is not None]
    
    if not x or not y:
        print("  No valid coordinates")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create color gradient based on position in lap
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    
    ax.scatter(x, y, c=range(len(x)), cmap='viridis', s=1, alpha=0.6)
    ax.plot(x, y, color='#333333', linewidth=0.5, alpha=0.3)
    
    # Mark start/finish
    ax.scatter(x[0], y[0], c='green', s=100, marker='o', label='Start', zorder=5)
    
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(f'Racing Line - {driver.name_acronym} ({driver.team_name})', 
                fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/09_racing_line_{driver.name_acronym}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/09_racing_line_{driver.name_acronym}.png")


def plot_racing_lines_comparison(client, session_key, driver_map, driver_nums):
    """
    Compare racing lines of multiple drivers.
    """
    print("\n📊 Generating racing lines comparison...")
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    colors = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261']
    
    for idx, driver_num in enumerate(driver_nums):
        driver = driver_map.get(driver_num)
        if not driver:
            continue
        
        location_data = client.location.list(
            session_key=session_key,
            driver_number=driver_num
        )
        
        if not location_data:
            continue
        
        # Sample a single lap worth of data
        sampled = location_data[::100][:200]
        
        x = [d.x for d in sampled if d.x is not None]
        y = [d.y for d in sampled if d.y is not None]
        
        if x and y:
            ax.plot(x, y, color=colors[idx % len(colors)], 
                   linewidth=1.5, alpha=0.7, label=driver.name_acronym)
    
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('Racing Lines Comparison', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/10_racing_lines_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/10_racing_lines_comparison.png")


def plot_track_with_speed(client, session_key, driver_map, driver_num):
    """
    Plot track map colored by speed (requires both location and car_data).
    """
    print(f"\n📊 Generating track speed map for driver {driver_num}...")
    
    driver = driver_map.get(driver_num)
    if not driver:
        print(f"  Driver {driver_num} not found")
        return
    
    # Get location data
    location_data = client.location.list(
        session_key=session_key,
        driver_number=driver_num
    )
    
    # Get car data for speed
    car_data = client.car_data.list(
        session_key=session_key,
        driver_number=driver_num
    )
    
    if not location_data or not car_data:
        print("  Insufficient data")
        return
    
    # Match location and speed data (sample both)
    sample_rate = 100
    loc_sampled = location_data[::sample_rate]
    car_sampled = car_data[::sample_rate]
    
    # Take minimum length
    min_len = min(len(loc_sampled), len(car_sampled))
    
    x = [loc_sampled[i].x for i in range(min_len) if loc_sampled[i].x is not None]
    y = [loc_sampled[i].y for i in range(min_len) if loc_sampled[i].y is not None]
    speed = [car_sampled[i].speed if car_sampled[i].speed else 0 for i in range(min_len)]
    
    # Ensure all arrays same length
    min_len = min(len(x), len(y), len(speed))
    x, y, speed = x[:min_len], y[:min_len], speed[:min_len]
    
    if not x:
        print("  No valid data points")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(x, y, c=speed, cmap='RdYlGn', s=5, alpha=0.8)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Speed (km/h)', fontsize=12)
    
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(f'Track Speed Map - {driver.name_acronym}', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/11_track_speed_{driver.name_acronym}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/11_track_speed_{driver.name_acronym}.png")


# =============================================================================
# SUMMARY VISUALIZATIONS
# =============================================================================

def plot_race_summary(client, session_key, driver_map, race_info):
    """
    Create a summary dashboard of the race.
    """
    print("\n📊 Generating race summary dashboard...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Title
    fig.suptitle(f"{race_info.session_name} - {race_info.location}, {race_info.country_name}", 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Fastest Lap Comparison (bar chart)
    ax1 = fig.add_subplot(gs[0, 0])
    
    fastest_laps = []
    for driver_num, driver in driver_map.items():
        fastest = client.laps.get_fastest_lap(
            session_key=session_key,
            driver_number=driver_num
        )
        if fastest and fastest.lap_duration:
            fastest_laps.append((driver.name_acronym, fastest.lap_duration))
    
    fastest_laps.sort(key=lambda x: x[1])
    top_10 = fastest_laps[:10]
    
    if top_10:
        drivers_names = [d[0] for d in top_10]
        times = [d[1] for d in top_10]
        min_time = min(times)
        deltas = [t - min_time for t in times]
        
        colors = ['gold' if i == 0 else 'steelblue' for i in range(len(deltas))]
        ax1.barh(drivers_names[::-1], deltas[::-1], color=colors[::-1])
        ax1.set_xlabel('Gap to Fastest (s)')
        ax1.set_title('Fastest Lap Comparison', fontweight='bold')
    
    # 2. Pit Stop Count (bar chart)
    ax2 = fig.add_subplot(gs[0, 1])
    
    pit_counts = []
    for driver_num, driver in list(driver_map.items())[:10]:
        count = client.pit.count(
            session_key=session_key,
            driver_number=driver_num
        )
        pit_counts.append((driver.name_acronym, count))
    
    pit_counts.sort(key=lambda x: x[1], reverse=True)
    
    if pit_counts:
        ax2.bar([p[0] for p in pit_counts], [p[1] for p in pit_counts], color='coral')
        ax2.set_ylabel('Pit Stops')
        ax2.set_title('Pit Stop Count', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. Weather Summary (text box)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    weather = client.weather.list(session_key=session_key)
    if weather:
        first_w = weather[0]
        last_w = weather[-1]
        
        weather_text = f"""
WEATHER CONDITIONS

Start of Race:
  Air Temp: {first_w.air_temperature}°C
  Track Temp: {first_w.track_temperature}°C
  Humidity: {first_w.humidity}%
  
End of Race:
  Air Temp: {last_w.air_temperature}°C
  Track Temp: {last_w.track_temperature}°C
  Humidity: {last_w.humidity}%
  
Rain: {'Yes' if any(w.rainfall for w in weather) else 'No'}
"""
        ax3.text(0.1, 0.5, weather_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax3.set_title('Weather Summary', fontweight='bold')
    
    # 4. Average Lap Time Distribution
    ax4 = fig.add_subplot(gs[1, :2])
    
    avg_times = []
    for driver_num, driver in driver_map.items():
        laps = client.laps.list(
            session_key=session_key,
            driver_number=driver_num
        )
        valid_laps = [l.lap_duration for l in laps 
                     if l.lap_duration and l.lap_duration < 150]
        if valid_laps:
            avg_times.append((driver.name_acronym, np.mean(valid_laps), driver.team_name))
    
    avg_times.sort(key=lambda x: x[1])
    
    if avg_times:
        teams = list(set([a[2] for a in avg_times]))
        team_colors = {t: plt.cm.tab20(i/len(teams)) for i, t in enumerate(teams)}
        
        colors = [team_colors.get(a[2], 'gray') for a in avg_times]
        ax4.barh([a[0] for a in avg_times[::-1]], 
                [a[1] for a in avg_times[::-1]], 
                color=colors[::-1])
        ax4.set_xlabel('Average Lap Time (s)')
        ax4.set_title('Average Race Pace Comparison', fontweight='bold')
    
    # 5. Race Control Events Summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    race_control = client.race_control.list(session_key=session_key)
    
    event_counts = {}
    for rc in race_control:
        cat = rc.category if rc.category else 'Other'
        event_counts[cat] = event_counts.get(cat, 0) + 1
    
    if event_counts:
        events_text = "RACE CONTROL EVENTS\n\n"
        for cat, count in sorted(event_counts.items(), key=lambda x: -x[1]):
            events_text += f"  {cat}: {count}\n"
        
        ax5.text(0.1, 0.5, events_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        ax5.set_title('Race Events', fontweight='bold')
    
    plt.savefig(f"{OUTPUT_DIR}/00_race_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/00_race_summary.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("🏎️  F1 RACE ANALYSIS & VISUALIZATION")
    print("=" * 70)
    
    with OpenF1Client() as client:
        # Get latest race (or specify a session_key with known data)
        # Session 9161 = Singapore 2023 GP (known to have good data)
        # Session 9158 = Monza 2023 GP
        # Set to None to auto-find latest with data
        race = get_latest_race_session(client, session_key=None)
        session_key = race.session_key
        
        # Get drivers
        drivers, driver_map = get_drivers(client, session_key)
        print(f"Found {len(drivers)} drivers")
        
        if len(drivers) == 0:
            print("\n⚠️  No driver data available for this session.")
            print("Trying with a known session (Singapore 2023 GP)...")
            race = get_latest_race_session(client, session_key=9161)
            session_key = race.session_key
            drivers, driver_map = get_drivers(client, session_key)
            print(f"Found {len(drivers)} drivers")
        
        if len(drivers) == 0:
            print("\n❌ No driver data available. Please check API connectivity.")
            return
        
        # Find top 3 drivers for detailed analysis
        all_fastest = []
        for driver_num, driver in driver_map.items():
            fastest = client.laps.get_fastest_lap(
                session_key=session_key,
                driver_number=driver_num
            )
            if fastest and fastest.lap_duration:
                all_fastest.append((driver_num, fastest.lap_duration))
        
        all_fastest.sort(key=lambda x: x[1])
        top_3_drivers = [d[0] for d in all_fastest[:3]]
        
        if top_3_drivers:
            print(f"\nTop 3 drivers by fastest lap: {[driver_map[d].name_acronym for d in top_3_drivers]}")
        else:
            # Fallback: just use first 3 drivers
            top_3_drivers = list(driver_map.keys())[:3]
            print(f"\nUsing first 3 drivers: {[driver_map[d].name_acronym for d in top_3_drivers]}")
        
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        # Generate all visualizations
        
        # 1. Race Summary Dashboard
        plot_race_summary(client, session_key, driver_map, race)
        
        # 2. Lap Time Analysis
        plot_lap_times_comparison(client, session_key, driver_map)
        plot_sector_times_heatmap(client, session_key, driver_map)
        plot_lap_consistency(client, session_key, driver_map)
        plot_pace_degradation(client, session_key, driver_map, top_3_drivers)
        
        # 3. Telemetry Analysis (for top driver)
        if top_3_drivers:
            top_driver = top_3_drivers[0]
            plot_speed_trace(client, session_key, driver_map, top_driver)
            plot_throttle_brake_analysis(client, session_key, driver_map, top_driver)
            plot_gear_distribution(client, session_key, driver_map, top_driver)
        
        # 4. DRS Usage
        plot_drs_usage(client, session_key, driver_map)
        
        # 5. Racing Lines
        if top_3_drivers:
            plot_racing_line(client, session_key, driver_map, top_3_drivers[0])
            plot_racing_lines_comparison(client, session_key, driver_map, top_3_drivers)
            plot_track_with_speed(client, session_key, driver_map, top_3_drivers[0])
        
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
