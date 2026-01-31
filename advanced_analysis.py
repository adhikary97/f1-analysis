"""
Advanced F1 Race Analysis - Interesting Correlations using OpenF1 API
Creates visualizations that reveal hidden patterns and relationships in race data.
"""

from datetime import datetime, timezone
from collections import defaultdict
import os

from openf1_client import OpenF1Client
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np


# Team colors
TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "McLaren": "#FF8000",
    "Ferrari": "#E8002D",
    "Mercedes": "#27F4D2",
    "Aston Martin": "#229971",
    "Alpine": "#0093CC",
    "Williams": "#64C4FF",
    "Racing Bulls": "#6692FF",
    "Kick Sauber": "#52E252",
    "Haas F1 Team": "#B6BABD",
}


def get_driver_color(team_name):
    """Get color based on team."""
    return TEAM_COLORS.get(team_name, "#888888")


def find_latest_completed_race(client):
    """Find the most recent race with data."""
    sessions = client.sessions.list()
    if not sessions:
        return None
    
    race_sessions = [s for s in sessions if s.session_type == "Race"]
    now = datetime.now(timezone.utc)
    
    for race in reversed(race_sessions):
        if race.date_start:
            try:
                race_date = datetime.fromisoformat(str(race.date_start).replace('Z', '+00:00'))
                if race_date > now:
                    continue
            except (ValueError, TypeError):
                pass
        
        laps = client.laps.list(session_key=race.session_key)
        if laps:
            return race
    
    return None


def create_battle_intensity_plot(client, session_key, drivers, driver_map):
    """Create a heatmap showing battle intensity between drivers throughout the race."""
    print("  Creating battle intensity analysis...")
    
    # Get interval data
    intervals = client.intervals.list(session_key=session_key)
    
    if not intervals:
        print("    No interval data available")
        return None
    
    # Group intervals by lap (approximate) and track close battles
    driver_numbers = [d.driver_number for d in drivers]
    battle_matrix = defaultdict(lambda: defaultdict(int))
    
    # Track when drivers are within 1 second
    close_battles = defaultdict(list)
    
    for interval in intervals:
        driver_num = interval.driver_number
        gap = getattr(interval, 'interval', None)
        
        if gap and isinstance(gap, (int, float)) and abs(gap) < 1.0:
            # This driver is in a close battle
            close_battles[driver_num].append(gap)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.patch.set_facecolor('#0d1117')
    
    # ===== Plot 1: Close Battle Duration (seconds within DRS range) =====
    ax1 = axes[0, 0]
    ax1.set_facecolor('#161b22')
    
    battle_durations = []
    driver_names = []
    colors = []
    
    for driver in drivers:
        battles = close_battles.get(driver.driver_number, [])
        # Each interval reading is ~4 seconds, count readings within 1 second
        duration = len(battles) * 4  # Approximate seconds in battle
        battle_durations.append(duration)
        driver_names.append(driver.name_acronym)
        colors.append(get_driver_color(driver.team_name))
    
    # Sort by battle duration
    sorted_data = sorted(zip(battle_durations, driver_names, colors), reverse=True)
    battle_durations, driver_names, colors = zip(*sorted_data) if sorted_data else ([], [], [])
    
    bars = ax1.barh(range(len(driver_names)), battle_durations, color=colors, 
                   edgecolor='white', linewidth=0.5, alpha=0.85)
    
    ax1.set_yticks(range(len(driver_names)))
    ax1.set_yticklabels(driver_names, fontsize=10, color='white')
    ax1.set_xlabel('Time in Close Battle (seconds)', fontsize=11, color='white')
    ax1.set_title('Time Spent Within 1 Second of Another Car', fontsize=13, color='white', fontweight='bold')
    ax1.invert_yaxis()
    
    for spine in ax1.spines.values():
        spine.set_color('#30363d')
    ax1.tick_params(colors='white')
    ax1.grid(True, axis='x', alpha=0.2, color='white')
    
    # ===== Plot 2: Overtake Opportunities (DRS within range) =====
    ax2 = axes[0, 1]
    ax2.set_facecolor('#161b22')
    
    # Count times each driver was within DRS range (< 1 second behind)
    drs_opportunities = defaultdict(int)
    for driver_num, gaps in close_battles.items():
        # Negative gap means behind the car ahead
        behind_count = sum(1 for g in gaps if g > 0 and g < 1.0)
        drs_opportunities[driver_num] = behind_count
    
    driver_drs = [(driver_map.get(num), count) for num, count in drs_opportunities.items() if driver_map.get(num)]
    driver_drs.sort(key=lambda x: x[1], reverse=True)
    
    if driver_drs:
        names = [d[0].name_acronym for d in driver_drs[:15]]
        counts = [d[1] for d in driver_drs[:15]]
        colors = [get_driver_color(d[0].team_name) for d in driver_drs[:15]]
        
        ax2.barh(range(len(names)), counts, color=colors, edgecolor='white', linewidth=0.5, alpha=0.85)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=10, color='white')
        ax2.invert_yaxis()
    
    ax2.set_xlabel('DRS Activation Opportunities', fontsize=11, color='white')
    ax2.set_title('Potential DRS Opportunities', fontsize=13, color='white', fontweight='bold')
    
    for spine in ax2.spines.values():
        spine.set_color('#30363d')
    ax2.tick_params(colors='white')
    ax2.grid(True, axis='x', alpha=0.2, color='white')
    
    # ===== Plot 3: Average Gap to Car Ahead =====
    ax3 = axes[1, 0]
    ax3.set_facecolor('#161b22')
    
    avg_gaps = defaultdict(list)
    for interval in intervals:
        gap = getattr(interval, 'gap_to_leader', None)
        if gap and isinstance(gap, (int, float)):
            avg_gaps[interval.driver_number].append(gap)
    
    driver_avg_gaps = []
    for driver in drivers:
        gaps = avg_gaps.get(driver.driver_number, [])
        if gaps:
            driver_avg_gaps.append((driver, np.mean(gaps), np.std(gaps)))
    
    driver_avg_gaps.sort(key=lambda x: x[1])
    
    if driver_avg_gaps:
        names = [d[0].name_acronym for d in driver_avg_gaps]
        means = [d[1] for d in driver_avg_gaps]
        stds = [d[2] for d in driver_avg_gaps]
        colors = [get_driver_color(d[0].team_name) for d in driver_avg_gaps]
        
        ax3.barh(range(len(names)), means, xerr=stds, color=colors, 
                edgecolor='white', linewidth=0.5, alpha=0.85, capsize=3)
        ax3.set_yticks(range(len(names)))
        ax3.set_yticklabels(names, fontsize=10, color='white')
        ax3.invert_yaxis()
    
    ax3.set_xlabel('Average Gap to Leader (seconds)', fontsize=11, color='white')
    ax3.set_title('Average Race Position (Gap to Leader)', fontsize=13, color='white', fontweight='bold')
    
    for spine in ax3.spines.values():
        spine.set_color('#30363d')
    ax3.tick_params(colors='white')
    ax3.grid(True, axis='x', alpha=0.2, color='white')
    
    # ===== Plot 4: Position Volatility =====
    ax4 = axes[1, 1]
    ax4.set_facecolor('#161b22')
    
    # Get position data
    positions = client.position.list(session_key=session_key)
    
    driver_positions = defaultdict(list)
    for pos in positions:
        driver_positions[pos.driver_number].append(pos.position)
    
    volatility_data = []
    for driver in drivers:
        pos_list = driver_positions.get(driver.driver_number, [])
        if len(pos_list) > 1:
            # Calculate position changes
            changes = sum(abs(pos_list[i] - pos_list[i-1]) for i in range(1, len(pos_list)))
            volatility_data.append((driver, changes))
    
    volatility_data.sort(key=lambda x: x[1], reverse=True)
    
    if volatility_data:
        names = [d[0].name_acronym for d in volatility_data[:15]]
        changes = [d[1] for d in volatility_data[:15]]
        colors = [get_driver_color(d[0].team_name) for d in volatility_data[:15]]
        
        ax4.barh(range(len(names)), changes, color=colors, edgecolor='white', linewidth=0.5, alpha=0.85)
        ax4.set_yticks(range(len(names)))
        ax4.set_yticklabels(names, fontsize=10, color='white')
        ax4.invert_yaxis()
    
    ax4.set_xlabel('Total Position Changes', fontsize=11, color='white')
    ax4.set_title('Race Volatility (Position Changes)', fontsize=13, color='white', fontweight='bold')
    
    for spine in ax4.spines.values():
        spine.set_color('#30363d')
    ax4.tick_params(colors='white')
    ax4.grid(True, axis='x', alpha=0.2, color='white')
    
    plt.suptitle('Battle Analysis', fontsize=16, color='white', fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def create_tyre_performance_analysis(client, session_key, drivers, driver_map):
    """Analyze tyre compound performance and degradation."""
    print("  Creating tyre performance analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.patch.set_facecolor('#0d1117')
    
    compound_colors = {
        "SOFT": "#FF3333",
        "MEDIUM": "#FFD700",
        "HARD": "#EEEEEE",
        "INTERMEDIATE": "#43B02A",
        "WET": "#0067AD",
    }
    
    # Get all stints
    all_stints = []
    for driver in drivers:
        stints = client.stints.list(session_key=session_key, driver_number=driver.driver_number)
        for stint in stints:
            if stint.compound:
                all_stints.append({
                    "driver": driver,
                    "compound": stint.compound.upper(),
                    "lap_start": stint.lap_start,
                    "lap_end": stint.lap_end,
                    "stint_length": (stint.lap_end or 0) - (stint.lap_start or 0),
                    "tyre_age": getattr(stint, "tyre_age_at_start", 0) or 0,
                })
    
    # ===== Plot 1: Stint Length by Compound =====
    ax1 = axes[0, 0]
    ax1.set_facecolor('#161b22')
    
    compound_lengths = defaultdict(list)
    for stint in all_stints:
        if stint["stint_length"] > 0:
            compound_lengths[stint["compound"]].append(stint["stint_length"])
    
    compounds = list(compound_lengths.keys())
    if compounds:
        positions = range(len(compounds))
        
        # Box plot data
        box_data = [compound_lengths[c] for c in compounds]
        colors = [compound_colors.get(c, "#888888") for c in compounds]
        
        bp = ax1.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for whisker in bp['whiskers']:
            whisker.set_color('white')
        for cap in bp['caps']:
            cap.set_color('white')
        for median in bp['medians']:
            median.set_color('#0d1117')
            median.set_linewidth(2)
        for flier in bp['fliers']:
            flier.set_markerfacecolor('white')
            flier.set_markeredgecolor('white')
        
        ax1.set_xticks(positions)
        ax1.set_xticklabels(compounds, fontsize=11, color='white')
    
    ax1.set_ylabel('Stint Length (laps)', fontsize=11, color='white')
    ax1.set_title('Stint Duration by Tyre Compound', fontsize=13, color='white', fontweight='bold')
    
    for spine in ax1.spines.values():
        spine.set_color('#30363d')
    ax1.tick_params(colors='white')
    ax1.grid(True, axis='y', alpha=0.2, color='white')
    
    # ===== Plot 2: Compound Usage Distribution =====
    ax2 = axes[0, 1]
    ax2.set_facecolor('#161b22')
    
    compound_counts = defaultdict(int)
    for stint in all_stints:
        compound_counts[stint["compound"]] += 1
    
    if compound_counts:
        labels = list(compound_counts.keys())
        sizes = list(compound_counts.values())
        colors = [compound_colors.get(c, "#888888") for c in labels]
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                           startangle=90, textprops={'color': 'white', 'fontsize': 11})
        for autotext in autotexts:
            autotext.set_color('#0d1117')
            autotext.set_fontweight('bold')
    
    ax2.set_title('Tyre Compound Distribution', fontsize=13, color='white', fontweight='bold')
    
    # ===== Plot 3: Strategy Patterns =====
    ax3 = axes[1, 0]
    ax3.set_facecolor('#161b22')
    
    # Group drivers by strategy
    driver_strategies = defaultdict(list)
    for driver in drivers:
        stints = client.stints.list(session_key=session_key, driver_number=driver.driver_number)
        compounds = [s.compound.upper() for s in stints if s.compound]
        if compounds:
            strategy = " → ".join(compounds)
            driver_strategies[strategy].append(driver)
    
    # Plot strategy distribution
    strategies = list(driver_strategies.keys())
    counts = [len(driver_strategies[s]) for s in strategies]
    
    # Sort by count
    sorted_data = sorted(zip(counts, strategies), reverse=True)
    if sorted_data:
        counts, strategies = zip(*sorted_data)
        
        # Truncate long strategy names
        display_strategies = [s if len(s) < 25 else s[:22] + "..." for s in strategies]
        
        bars = ax3.barh(range(len(display_strategies)), counts, color='#3498DB', 
                       edgecolor='white', linewidth=0.5, alpha=0.85)
        ax3.set_yticks(range(len(display_strategies)))
        ax3.set_yticklabels(display_strategies, fontsize=9, color='white')
        ax3.invert_yaxis()
        
        # Add driver names
        for i, (strategy, count) in enumerate(zip(strategies, counts)):
            drivers_using = driver_strategies[strategy]
            names = ", ".join([d.name_acronym for d in drivers_using])
            ax3.text(count + 0.1, i, names, va='center', fontsize=8, color='#8B949E')
    
    ax3.set_xlabel('Number of Drivers', fontsize=11, color='white')
    ax3.set_title('Strategy Choices', fontsize=13, color='white', fontweight='bold')
    
    for spine in ax3.spines.values():
        spine.set_color('#30363d')
    ax3.tick_params(colors='white')
    ax3.grid(True, axis='x', alpha=0.2, color='white')
    
    # ===== Plot 4: Pit Stop Windows =====
    ax4 = axes[1, 1]
    ax4.set_facecolor('#161b22')
    
    # Get pit stops
    pit_laps = defaultdict(int)
    for driver in drivers:
        pits = client.pit.list(session_key=session_key, driver_number=driver.driver_number)
        for pit in pits:
            lap = getattr(pit, "lap_number", None)
            if lap:
                pit_laps[lap] += 1
    
    if pit_laps:
        laps = sorted(pit_laps.keys())
        counts = [pit_laps[l] for l in laps]
        
        ax4.bar(laps, counts, color='#E74C3C', edgecolor='white', linewidth=0.5, alpha=0.85)
        ax4.set_xlabel('Lap Number', fontsize=11, color='white')
        ax4.set_ylabel('Number of Pit Stops', fontsize=11, color='white')
    
    ax4.set_title('Pit Stop Windows', fontsize=13, color='white', fontweight='bold')
    
    for spine in ax4.spines.values():
        spine.set_color('#30363d')
    ax4.tick_params(colors='white')
    ax4.grid(True, axis='y', alpha=0.2, color='white')
    
    plt.suptitle('Tyre Strategy Analysis', fontsize=16, color='white', fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def create_pace_weather_correlation(client, session_key, drivers, driver_map):
    """Analyze correlation between weather conditions and lap times."""
    print("  Creating pace-weather correlation analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.patch.set_facecolor('#0d1117')
    
    # Get weather data
    weather = client.weather.list(session_key=session_key)
    
    if not weather:
        print("    No weather data available")
        return None
    
    # ===== Plot 1: Temperature Evolution =====
    ax1 = axes[0, 0]
    ax1.set_facecolor('#161b22')
    
    times = list(range(len(weather)))
    air_temps = [w.air_temperature for w in weather if w.air_temperature]
    track_temps = [w.track_temperature for w in weather if w.track_temperature]
    
    if air_temps and track_temps:
        ax1.plot(range(len(air_temps)), air_temps, color='#3498DB', linewidth=2, label='Air Temp', marker='o', markersize=3)
        ax1.plot(range(len(track_temps)), track_temps, color='#E74C3C', linewidth=2, label='Track Temp', marker='o', markersize=3)
        ax1.fill_between(range(len(track_temps)), air_temps[:len(track_temps)], track_temps, alpha=0.2, color='#E74C3C')
        
        ax1.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
    
    ax1.set_xlabel('Time (readings)', fontsize=11, color='white')
    ax1.set_ylabel('Temperature (°C)', fontsize=11, color='white')
    ax1.set_title('Temperature Evolution During Race', fontsize=13, color='white', fontweight='bold')
    
    for spine in ax1.spines.values():
        spine.set_color('#30363d')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.2, color='white')
    
    # ===== Plot 2: Humidity & Wind =====
    ax2 = axes[0, 1]
    ax2.set_facecolor('#161b22')
    
    humidity = [w.humidity for w in weather if w.humidity]
    wind_speed = [getattr(w, 'wind_speed', 0) or 0 for w in weather]
    
    ax2_twin = ax2.twinx()
    
    if humidity:
        line1, = ax2.plot(range(len(humidity)), humidity, color='#2ECC71', linewidth=2, label='Humidity %')
    if wind_speed and any(wind_speed):
        line2, = ax2_twin.plot(range(len(wind_speed)), wind_speed, color='#9B59B6', linewidth=2, label='Wind Speed')
        ax2_twin.set_ylabel('Wind Speed (m/s)', fontsize=11, color='#9B59B6')
        ax2_twin.tick_params(axis='y', colors='#9B59B6')
    
    ax2.set_xlabel('Time (readings)', fontsize=11, color='white')
    ax2.set_ylabel('Humidity (%)', fontsize=11, color='#2ECC71')
    ax2.set_title('Humidity & Wind Conditions', fontsize=13, color='white', fontweight='bold')
    ax2.tick_params(axis='y', colors='#2ECC71')
    
    for spine in ax2.spines.values():
        spine.set_color('#30363d')
    ax2.tick_params(axis='x', colors='white')
    ax2.grid(True, alpha=0.2, color='white')
    
    # ===== Plot 3: Track Temp vs Fastest Laps =====
    ax3 = axes[1, 0]
    ax3.set_facecolor('#161b22')
    
    # Get lap times for top drivers
    lap_times_by_driver = {}
    for driver in drivers[:6]:  # Top 6 drivers
        laps = client.laps.list(session_key=session_key, driver_number=driver.driver_number)
        lap_times_by_driver[driver.name_acronym] = [(l.lap_number, l.lap_duration) 
                                                     for l in laps if l.lap_duration and l.lap_duration < 120]
    
    for driver_name, laps in lap_times_by_driver.items():
        if laps:
            lap_nums, times = zip(*laps)
            driver = next((d for d in drivers if d.name_acronym == driver_name), None)
            color = get_driver_color(driver.team_name) if driver else '#888888'
            ax3.scatter(lap_nums, times, s=20, alpha=0.6, color=color, label=driver_name)
    
    ax3.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='white', fontsize=9)
    ax3.set_xlabel('Lap Number', fontsize=11, color='white')
    ax3.set_ylabel('Lap Time (seconds)', fontsize=11, color='white')
    ax3.set_title('Lap Time Progression (Top 6)', fontsize=13, color='white', fontweight='bold')
    
    for spine in ax3.spines.values():
        spine.set_color('#30363d')
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.2, color='white')
    
    # ===== Plot 4: Conditions Summary =====
    ax4 = axes[1, 1]
    ax4.set_facecolor('#161b22')
    ax4.axis('off')
    
    # Calculate summary stats
    if weather:
        first_w = weather[0]
        last_w = weather[-1]
        
        summary_text = f"""
RACE WEATHER SUMMARY
{'='*40}

START CONDITIONS:
  Air Temperature:    {first_w.air_temperature}°C
  Track Temperature:  {first_w.track_temperature}°C
  Humidity:           {first_w.humidity}%
  Pressure:           {getattr(first_w, 'pressure', 'N/A')} mbar

END CONDITIONS:
  Air Temperature:    {last_w.air_temperature}°C
  Track Temperature:  {last_w.track_temperature}°C
  Humidity:           {last_w.humidity}%

CHANGES:
  Air Temp Change:    {(last_w.air_temperature or 0) - (first_w.air_temperature or 0):+.1f}°C
  Track Temp Change:  {(last_w.track_temperature or 0) - (first_w.track_temperature or 0):+.1f}°C
  Humidity Change:    {(last_w.humidity or 0) - (first_w.humidity or 0):+.1f}%

RAIN: {'Yes' if any(w.rainfall for w in weather) else 'No'}
"""
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace', color='white',
                bbox=dict(boxstyle='round', facecolor='#21262d', edgecolor='#30363d'))
    
    plt.suptitle('Weather & Performance Analysis', fontsize=16, color='white', fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def create_driver_performance_radar(client, session_key, drivers, driver_map, results):
    """Create a radar chart comparing driver performance across multiple metrics."""
    print("  Creating driver performance radar...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('#0d1117')
    
    # Get top 5 finishers
    top_drivers = []
    if results:
        sorted_results = sorted([r for r in results if r.position], key=lambda x: x.position)
        for r in sorted_results[:5]:
            driver = driver_map.get(r.driver_number)
            if driver:
                top_drivers.append(driver)
    
    if not top_drivers:
        top_drivers = drivers[:5]
    
    # Calculate metrics for each driver
    metrics = ['Fastest Lap', 'Consistency', 'Overtakes', 'Track Position', 'Pit Efficiency']
    driver_scores = {}
    
    for driver in top_drivers:
        scores = []
        
        # 1. Fastest lap ranking
        fastest = client.laps.get_fastest_lap(session_key=session_key, driver_number=driver.driver_number)
        fastest_time = fastest.lap_duration if fastest else 999
        scores.append(fastest_time)
        
        # 2. Consistency (std of lap times)
        laps = client.laps.list(session_key=session_key, driver_number=driver.driver_number)
        lap_times = [l.lap_duration for l in laps if l.lap_duration and l.lap_duration < 120]
        consistency = np.std(lap_times) if lap_times else 10
        scores.append(consistency)
        
        # 3. Position changes (gains)
        positions = client.position.list(session_key=session_key, driver_number=driver.driver_number)
        if positions:
            start_pos = positions[0].position
            end_pos = positions[-1].position
            gains = start_pos - end_pos  # Positive = gained positions
            scores.append(gains)
        else:
            scores.append(0)
        
        # 4. Average track position
        intervals = client.intervals.list(session_key=session_key, driver_number=driver.driver_number)
        gaps = [i.gap_to_leader for i in intervals if i.gap_to_leader and isinstance(i.gap_to_leader, (int, float))]
        avg_gap = np.mean(gaps) if gaps else 60
        scores.append(avg_gap)
        
        # 5. Pit stop count
        pits = client.pit.list(session_key=session_key, driver_number=driver.driver_number)
        scores.append(len(pits))
        
        driver_scores[driver.name_acronym] = scores
    
    # Normalize scores (0-1 scale, higher is better)
    normalized_scores = {}
    for driver_name, scores in driver_scores.items():
        normalized = []
        # Fastest lap (lower is better)
        all_fastest = [s[0] for s in driver_scores.values()]
        normalized.append(1 - (scores[0] - min(all_fastest)) / (max(all_fastest) - min(all_fastest) + 0.001))
        
        # Consistency (lower std is better)
        all_consistency = [s[1] for s in driver_scores.values()]
        normalized.append(1 - (scores[1] - min(all_consistency)) / (max(all_consistency) - min(all_consistency) + 0.001))
        
        # Overtakes (higher is better)
        all_gains = [s[2] for s in driver_scores.values()]
        normalized.append((scores[2] - min(all_gains)) / (max(all_gains) - min(all_gains) + 0.001))
        
        # Track position (lower gap is better)
        all_gaps = [s[3] for s in driver_scores.values()]
        normalized.append(1 - (scores[3] - min(all_gaps)) / (max(all_gaps) - min(all_gaps) + 0.001))
        
        # Pit efficiency (fewer is better for most races)
        all_pits = [s[4] for s in driver_scores.values()]
        normalized.append(1 - (scores[4] - min(all_pits)) / (max(all_pits) - min(all_pits) + 0.001))
        
        normalized_scores[driver_name] = normalized
    
    # ===== Plot 1: Radar Chart =====
    ax1 = axes[0]
    ax1.set_facecolor('#161b22')
    
    # Number of variables
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    ax1 = fig.add_subplot(121, polar=True)
    ax1.set_facecolor('#161b22')
    
    for driver_name, scores in normalized_scores.items():
        driver = next((d for d in top_drivers if d.name_acronym == driver_name), None)
        color = get_driver_color(driver.team_name) if driver else '#888888'
        
        values = scores + scores[:1]  # Complete the loop
        ax1.plot(angles, values, linewidth=2, label=driver_name, color=color)
        ax1.fill(angles, values, alpha=0.15, color=color)
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics, fontsize=10, color='white')
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), 
              facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
    ax1.set_title('Performance Radar (Top 5)', fontsize=13, color='white', fontweight='bold', pad=20)
    
    # Style polar plot
    ax1.tick_params(colors='white')
    ax1.spines['polar'].set_color('#30363d')
    ax1.set_facecolor('#161b22')
    
    # ===== Plot 2: Performance Bar Comparison =====
    ax2 = axes[1]
    ax2.set_facecolor('#161b22')
    
    x = np.arange(len(metrics))
    width = 0.15
    
    for i, (driver_name, scores) in enumerate(normalized_scores.items()):
        driver = next((d for d in top_drivers if d.name_acronym == driver_name), None)
        color = get_driver_color(driver.team_name) if driver else '#888888'
        offset = (i - len(normalized_scores)/2 + 0.5) * width
        ax2.bar(x + offset, scores, width, label=driver_name, color=color, alpha=0.85)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=10, color='white', rotation=15)
    ax2.set_ylabel('Normalized Score', fontsize=11, color='white')
    ax2.set_title('Performance Comparison', fontsize=13, color='white', fontweight='bold')
    ax2.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
    ax2.set_ylim(0, 1.1)
    
    for spine in ax2.spines.values():
        spine.set_color('#30363d')
    ax2.tick_params(colors='white')
    ax2.grid(True, axis='y', alpha=0.2, color='white')
    
    plt.suptitle('Driver Performance Analysis', fontsize=16, color='white', fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def main():
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    print("=" * 60)
    print("ADVANCED F1 RACE ANALYSIS")
    print("=" * 60)
    
    with OpenF1Client() as client:
        # Find latest race
        print("\nFinding latest completed race...")
        latest_race = find_latest_completed_race(client)
        
        if not latest_race:
            print("No completed race found.")
            return
        
        session_key = latest_race.session_key
        print(f"\nRace: {latest_race.location}, {latest_race.country_name}")
        print(f"Session Key: {session_key}")
        
        # Get drivers and results
        print("\nFetching race data...")
        drivers = client.drivers.list(session_key=session_key)
        driver_map = {d.driver_number: d for d in drivers}
        results = client.session_result.list(session_key=session_key)
        
        print(f"Found {len(drivers)} drivers")
        
        # Generate plots
        print("\nGenerating advanced analysis plots...")
        
        # 1. Battle Intensity
        fig1 = create_battle_intensity_plot(client, session_key, drivers, driver_map)
        if fig1:
            fig1.savefig("plots/12_battle_analysis.png", dpi=150, 
                        facecolor=fig1.get_facecolor(), bbox_inches='tight')
            print("  Saved: plots/12_battle_analysis.png")
            plt.close(fig1)
        
        # 2. Tyre Performance
        fig2 = create_tyre_performance_analysis(client, session_key, drivers, driver_map)
        if fig2:
            fig2.savefig("plots/13_tyre_analysis.png", dpi=150,
                        facecolor=fig2.get_facecolor(), bbox_inches='tight')
            print("  Saved: plots/13_tyre_analysis.png")
            plt.close(fig2)
        
        # 3. Weather Correlation
        fig3 = create_pace_weather_correlation(client, session_key, drivers, driver_map)
        if fig3:
            fig3.savefig("plots/14_weather_analysis.png", dpi=150,
                        facecolor=fig3.get_facecolor(), bbox_inches='tight')
            print("  Saved: plots/14_weather_analysis.png")
            plt.close(fig3)
        
        # 4. Driver Performance Radar
        fig4 = create_driver_performance_radar(client, session_key, drivers, driver_map, results)
        if fig4:
            fig4.savefig("plots/15_performance_radar.png", dpi=150,
                        facecolor=fig4.get_facecolor(), bbox_inches='tight')
            print("  Saved: plots/15_performance_radar.png")
            plt.close(fig4)
        
        print("\n" + "=" * 60)
        print("Analysis complete! Check the plots/ folder.")
        print("=" * 60)


if __name__ == "__main__":
    main()
