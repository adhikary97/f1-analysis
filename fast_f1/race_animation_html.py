"""
F1 Race Animation - HTML5 Interactive Version with Leaderboard

Features:
- Cars moving around track with correct team colors
- Driver number labels
- Accurate race timing from Lap 1
- Live leaderboard showing positions and gaps
- Works without FFmpeg
"""

import os
import warnings
from pathlib import Path

import fastf1
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'animations')

# Enable FastF1 cache
fastf1.Cache.enable_cache(CACHE_DIR)

# F1 2025 Team Colors
TEAM_COLORS = {
    'Red Bull Racing': '#3671C6',
    'McLaren': '#FF8000',
    'Ferrari': '#E8002D',
    'Mercedes': '#27F4D2',
    'Aston Martin': '#229971',
    'Alpine': '#00A1E8',
    'Williams': '#1868DB',
    'Racing Bulls': '#6C98FF',
    'Kick Sauber': '#01C00E',
    'Haas F1 Team': '#9C9FA2',
}


def load_race_data(year=2025, event_name='Abu Dhabi Grand Prix'):
    """Load race data from FastF1."""
    print(f"Loading {year} {event_name}...")
    session = fastf1.get_session(year, event_name, 'R')
    session.load(telemetry=True, laps=True, weather=False)
    return session


def get_driver_info(session):
    """Get driver info with team colors."""
    driver_info = {}
    
    for driver_num in session.drivers:
        try:
            info = session.get_driver(driver_num)
            team_name = info.get('TeamName', 'Unknown')
            color = TEAM_COLORS.get(team_name)
            
            if color is None:
                tc = info.get('TeamColor', 'FFFFFF')
                color = f'#{tc}' if tc and not tc.startswith('#') else (tc or '#FFFFFF')
            
            driver_info[driver_num] = {
                'number': driver_num,
                'abbreviation': info.get('Abbreviation', str(driver_num)),
                'team': team_name,
                'color': color,
                'name': info.get('FullName', f'Driver {driver_num}'),
            }
        except:
            driver_info[driver_num] = {
                'number': driver_num,
                'abbreviation': str(driver_num),
                'team': 'Unknown',
                'color': '#FFFFFF',
                'name': f'Driver {driver_num}',
            }
    
    return driver_info


def get_race_start_time(session):
    """Find the actual race start time (Lap 1 start for leader)."""
    laps = session.laps
    lap1_data = laps[laps['LapNumber'] == 1]
    lap1_starts = lap1_data['LapStartTime'].dropna()
    
    if len(lap1_starts) > 0:
        race_start = lap1_starts.min()
        return race_start
    
    return pd.Timedelta(seconds=3500)


def prepare_animation_data(session, time_resolution_ms=250, num_laps=15):
    """
    Prepare interpolated position data with lap timing information.
    """
    print("Preparing animation data...")
    
    laps = session.laps
    
    # Get race start time
    race_start_time = get_race_start_time(session)
    race_start_seconds = race_start_time.total_seconds()
    print(f"Race start time: {race_start_time} ({race_start_seconds:.1f}s into session)")
    
    # Get actual lap times to calculate duration
    leader_laps = laps[laps['Driver'] == 'VER'].sort_values('LapNumber')
    if len(leader_laps) >= num_laps:
        target_lap = leader_laps[leader_laps['LapNumber'] == num_laps]
        if not target_lap.empty:
            race_end_time = target_lap['Time'].iloc[0]
        else:
            # Estimate based on average lap time
            avg_lap = leader_laps['LapTime'].mean()
            if pd.notna(avg_lap):
                race_end_time = race_start_time + avg_lap * num_laps
            else:
                race_end_time = race_start_time + pd.Timedelta(seconds=num_laps * 90)
    else:
        race_end_time = race_start_time + pd.Timedelta(seconds=num_laps * 90)
    
    duration_seconds = (race_end_time - race_start_time).total_seconds()
    print(f"Animating {num_laps} laps ({duration_seconds:.0f}s = {duration_seconds/60:.1f} minutes)")
    
    driver_data = {}
    
    for driver_num in session.drivers:
        try:
            pos = session.pos_data.get(driver_num)
            car = session.car_data.get(driver_num)
            
            if pos is None or pos.empty:
                continue
            
            # Filter to valid data
            pos = pos.dropna(subset=['SessionTime', 'X', 'Y'])
            pos = pos[(pos['X'] != 0) | (pos['Y'] != 0)]
            pos = pos[(pos['SessionTime'] >= race_start_time) & 
                      (pos['SessionTime'] <= race_end_time)]
            
            if len(pos) < 10:
                continue
            
            driver_data[driver_num] = {
                'pos': pos,
                'car': car if car is not None and not car.empty else None
            }
        except Exception as e:
            print(f"  Warning: Driver {driver_num}: {e}")
    
    print(f"Got data for {len(driver_data)} drivers")
    
    # Create time grid
    time_step = pd.Timedelta(milliseconds=time_resolution_ms)
    time_points = pd.timedelta_range(start=race_start_time, end=race_end_time, freq=time_step)
    time_seconds = np.array([t.total_seconds() for t in time_points])
    
    print(f"Time points: {len(time_points)}")
    
    # Interpolate each driver
    result = {
        'time_points': time_points,
        'time_seconds': time_seconds,
        'race_start_seconds': race_start_seconds,
        'drivers': {},
        'laps_data': laps,
    }
    
    for driver_num, data in driver_data.items():
        pos = data['pos']
        car = data['car']
        
        pos_times = pos['SessionTime'].dt.total_seconds().values
        sort_idx = np.argsort(pos_times)
        pos_times = pos_times[sort_idx]
        pos_x = pos['X'].values[sort_idx]
        pos_y = pos['Y'].values[sort_idx]
        
        x_interp = np.interp(time_seconds, pos_times, pos_x, left=np.nan, right=np.nan)
        y_interp = np.interp(time_seconds, pos_times, pos_y, left=np.nan, right=np.nan)
        
        speed_interp = np.full_like(time_seconds, np.nan)
        if car is not None and 'Speed' in car.columns and 'SessionTime' in car.columns:
            car_filtered = car[(car['SessionTime'] >= race_start_time) & 
                               (car['SessionTime'] <= race_end_time)]
            car_filtered = car_filtered.dropna(subset=['SessionTime', 'Speed'])
            if len(car_filtered) > 1:
                car_times = car_filtered['SessionTime'].dt.total_seconds().values
                sort_idx = np.argsort(car_times)
                speed_interp = np.interp(time_seconds, car_times[sort_idx], 
                                        car_filtered['Speed'].values[sort_idx],
                                        left=np.nan, right=np.nan)
        
        result['drivers'][driver_num] = {
            'x': x_interp,
            'y': y_interp,
            'speed': speed_interp
        }
        
        valid = np.sum(~np.isnan(x_interp))
        print(f"  Driver {driver_num}: {valid} valid points")
    
    # Pre-compute positions and gaps for each time point
    print("Computing positions and gaps...")
    result['positions'] = compute_positions_and_gaps(laps, time_seconds, race_start_seconds)
    
    return result


def compute_positions_and_gaps(laps, time_seconds, race_start_seconds):
    """
    Compute race positions and gaps at each time point based on lap data.
    """
    positions_data = []
    
    # Group laps by driver
    driver_laps = {}
    for driver in laps['Driver'].unique():
        dlaps = laps[laps['Driver'] == driver].sort_values('LapNumber')
        if not dlaps.empty:
            driver_laps[driver] = dlaps
    
    for t_idx, t in enumerate(time_seconds):
        # For each driver, find their current lap and progress
        driver_status = []
        
        for driver, dlaps in driver_laps.items():
            # Find which lap they're on
            current_lap = 0
            lap_progress = 0
            last_lap_time = race_start_seconds
            
            for _, lap in dlaps.iterrows():
                lap_end_time = lap.get('Time')
                if pd.notna(lap_end_time):
                    lap_end_s = lap_end_time.total_seconds()
                    if t <= lap_end_s:
                        current_lap = lap.get('LapNumber', 0)
                        lap_start = lap.get('LapStartTime')
                        if pd.notna(lap_start):
                            lap_start_s = lap_start.total_seconds()
                            lap_duration = lap_end_s - lap_start_s
                            if lap_duration > 0:
                                lap_progress = (t - lap_start_s) / lap_duration
                            last_lap_time = lap_end_s
                        break
                    else:
                        current_lap = lap.get('LapNumber', 0)
                        last_lap_time = lap_end_s
            
            # Calculate total race progress (laps + fraction of current lap)
            total_progress = current_lap + lap_progress - 1  # -1 because lap 1 is the first lap
            
            driver_status.append({
                'driver': driver,
                'lap': current_lap,
                'progress': total_progress,
                'last_time': last_lap_time,
            })
        
        # Sort by progress (descending) to get positions
        driver_status.sort(key=lambda x: -x['progress'])
        
        # Calculate gaps
        position_info = {}
        leader_progress = driver_status[0]['progress'] if driver_status else 0
        
        for pos, ds in enumerate(driver_status):
            gap_to_leader = leader_progress - ds['progress']
            gap_to_front = 0
            if pos > 0:
                gap_to_front = driver_status[pos-1]['progress'] - ds['progress']
            
            # Convert progress gap to approximate time (using ~90s lap time)
            gap_seconds = gap_to_front * 90
            leader_gap_seconds = gap_to_leader * 90
            
            position_info[ds['driver']] = {
                'position': pos + 1,
                'lap': ds['lap'],
                'gap_to_front': gap_seconds,
                'gap_to_leader': leader_gap_seconds,
            }
        
        positions_data.append(position_info)
    
    return positions_data


def create_html_animation(session, anim_data, driver_info,
                          speed_multiplier=10, fps=20, 
                          max_frames=None, output_file='race_animation.html'):
    """
    Create HTML5 animation with leaderboard.
    """
    print("\nCreating HTML5 animation...")
    
    mpl.rcParams['animation.embed_limit'] = 1000  # 1 GB limit for real-time animations
    
    # Get track shape
    fastest_lap = session.laps.pick_fastest()
    telemetry = fastest_lap.get_telemetry()
    track_x = telemetry['X'].values
    track_y = telemetry['Y'].values
    
    # Calculate frame skip
    time_resolution_ms = 250
    ms_per_frame = (1000 / fps) * speed_multiplier
    frame_skip = max(1, int(ms_per_frame / time_resolution_ms))
    
    time_points = anim_data['time_points']
    race_start_seconds = anim_data['race_start_seconds']
    positions_data = anim_data['positions']
    total_data_points = len(time_points)
    total_frames = total_data_points // frame_skip
    
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    
    print(f"Frame skip: {frame_skip}")
    print(f"Total frames: {total_frames}")
    
    # Create figure with track and leaderboard panels
    fig = plt.figure(figsize=(16, 9), facecolor='#1a1a1a')
    
    # Track area (left side - 75% width)
    ax_track = fig.add_axes([0.02, 0.08, 0.68, 0.88], facecolor='#1a1a1a')
    
    # Leaderboard area (right side - 25% width)
    ax_board = fig.add_axes([0.72, 0.08, 0.26, 0.88], facecolor='#1a1a1a')
    
    # Draw track
    ax_track.plot(track_x, track_y, color='#2a2a2a', linewidth=16, solid_capstyle='round', zorder=1)
    ax_track.plot(track_x, track_y, color='#404040', linewidth=8, solid_capstyle='round', zorder=2)
    
    # Set track limits
    ax_track.set_aspect('equal')
    margin = 500
    ax_track.set_xlim(track_x.min() - margin, track_x.max() + margin)
    ax_track.set_ylim(track_y.min() - margin, track_y.max() + margin)
    ax_track.axis('off')
    
    # Title
    event_name = session.event.get('EventName', 'Race')
    year = session.event.get('EventDate', pd.Timestamp.now()).year
    ax_track.set_title(f'{event_name} {year}', fontsize=14, fontweight='bold', 
                       color='white', pad=5)
    
    # Lap/Time info
    info_text = ax_track.text(0.02, 0.98, '', transform=ax_track.transAxes,
                              fontsize=12, color='white', fontfamily='monospace',
                              ha='left', va='top',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='#222222', 
                                       edgecolor='#444444', alpha=0.95))
    
    # Setup leaderboard
    ax_board.set_xlim(0, 1)
    ax_board.set_ylim(0, 1)
    ax_board.axis('off')
    
    # Leaderboard title
    ax_board.text(0.5, 0.98, 'LEADERBOARD', fontsize=11, fontweight='bold',
                  color='white', ha='center', va='top', transform=ax_board.transAxes)
    ax_board.text(0.5, 0.94, 'Gap to car ahead', fontsize=8, color='#888888',
                  ha='center', va='top', transform=ax_board.transAxes)
    
    # Create leaderboard entries (20 drivers)
    drivers_list = list(anim_data['drivers'].keys())
    board_entries = {}
    
    row_height = 0.042
    start_y = 0.90
    
    for i in range(20):
        y_pos = start_y - (i * row_height)
        
        # Position number
        pos_text = ax_board.text(0.08, y_pos, f'{i+1}', fontsize=9, color='white',
                                 ha='center', va='center', fontweight='bold',
                                 transform=ax_board.transAxes)
        
        # Team color bar (will be updated)
        color_bar = ax_board.add_patch(FancyBboxPatch((0.12, y_pos - 0.015), 0.04, 0.028,
                                                      boxstyle="round,pad=0.01",
                                                      facecolor='#444444',
                                                      edgecolor='none',
                                                      transform=ax_board.transAxes))
        
        # Driver abbreviation
        abbr_text = ax_board.text(0.22, y_pos, '---', fontsize=9, color='white',
                                  ha='left', va='center', fontfamily='monospace',
                                  fontweight='bold', transform=ax_board.transAxes)
        
        # Gap text
        gap_text = ax_board.text(0.95, y_pos, '', fontsize=8, color='#aaaaaa',
                                 ha='right', va='center', fontfamily='monospace',
                                 transform=ax_board.transAxes)
        
        board_entries[i] = {
            'pos_text': pos_text,
            'color_bar': color_bar,
            'abbr_text': abbr_text,
            'gap_text': gap_text,
        }
    
    # Create car markers
    car_markers = {}
    car_labels = {}
    
    for driver_num in drivers_list:
        if driver_num not in driver_info:
            continue
        
        info = driver_info[driver_num]
        color = info['color']
        
        marker, = ax_track.plot([], [], 'o', markersize=10, color=color,
                                markeredgecolor='white', markeredgewidth=1, zorder=100)
        car_markers[driver_num] = marker
        
        label = ax_track.text(0, 0, str(driver_num), fontsize=6, fontweight='bold',
                              color='white', ha='center', va='center', zorder=101,
                              bbox=dict(boxstyle='round,pad=0.12', facecolor=color,
                                       edgecolor='white', linewidth=0.5, alpha=0.9))
        label.set_visible(False)
        car_labels[driver_num] = label
    
    # Create a mapping from driver abbreviation to driver number
    abbr_to_num = {}
    for num, info in driver_info.items():
        abbr_to_num[info['abbreviation']] = num
    
    def init():
        for m in car_markers.values():
            m.set_data([], [])
        for l in car_labels.values():
            l.set_visible(False)
        return list(car_markers.values())
    
    def update(frame):
        data_idx = frame * frame_skip
        
        if data_idx >= len(time_points) or data_idx >= len(positions_data):
            return list(car_markers.values())
        
        current_time = time_points[data_idx]
        session_time_s = current_time.total_seconds()
        race_time_s = session_time_s - race_start_seconds
        
        minutes = int(race_time_s // 60)
        seconds = int(race_time_s % 60)
        
        # Get position data for current time
        pos_info = positions_data[data_idx]
        
        # Find current lap from leader
        leader_lap = 1
        for driver, pdata in pos_info.items():
            if pdata['position'] == 1:
                leader_lap = pdata['lap']
                break
        
        info_text.set_text(f'Lap {int(leader_lap)}/58\n{minutes:02d}:{seconds:02d}')
        
        # Update car positions on track
        for driver_num in drivers_list:
            if driver_num not in driver_info:
                continue
            
            ddata = anim_data['drivers'].get(driver_num)
            if ddata is None:
                continue
            
            x = ddata['x'][data_idx]
            y = ddata['y'][data_idx]
            
            if np.isnan(x) or np.isnan(y):
                car_markers[driver_num].set_data([], [])
                car_labels[driver_num].set_visible(False)
            else:
                car_markers[driver_num].set_data([x], [y])
                car_labels[driver_num].set_position((x, y + 120))
                car_labels[driver_num].set_visible(True)
        
        # Update leaderboard
        sorted_drivers = sorted(pos_info.items(), key=lambda x: x[1]['position'])
        
        for i, (driver_abbr, pdata) in enumerate(sorted_drivers[:20]):
            if i >= 20:
                break
            
            entry = board_entries[i]
            
            # Find driver number from abbreviation
            driver_num = abbr_to_num.get(driver_abbr, driver_abbr)
            
            if driver_num in driver_info:
                info = driver_info[driver_num]
                entry['color_bar'].set_facecolor(info['color'])
                entry['abbr_text'].set_text(info['abbreviation'])
            else:
                entry['abbr_text'].set_text(str(driver_abbr)[:3])
            
            # Gap text
            if pdata['position'] == 1:
                entry['gap_text'].set_text('LEADER')
                entry['gap_text'].set_color('#00ff00')
            else:
                gap = pdata['gap_to_front']
                if gap < 1:
                    entry['gap_text'].set_text(f'+{gap:.3f}')
                elif gap < 60:
                    entry['gap_text'].set_text(f'+{gap:.1f}s')
                else:
                    laps_behind = int(gap / 90)
                    entry['gap_text'].set_text(f'+{laps_behind} LAP{"S" if laps_behind > 1 else ""}')
                entry['gap_text'].set_color('#aaaaaa')
        
        return list(car_markers.values())
    
    print("Rendering frames...")
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=total_frames, interval=1000/fps,
                                   blit=False)
    
    # Save as HTML5
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_file)
    
    print(f"Saving HTML5 animation to {output_path}...")
    print("This may take a few minutes for longer animations...")
    
    try:
        html_content = anim.to_jshtml(fps=fps, embed_frames=True, default_mode='loop')
        
        full_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>F1 Race Animation - {event_name} {year}</title>
    <style>
        body {{
            background-color: #0d0d0d;
            color: white;
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 15px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        h1 {{
            color: #ff1801;
            margin-bottom: 5px;
            font-size: 24px;
        }}
        .info {{
            color: #888;
            margin-bottom: 15px;
            font-size: 14px;
        }}
        .animation-container {{
            background: #1a1a1a;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(255, 24, 1, 0.3);
        }}
        .controls-info {{
            color: #555;
            font-size: 11px;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <h1>F1 Race Animation</h1>
    <p class="info">{event_name} {year} | {speed_multiplier}x Speed | Shows positions and gaps</p>
    <div class="animation-container">
        {html_content}
    </div>
    <p class="controls-info">Click PLAY (▶) to start | Loop mode enabled</p>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(full_html)
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"HTML5 animation saved: {output_path} ({file_size_mb:.1f} MB)")
        
    except Exception as e:
        print(f"Error creating HTML: {e}")
        import traceback
        traceback.print_exc()
    
    plt.close(fig)
    return output_path


def main():
    """Main function."""
    print("=" * 60)
    print("F1 Race Animation with Leaderboard")
    print("=" * 60)
    
    session = load_race_data(year=2025, event_name='Abu Dhabi Grand Prix')
    driver_info = get_driver_info(session)
    
    print("\nDrivers:")
    for num, info in sorted(driver_info.items()):
        print(f"  {num:>2} {info['abbreviation']:>3} - {info['team']}")
    
    # Prepare animation data (15 laps)
    anim_data = prepare_animation_data(session, time_resolution_ms=250, num_laps=15)
    
    # Create HTML animation
    create_html_animation(
        session, anim_data, driver_info,
        speed_multiplier=10,
        fps=20,
        max_frames=1000,
        output_file='race_animation.html'
    )
    
    print("\n" + "=" * 60)
    print("Animation complete!")
    print(f"Open '{OUTPUT_DIR}/race_animation.html' in a web browser.")
    print("=" * 60)


if __name__ == "__main__":
    main()
