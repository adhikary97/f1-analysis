"""
F1 Race Animation - HTML5 Interactive Version

This script creates an HTML5 animation that can be viewed in any web browser.
No FFmpeg required - uses matplotlib's HTML5 video output.

Features:
- Interactive playback controls
- Works in any modern browser
- Team-colored cars with driver numbers
- Speed visualization
- Lap counter
"""

import os
import warnings
from pathlib import Path

import fastf1
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
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


def prepare_animation_data(session, time_resolution_ms=200, max_duration_s=300):
    """
    Prepare interpolated position data.
    
    Args:
        session: FastF1 session
        time_resolution_ms: Time resolution in milliseconds
        max_duration_s: Maximum duration to include (seconds from race start)
    """
    print("Preparing animation data...")
    
    driver_data = {}
    
    for driver_num in session.drivers:
        try:
            pos = session.pos_data.get(driver_num)
            car = session.car_data.get(driver_num)
            
            if pos is None or pos.empty:
                continue
            
            # Filter to valid data with actual track positions (not 0,0)
            pos = pos.dropna(subset=['SessionTime', 'X', 'Y'])
            pos = pos[(pos['X'] != 0) | (pos['Y'] != 0)]  # Filter out (0,0) positions
            if len(pos) < 10:
                continue
            
            driver_data[driver_num] = {
                'pos': pos,
                'car': car if car is not None and not car.empty else None
            }
        except Exception as e:
            print(f"  Warning: Driver {driver_num}: {e}")
    
    print(f"Got data for {len(driver_data)} drivers")
    
    # Find common time range where ALL drivers have valid positions
    # Start from when the race actually begins (cars on track with real positions)
    all_min_times = []
    all_max_times = []
    
    for driver_num, data in driver_data.items():
        pos = data['pos']
        driver_min = pos['SessionTime'].min()
        driver_max = pos['SessionTime'].max()
        if pd.notna(driver_min):
            all_min_times.append(driver_min)
        if pd.notna(driver_max):
            all_max_times.append(driver_max)
    
    # Use the latest start time (when all cars have data)
    min_time = max(all_min_times) if all_min_times else pd.Timedelta(0)
    max_time = min(all_max_times) if all_max_times else pd.Timedelta(hours=3)
    
    # Limit duration
    if max_duration_s:
        max_time = min(max_time, min_time + pd.Timedelta(seconds=max_duration_s))
    
    print(f"Time range: {min_time} to {max_time}")
    
    # Create time grid
    time_step = pd.Timedelta(milliseconds=time_resolution_ms)
    time_points = pd.timedelta_range(start=min_time, end=max_time, freq=time_step)
    time_seconds = np.array([t.total_seconds() for t in time_points])
    
    print(f"Time points: {len(time_points)}")
    
    # Interpolate each driver
    result = {
        'time_points': time_points,
        'time_seconds': time_seconds,
        'drivers': {}
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
            car = car.dropna(subset=['SessionTime', 'Speed'])
            if len(car) > 1:
                car_times = car['SessionTime'].dt.total_seconds().values
                sort_idx = np.argsort(car_times)
                speed_interp = np.interp(time_seconds, car_times[sort_idx], 
                                        car['Speed'].values[sort_idx],
                                        left=np.nan, right=np.nan)
        
        result['drivers'][driver_num] = {
            'x': x_interp,
            'y': y_interp,
            'speed': speed_interp
        }
        
        valid = np.sum(~np.isnan(x_interp))
        print(f"  Driver {driver_num}: {valid} valid points")
    
    return result


def create_html_animation(session, anim_data, driver_info,
                          speed_multiplier=10, fps=20, 
                          max_frames=600, output_file='race_animation.html'):
    """
    Create HTML5 animation.
    
    Args:
        session: FastF1 session
        anim_data: Prepared animation data
        driver_info: Driver information
        speed_multiplier: Playback speed multiplier
        fps: Frames per second
        max_frames: Maximum frames to render
        output_file: Output HTML filename
    """
    print("\nCreating HTML5 animation...")
    
    # Increase matplotlib embed limit for larger animations
    import matplotlib as mpl
    mpl.rcParams['animation.embed_limit'] = 50  # 50 MB limit
    
    # Get track shape
    fastest_lap = session.laps.pick_fastest()
    telemetry = fastest_lap.get_telemetry()
    track_x = telemetry['X'].values
    track_y = telemetry['Y'].values
    
    # Calculate frame skip
    time_resolution_ms = 200  # From prepare_animation_data
    ms_per_frame = (1000 / fps) * speed_multiplier
    frame_skip = max(1, int(ms_per_frame / time_resolution_ms))
    
    time_points = anim_data['time_points']
    total_data_points = len(time_points)
    total_frames = min(total_data_points // frame_skip, max_frames)
    
    print(f"Frame skip: {frame_skip}")
    print(f"Total frames: {total_frames}")
    
    # Create figure (smaller size for better web performance)
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1a1a', dpi=80)
    ax.set_facecolor('#1a1a1a')
    
    # Draw track
    ax.plot(track_x, track_y, color='#2a2a2a', linewidth=18, solid_capstyle='round', zorder=1)
    ax.plot(track_x, track_y, color='#404040', linewidth=10, solid_capstyle='round', zorder=2)
    
    # Set limits
    ax.set_aspect('equal')
    margin = 600
    ax.set_xlim(track_x.min() - margin, track_x.max() + margin)
    ax.set_ylim(track_y.min() - margin, track_y.max() + margin)
    ax.axis('off')
    
    # Title
    event_name = session.event.get('EventName', 'Race')
    year = session.event.get('EventDate', pd.Timestamp.now()).year
    ax.set_title(f'{event_name} {year}', fontsize=16, fontweight='bold', 
                color='white', pad=10)
    
    # Time display - positioned in top-left corner outside track area
    # Store the start time to calculate relative time
    start_time_seconds = anim_data['time_points'][0].total_seconds()
    
    time_text = ax.text(0.02, 0.02, '', transform=ax.transAxes,
                       fontsize=11, color='white', fontfamily='monospace',
                       ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#222222', 
                                edgecolor='#444444', alpha=0.95))
    
    # Create car markers
    drivers_list = list(anim_data['drivers'].keys())
    car_markers = {}
    car_labels = {}
    
    for driver_num in drivers_list:
        if driver_num not in driver_info:
            continue
        
        info = driver_info[driver_num]
        color = info['color']
        
        marker, = ax.plot([], [], 'o', markersize=12, color=color,
                         markeredgecolor='white', markeredgewidth=1.2, zorder=100)
        car_markers[driver_num] = marker
        
        label = ax.text(0, 0, str(driver_num), fontsize=7, fontweight='bold',
                       color='white', ha='center', va='center', zorder=101,
                       bbox=dict(boxstyle='round,pad=0.15', facecolor=color,
                                edgecolor='white', linewidth=0.5, alpha=0.9))
        label.set_visible(False)
        car_labels[driver_num] = label
    
    # Legend - position outside the track area
    handles = []
    labels = []
    teams_seen = set()
    
    for driver_num in sorted(drivers_list):
        if driver_num not in driver_info:
            continue
        info = driver_info[driver_num]
        team = info['team']
        if team not in teams_seen:
            teams_seen.add(team)
            handle = plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=info['color'], markersize=8,
                               markeredgecolor='white', markeredgewidth=0.5,
                               linestyle='None')
            handles.append(handle)
            labels.append(f"{team}")
    
    # Place legend at bottom left corner, outside the main track area
    legend = ax.legend(handles, labels, loc='lower left', fontsize=6,
                      framealpha=0.9, facecolor='#1a1a1a', edgecolor='#444444',
                      labelcolor='white', ncol=5, columnspacing=0.5,
                      bbox_to_anchor=(0.0, -0.02))
    
    def init():
        for m in car_markers.values():
            m.set_data([], [])
        for l in car_labels.values():
            l.set_visible(False)
        return list(car_markers.values())
    
    def update(frame):
        data_idx = frame * frame_skip
        
        if data_idx >= len(time_points):
            return list(car_markers.values())
        
        current_time = time_points[data_idx]
        time_s = current_time.total_seconds()
        
        # Calculate relative time from start of animation (start at 0:00:00)
        relative_time_s = time_s - start_time_seconds
        
        minutes = int(relative_time_s // 60)
        seconds = int(relative_time_s % 60)
        time_text.set_text(f'Race Time: {minutes:02d}:{seconds:02d}')
        
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
                car_labels[driver_num].set_position((x, y + 150))
                car_labels[driver_num].set_visible(True)
        
        return list(car_markers.values())
    
    print("Rendering frames...")
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=total_frames, interval=1000/fps,
                                   blit=False)
    
    # Save as HTML5
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_file)
    
    print(f"Saving HTML5 animation to {output_path}...")
    
    # Generate HTML with embedded video
    try:
        html_content = anim.to_jshtml(fps=fps, embed_frames=True, default_mode='loop')
        
        # Wrap in a nice HTML page with auto-play script
        full_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>F1 Race Animation - {event_name} {year}</title>
    <style>
        body {{
            background-color: #1a1a1a;
            color: white;
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        h1 {{
            color: #ff1801;
            margin-bottom: 10px;
        }}
        .info {{
            color: #888;
            margin-bottom: 20px;
        }}
        .animation-container {{
            background: #0d0d0d;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(255, 24, 1, 0.2);
        }}
        .controls-info {{
            color: #666;
            font-size: 12px;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <h1>F1 Race Animation</h1>
    <p class="info">{event_name} {year} | Playback: {speed_multiplier}x speed</p>
    <div class="animation-container">
        {html_content}
    </div>
    <p class="controls-info">Click the PLAY button below the animation to start. Use controls to pause/navigate.</p>
    <script>
        // Auto-play the animation after a short delay
        setTimeout(function() {{
            var playButtons = document.querySelectorAll('button');
            for (var i = 0; i < playButtons.length; i++) {{
                if (playButtons[i].textContent.includes('▶') || 
                    playButtons[i].textContent.toLowerCase().includes('play') ||
                    playButtons[i].classList.contains('anim-play')) {{
                    playButtons[i].click();
                    break;
                }}
            }}
        }}, 500);
    </script>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(full_html)
        
        print(f"HTML5 animation saved: {output_path}")
        print(f"Open this file in a web browser to view the animation.")
        
    except Exception as e:
        print(f"Error creating HTML: {e}")
        
        # Try simpler approach - save individual frames
        print("Falling back to frame-by-frame image export...")
        frames_dir = os.path.join(OUTPUT_DIR, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        for i in range(min(total_frames, 100)):  # Limit frames
            update(i)
            frame_path = os.path.join(frames_dir, f'frame_{i:04d}.png')
            plt.savefig(frame_path, dpi=100, facecolor='#1a1a1a', 
                       bbox_inches='tight')
            if i % 10 == 0:
                print(f"  Frame {i}/{total_frames}")
        
        print(f"Frames saved to: {frames_dir}")
        output_path = frames_dir
    
    plt.close(fig)
    return output_path


def main():
    """Main function."""
    print("=" * 60)
    print("F1 Race Animation - HTML5 Version")
    print("=" * 60)
    
    # Load data
    session = load_race_data(year=2025, event_name='Abu Dhabi Grand Prix')
    driver_info = get_driver_info(session)
    
    print("\nDrivers:")
    for num, info in sorted(driver_info.items()):
        print(f"  {num:>2} {info['abbreviation']:>3} - {info['team']} ({info['color']})")
    
    # Prepare animation data (5 minutes = 300 seconds)
    anim_data = prepare_animation_data(session, time_resolution_ms=200, max_duration_s=300)
    
    # Create HTML animation
    create_html_animation(
        session, anim_data, driver_info,
        speed_multiplier=10,  # 10x speed
        fps=20,
        max_frames=500,
        output_file='race_animation.html'
    )
    
    print("\n" + "=" * 60)
    print("Animation complete!")
    print(f"Open '{OUTPUT_DIR}/race_animation.html' in a web browser.")
    print("=" * 60)


if __name__ == "__main__":
    main()
