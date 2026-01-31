"""
F1 Race Animation - Cars moving around the track with real-time speed visualization.

This script creates an animated visualization of the race showing:
- All cars moving around the track in real-time (accelerated)
- Cars colored by their team colors
- Driver numbers as labels
- Real-time speed display
- Consistent timing and intervals throughout the race
"""

import os
import warnings
from pathlib import Path

import fastf1
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from datetime import timedelta

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configuration
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'animations')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Enable FastF1 cache
fastf1.Cache.enable_cache(CACHE_DIR)


def load_race_session(year=2025, event_name='Abu Dhabi Grand Prix'):
    """Load the race session with telemetry data."""
    print(f"Loading {year} {event_name} race session...")
    session = fastf1.get_session(year, event_name, 'R')
    session.load(telemetry=True, laps=True, weather=False)
    print(f"Session loaded: {session.name}")
    print(f"Total drivers: {len(session.drivers)}")
    return session


def get_driver_info(session):
    """Extract driver information including team colors."""
    driver_info = {}
    
    for driver_num in session.drivers:
        try:
            info = session.get_driver(driver_num)
            # Convert team color from hex to RGB
            team_color = info.get('TeamColor', 'FFFFFF')
            if team_color:
                # Ensure it's a valid hex color
                if not team_color.startswith('#'):
                    team_color = f'#{team_color}'
            else:
                team_color = '#FFFFFF'
            
            driver_info[driver_num] = {
                'abbreviation': info.get('Abbreviation', str(driver_num)),
                'number': driver_num,
                'team_name': info.get('TeamName', 'Unknown'),
                'team_color': team_color,
                'full_name': info.get('FullName', f'Driver {driver_num}'),
            }
        except Exception as e:
            print(f"Could not get info for driver {driver_num}: {e}")
            driver_info[driver_num] = {
                'abbreviation': str(driver_num),
                'number': driver_num,
                'team_name': 'Unknown',
                'team_color': '#FFFFFF',
                'full_name': f'Driver {driver_num}',
            }
    
    return driver_info


def get_track_coordinates(session):
    """Extract track coordinates from fastest lap telemetry."""
    # Get fastest lap from any driver to extract track shape
    fastest_lap = session.laps.pick_fastest()
    if fastest_lap is None or fastest_lap.empty:
        raise ValueError("No fastest lap found")
    
    telemetry = fastest_lap.get_telemetry()
    x = telemetry['X'].values
    y = telemetry['Y'].values
    
    return x, y


def prepare_position_data(session, driver_info, time_resolution_ms=100):
    """
    Prepare position data for all drivers at regular time intervals.
    
    This function interpolates car positions to ensure all drivers have
    positions at the same time points for synchronized animation.
    """
    print("Preparing position data for animation...")
    
    # Get the session time range
    all_positions = []
    driver_data = {}
    
    for driver_num in session.drivers:
        try:
            # Get position data for this driver
            if driver_num in session.pos_data and session.pos_data[driver_num] is not None:
                pos = session.pos_data[driver_num].copy()
                if pos.empty:
                    continue
                
                # Get car data for speed information
                car_data = None
                if driver_num in session.car_data and session.car_data[driver_num] is not None:
                    car_data = session.car_data[driver_num].copy()
                
                # Store the data
                driver_data[driver_num] = {
                    'pos': pos,
                    'car': car_data
                }
                
        except Exception as e:
            print(f"Error getting data for driver {driver_num}: {e}")
            continue
    
    if not driver_data:
        raise ValueError("No position data available for any driver")
    
    # Find common time range across all drivers
    min_time = pd.Timedelta(0)
    max_time = pd.Timedelta(hours=3)  # Max reasonable race duration
    
    for driver_num, data in driver_data.items():
        if 'SessionTime' in data['pos'].columns:
            driver_min = data['pos']['SessionTime'].min()
            driver_max = data['pos']['SessionTime'].max()
            if pd.notna(driver_min):
                min_time = max(min_time, driver_min)
            if pd.notna(driver_max):
                max_time = min(max_time, driver_max)
    
    print(f"Time range: {min_time} to {max_time}")
    
    # Create regular time intervals
    time_step = pd.Timedelta(milliseconds=time_resolution_ms)
    time_points = pd.timedelta_range(start=min_time, end=max_time, freq=time_step)
    
    print(f"Total time points: {len(time_points)}")
    
    # Interpolate positions for each driver at each time point
    interpolated_data = {
        'time': time_points,
        'drivers': {}
    }
    
    for driver_num, data in driver_data.items():
        pos = data['pos']
        car = data['car']
        
        if 'SessionTime' not in pos.columns:
            continue
        
        # Convert session time to numeric for interpolation
        pos = pos.dropna(subset=['SessionTime', 'X', 'Y'])
        if pos.empty:
            continue
        
        pos_times = pos['SessionTime'].dt.total_seconds().values
        pos_x = pos['X'].values
        pos_y = pos['Y'].values
        
        # Interpolate X, Y positions
        time_seconds = np.array([t.total_seconds() for t in time_points])
        
        # Use numpy interpolation
        x_interp = np.interp(time_seconds, pos_times, pos_x, left=np.nan, right=np.nan)
        y_interp = np.interp(time_seconds, pos_times, pos_y, left=np.nan, right=np.nan)
        
        # Interpolate speed if available
        speed_interp = np.full_like(time_seconds, np.nan)
        if car is not None and 'Speed' in car.columns and 'SessionTime' in car.columns:
            car = car.dropna(subset=['SessionTime', 'Speed'])
            if not car.empty:
                car_times = car['SessionTime'].dt.total_seconds().values
                car_speed = car['Speed'].values
                speed_interp = np.interp(time_seconds, car_times, car_speed, left=np.nan, right=np.nan)
        
        interpolated_data['drivers'][driver_num] = {
            'x': x_interp,
            'y': y_interp,
            'speed': speed_interp
        }
        
        print(f"  Driver {driver_num}: {np.sum(~np.isnan(x_interp))} valid positions")
    
    return interpolated_data


def create_animation(session, interpolated_data, driver_info, 
                     speed_multiplier=10, output_file='race_animation.mp4',
                     fps=30, dpi=150, figsize=(16, 10)):
    """
    Create the race animation.
    
    Args:
        session: FastF1 session object
        interpolated_data: Dictionary with interpolated position data
        driver_info: Dictionary with driver information
        speed_multiplier: How many times faster than real-time
        output_file: Output filename
        fps: Frames per second for the animation
        dpi: DPI for the animation
        figsize: Figure size (width, height)
    """
    print(f"\nCreating animation with {speed_multiplier}x speed...")
    
    # Get track coordinates for background
    track_x, track_y = get_track_coordinates(session)
    
    # Calculate frame skip based on speed multiplier
    # If time resolution is 100ms and we want 30fps at 10x speed:
    # Each frame should advance by (1000/30) * 10 = 333ms worth of data
    # With 100ms resolution, that's ~3-4 data points per frame
    time_resolution_ms = 100  # From prepare_position_data
    ms_per_frame = (1000 / fps) * speed_multiplier
    frame_skip = max(1, int(ms_per_frame / time_resolution_ms))
    
    time_points = interpolated_data['time']
    total_frames = len(time_points) // frame_skip
    
    print(f"Total data points: {len(time_points)}")
    print(f"Frame skip: {frame_skip}")
    print(f"Total animation frames: {total_frames}")
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    
    # Plot track outline
    ax.plot(track_x, track_y, color='#404040', linewidth=8, alpha=0.5, zorder=1)
    ax.plot(track_x, track_y, color='#606060', linewidth=4, alpha=0.8, zorder=2)
    
    # Set axis properties
    ax.set_aspect('equal')
    margin = 500
    ax.set_xlim(track_x.min() - margin, track_x.max() + margin)
    ax.set_ylim(track_y.min() - margin, track_y.max() + margin)
    ax.axis('off')
    
    # Create car markers and labels for each driver
    car_markers = {}
    car_labels = {}
    speed_texts = {}
    
    drivers_in_animation = [d for d in interpolated_data['drivers'].keys() if d in driver_info]
    
    for driver_num in drivers_in_animation:
        info = driver_info[driver_num]
        color = info['team_color']
        
        # Car marker (circle)
        marker, = ax.plot([], [], 'o', markersize=12, color=color, 
                         markeredgecolor='white', markeredgewidth=1.5, zorder=10)
        car_markers[driver_num] = marker
        
        # Driver number label
        label = ax.text(0, 0, str(driver_num), fontsize=7, fontweight='bold',
                       color='white', ha='center', va='center', zorder=11,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor=color, 
                                edgecolor='white', linewidth=0.5, alpha=0.9))
        car_labels[driver_num] = label
    
    # Add title and info text
    title_text = ax.text(0.5, 0.98, f"{session.event['EventName']} {session.event['EventDate'].year}",
                        transform=ax.transAxes, fontsize=18, fontweight='bold',
                        color='white', ha='center', va='top')
    
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=14,
                       color='white', ha='left', va='top', fontfamily='monospace')
    
    lap_text = ax.text(0.02, 0.93, '', transform=ax.transAxes, fontsize=12,
                      color='#aaaaaa', ha='left', va='top')
    
    # Create legend showing team colors
    legend_y = 0.9
    legend_entries = []
    teams_shown = set()
    
    for driver_num in sorted(drivers_in_animation, key=lambda d: driver_info[d]['team_name']):
        info = driver_info[driver_num]
        team_name = info['team_name']
        if team_name not in teams_shown and len(teams_shown) < 10:
            teams_shown.add(team_name)
            ax.plot([], [], 'o', markersize=8, color=info['team_color'], 
                   label=f"{info['abbreviation']} ({info['number']})")
    
    legend = ax.legend(loc='upper right', fontsize=8, framealpha=0.8,
                      facecolor='#2e2e2e', edgecolor='white', 
                      labelcolor='white', ncol=2, columnspacing=1,
                      title='Drivers', title_fontsize=10)
    legend.get_title().set_color('white')
    
    # Speed display panel
    speed_panel_ax = fig.add_axes([0.12, 0.02, 0.76, 0.12], facecolor='#2e2e2e')
    speed_panel_ax.set_xlim(0, 1)
    speed_panel_ax.set_ylim(0, 1)
    speed_panel_ax.axis('off')
    speed_panel_ax.set_title('Speed (km/h)', fontsize=10, color='white', pad=2)
    
    # Create speed bars for each driver
    speed_bars = {}
    bar_width = 0.8 / max(len(drivers_in_animation), 1)
    
    for i, driver_num in enumerate(sorted(drivers_in_animation, key=lambda d: driver_info[d]['team_name'])):
        info = driver_info[driver_num]
        x_pos = 0.1 + i * bar_width
        
        # Background bar
        speed_panel_ax.add_patch(plt.Rectangle((x_pos, 0.2), bar_width * 0.8, 0.6,
                                               facecolor='#404040', edgecolor='none'))
        
        # Speed bar (will be updated)
        bar = speed_panel_ax.add_patch(plt.Rectangle((x_pos, 0.2), bar_width * 0.8, 0,
                                                     facecolor=info['team_color'], 
                                                     edgecolor='none'))
        speed_bars[driver_num] = {
            'bar': bar,
            'x_pos': x_pos,
            'width': bar_width * 0.8
        }
        
        # Driver label
        speed_panel_ax.text(x_pos + bar_width * 0.4, 0.05, str(driver_num),
                           fontsize=6, color='white', ha='center', va='bottom')
    
    def init():
        """Initialize animation."""
        for marker in car_markers.values():
            marker.set_data([], [])
        for label in car_labels.values():
            label.set_position((0, 0))
            label.set_visible(False)
        return list(car_markers.values()) + list(car_labels.values())
    
    def animate(frame_idx):
        """Update animation for each frame."""
        data_idx = frame_idx * frame_skip
        
        if data_idx >= len(time_points):
            return list(car_markers.values()) + list(car_labels.values())
        
        current_time = time_points[data_idx]
        
        # Update time display
        total_seconds = current_time.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        time_text.set_text(f'Race Time: {hours:02d}:{minutes:02d}:{seconds:02d}')
        
        # Update each car's position
        for driver_num in drivers_in_animation:
            driver_data = interpolated_data['drivers'].get(driver_num)
            if driver_data is None:
                continue
            
            x = driver_data['x'][data_idx]
            y = driver_data['y'][data_idx]
            speed = driver_data['speed'][data_idx]
            
            if np.isnan(x) or np.isnan(y):
                car_markers[driver_num].set_data([], [])
                car_labels[driver_num].set_visible(False)
                if driver_num in speed_bars:
                    speed_bars[driver_num]['bar'].set_height(0)
            else:
                car_markers[driver_num].set_data([x], [y])
                car_labels[driver_num].set_position((x, y + 150))
                car_labels[driver_num].set_visible(True)
                
                # Update speed bar
                if driver_num in speed_bars and not np.isnan(speed):
                    # Normalize speed (assume max ~350 km/h)
                    normalized_speed = min(speed / 350.0, 1.0) * 0.6
                    speed_bars[driver_num]['bar'].set_height(normalized_speed)
        
        return list(car_markers.values()) + list(car_labels.values())
    
    # Create animation
    print("Rendering animation frames...")
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=total_frames, interval=1000/fps,
                                   blit=False)
    
    # Save animation
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_file)
    
    print(f"Saving animation to {output_path}...")
    print("This may take several minutes...")
    
    # Try different writers
    try:
        writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='F1 Analysis'),
                                         bitrate=5000)
        anim.save(output_path, writer=writer, dpi=dpi)
    except Exception as e:
        print(f"FFmpeg writer failed: {e}")
        print("Trying pillow writer for GIF output...")
        output_path = output_path.replace('.mp4', '.gif')
        writer = animation.PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=dpi//2)  # Lower DPI for GIF
    
    plt.close(fig)
    print(f"Animation saved to: {output_path}")
    
    return output_path


def create_static_snapshot(session, interpolated_data, driver_info, 
                           time_point_seconds, output_file='race_snapshot.png'):
    """Create a static snapshot of the race at a specific time."""
    print(f"\nCreating snapshot at {time_point_seconds}s...")
    
    # Get track coordinates
    track_x, track_y = get_track_coordinates(session)
    
    # Find the index for the requested time
    time_points = interpolated_data['time']
    time_seconds = np.array([t.total_seconds() for t in time_points])
    idx = np.argmin(np.abs(time_seconds - time_point_seconds))
    
    current_time = time_points[idx]
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    
    # Plot track
    ax.plot(track_x, track_y, color='#404040', linewidth=10, alpha=0.5, zorder=1)
    ax.plot(track_x, track_y, color='#606060', linewidth=5, alpha=0.8, zorder=2)
    
    # Plot each car
    for driver_num, driver_data in interpolated_data['drivers'].items():
        if driver_num not in driver_info:
            continue
        
        info = driver_info[driver_num]
        x = driver_data['x'][idx]
        y = driver_data['y'][idx]
        speed = driver_data['speed'][idx]
        
        if np.isnan(x) or np.isnan(y):
            continue
        
        color = info['team_color']
        
        # Plot car marker
        ax.plot(x, y, 'o', markersize=15, color=color,
               markeredgecolor='white', markeredgewidth=2, zorder=10)
        
        # Add driver number label
        ax.text(x, y + 200, str(driver_num), fontsize=9, fontweight='bold',
               color='white', ha='center', va='center', zorder=11,
               bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                        edgecolor='white', linewidth=1, alpha=0.95))
        
        # Add speed label
        if not np.isnan(speed):
            ax.text(x, y - 250, f'{speed:.0f} km/h', fontsize=7,
                   color='white', ha='center', va='center', zorder=11,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='#333333',
                            edgecolor='#666666', linewidth=0.5, alpha=0.9))
    
    # Set axis properties
    ax.set_aspect('equal')
    margin = 500
    ax.set_xlim(track_x.min() - margin, track_x.max() + margin)
    ax.set_ylim(track_y.min() - margin, track_y.max() + margin)
    ax.axis('off')
    
    # Add title
    total_seconds = current_time.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    ax.set_title(f"{session.event['EventName']} {session.event['EventDate'].year}\n"
                f"Race Time: {hours:02d}:{minutes:02d}:{seconds:02d}",
                fontsize=16, fontweight='bold', color='white', pad=20)
    
    # Add legend
    teams_shown = set()
    handles = []
    labels = []
    
    for driver_num in sorted(interpolated_data['drivers'].keys()):
        if driver_num not in driver_info:
            continue
        info = driver_info[driver_num]
        team_name = info['team_name']
        if team_name not in teams_shown:
            teams_shown.add(team_name)
            handle = plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=info['team_color'],
                               markersize=10, markeredgecolor='white',
                               markeredgewidth=1, linestyle='None')
            handles.append(handle)
            labels.append(team_name)
    
    legend = ax.legend(handles, labels, loc='upper right', fontsize=9,
                      framealpha=0.8, facecolor='#2e2e2e', edgecolor='white',
                      labelcolor='white', ncol=2, columnspacing=1,
                      title='Teams', title_fontsize=11)
    legend.get_title().set_color('white')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_file)
    plt.savefig(output_path, dpi=150, facecolor='#1e1e1e', edgecolor='none',
                bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    
    print(f"Snapshot saved to: {output_path}")
    return output_path


def main():
    """Main function to create the race animation."""
    print("=" * 60)
    print("F1 Race Animation Generator")
    print("=" * 60)
    
    # Load session
    session = load_race_session(year=2025, event_name='Abu Dhabi Grand Prix')
    
    # Get driver info
    driver_info = get_driver_info(session)
    print(f"\nLoaded info for {len(driver_info)} drivers:")
    for num, info in sorted(driver_info.items()):
        print(f"  #{num} {info['abbreviation']} - {info['team_name']} ({info['team_color']})")
    
    # Prepare interpolated position data
    interpolated_data = prepare_position_data(session, driver_info, time_resolution_ms=200)
    
    # Create static snapshots at different race points
    print("\n" + "=" * 60)
    print("Creating static snapshots...")
    print("=" * 60)
    
    # Early race (lap 1-2, around 2 minutes in)
    create_static_snapshot(session, interpolated_data, driver_info, 
                          time_point_seconds=3600 + 120,  # ~1 hour + 2 min into race
                          output_file='snapshot_early_race.png')
    
    # Mid race
    create_static_snapshot(session, interpolated_data, driver_info,
                          time_point_seconds=3600 + 30*60,  # ~1 hour + 30 min
                          output_file='snapshot_mid_race.png')
    
    # Create animation (this takes a while)
    print("\n" + "=" * 60)
    print("Creating animation...")
    print("=" * 60)
    
    # Create a shorter animation (first 5 minutes of race from lights out)
    # Race typically starts around 1 hour into session time
    create_animation(session, interpolated_data, driver_info,
                    speed_multiplier=20,  # 20x speed
                    output_file='race_animation.mp4',
                    fps=30, dpi=100)
    
    print("\n" + "=" * 60)
    print("All outputs generated!")
    print(f"Check the '{OUTPUT_DIR}' directory for files.")
    print("=" * 60)


if __name__ == "__main__":
    main()
