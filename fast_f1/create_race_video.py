#!/usr/bin/env python3
"""
F1 Race Animation Video Creator

Creates a smooth MP4 video of the race with:
- All cars moving around the track with team colors
- Driver numbers labeled on each car
- Live leaderboard showing positions and gaps
- Lap counter and race timer
- Real-time playback speed

Requirements:
    - fastf1
    - matplotlib
    - numpy
    - pandas
    - ffmpeg (must be installed: brew install ffmpeg)

Usage:
    python create_race_video.py                    # Default: 15 laps, 20fps
    python create_race_video.py --laps 10          # First 10 laps
    python create_race_video.py --laps 58          # Full race
    python create_race_video.py --fps 30           # Higher FPS (smoother but larger file)
    python create_race_video.py --speed 2          # 2x speed playback
"""

import os
import argparse
import warnings
warnings.filterwarnings('ignore')

import fastf1
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

# Configuration
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'animations')

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


def create_race_video(year=2025, event_name='Abu Dhabi Grand Prix', 
                      num_laps=15, fps=20, speed_multiplier=1,
                      output_filename=None):
    """
    Create a race animation video.
    
    Args:
        year: Race year
        event_name: Name of the Grand Prix
        num_laps: Number of laps to animate
        fps: Frames per second (higher = smoother but larger file)
        speed_multiplier: Playback speed (1 = real-time, 2 = 2x speed, etc.)
        output_filename: Output filename (default: auto-generated)
    """
    
    # Enable cache
    fastf1.Cache.enable_cache(CACHE_DIR)
    
    print("=" * 60)
    print("F1 Race Animation Video Creator")
    print("=" * 60)
    
    # Load session
    print(f'\nLoading {year} {event_name}...')
    session = fastf1.get_session(year, event_name, 'R')
    session.load(telemetry=True, laps=True, weather=False)
    print(f'Loaded data for {len(session.drivers)} drivers')
    
    # Get driver info
    driver_info = {}
    for driver_num in session.drivers:
        try:
            info = session.get_driver(driver_num)
            team = info.get('TeamName', 'Unknown')
            driver_info[driver_num] = {
                'abbreviation': info.get('Abbreviation', str(driver_num)),
                'team': team,
                'color': TEAM_COLORS.get(team, '#FFFFFF'),
            }
        except:
            driver_info[driver_num] = {
                'abbreviation': str(driver_num), 
                'team': 'Unknown', 
                'color': '#FFFFFF'
            }
    
    # Get race timing
    laps = session.laps
    lap1_starts = laps[laps['LapNumber'] == 1]['LapStartTime'].dropna()
    race_start = lap1_starts.min()
    race_start_s = race_start.total_seconds()
    
    # Calculate race end time
    leader_laps = laps[laps['Driver'] == 'VER'].sort_values('LapNumber')
    target_lap = leader_laps[leader_laps['LapNumber'] == num_laps]
    if not target_lap.empty:
        race_end = target_lap['Time'].iloc[0]
    else:
        race_end = race_start + pd.Timedelta(seconds=num_laps * 90)
    
    duration_s = (race_end - race_start).total_seconds()
    print(f'\nRace segment: Lap 1 to Lap {num_laps}')
    print(f'Duration: {duration_s:.0f}s ({duration_s/60:.1f} minutes)')
    
    # Calculate time resolution based on FPS
    # For real-time: time_res_ms = 1000 / fps
    time_res_ms = int(1000 / fps / speed_multiplier)
    time_res_ms = max(20, min(time_res_ms, 200))  # Clamp between 20-200ms
    
    time_step = pd.Timedelta(milliseconds=time_res_ms)
    time_points = pd.timedelta_range(start=race_start, end=race_end, freq=time_step)
    time_seconds = np.array([t.total_seconds() for t in time_points])
    
    # Adjust FPS for actual playback speed
    actual_fps = int(1000 / time_res_ms / speed_multiplier)
    
    print(f'\nAnimation settings:')
    print(f'  Time resolution: {time_res_ms}ms')
    print(f'  Total frames: {len(time_points)}')
    print(f'  Output FPS: {actual_fps}')
    print(f'  Playback speed: {speed_multiplier}x')
    print(f'  Video duration: {len(time_points)/actual_fps:.0f}s ({len(time_points)/actual_fps/60:.1f} min)')
    
    # Get track shape
    print('\nExtracting track shape...')
    fastest = session.laps.pick_fastest()
    tel = fastest.get_telemetry()
    track_x, track_y = tel['X'].values, tel['Y'].values
    
    # Interpolate driver positions
    print('Interpolating driver positions...')
    driver_data = {}
    for driver_num in session.drivers:
        pos = session.pos_data.get(driver_num)
        if pos is None or pos.empty:
            continue
        pos = pos.dropna(subset=['SessionTime', 'X', 'Y'])
        pos = pos[(pos['X'] != 0) | (pos['Y'] != 0)]
        pos = pos[(pos['SessionTime'] >= race_start) & (pos['SessionTime'] <= race_end)]
        if len(pos) < 10:
            continue
        
        pos_times = pos['SessionTime'].dt.total_seconds().values
        idx = np.argsort(pos_times)
        x_interp = np.interp(time_seconds, pos_times[idx], pos['X'].values[idx], 
                            left=np.nan, right=np.nan)
        y_interp = np.interp(time_seconds, pos_times[idx], pos['Y'].values[idx], 
                            left=np.nan, right=np.nan)
        driver_data[driver_num] = {'x': x_interp, 'y': y_interp}
    
    print(f'Got position data for {len(driver_data)} drivers')
    
    # Compute positions and gaps
    print('Computing race positions and gaps...')
    driver_laps_map = {}
    for driver in laps['Driver'].unique():
        dlaps = laps[laps['Driver'] == driver].sort_values('LapNumber')
        if not dlaps.empty:
            driver_laps_map[driver] = dlaps
    
    positions_data = []
    update_interval = max(1, int(200 / time_res_ms))  # Update positions every ~200ms
    
    for i, t in enumerate(time_seconds):
        if i % update_interval == 0 or i == len(time_seconds) - 1:
            driver_status = []
            for driver, dlaps in driver_laps_map.items():
                current_lap = 0
                lap_progress = 0
                for _, lap in dlaps.iterrows():
                    lap_end = lap.get('Time')
                    if pd.notna(lap_end):
                        lap_end_s = lap_end.total_seconds()
                        if t <= lap_end_s:
                            current_lap = lap.get('LapNumber', 0)
                            lap_start = lap.get('LapStartTime')
                            if pd.notna(lap_start):
                                lap_start_s = lap_start.total_seconds()
                                lap_dur = lap_end_s - lap_start_s
                                if lap_dur > 0:
                                    lap_progress = (t - lap_start_s) / lap_dur
                            break
                        else:
                            current_lap = lap.get('LapNumber', 0)
                driver_status.append({
                    'driver': driver, 
                    'lap': current_lap, 
                    'progress': current_lap + lap_progress - 1
                })
            
            driver_status.sort(key=lambda x: -x['progress'])
            pos_info = {}
            for pos, ds in enumerate(driver_status):
                gap = 0
                if pos > 0:
                    gap = (driver_status[pos-1]['progress'] - ds['progress']) * 90
                pos_info[ds['driver']] = {'position': pos + 1, 'lap': ds['lap'], 'gap': gap}
            positions_data.append(pos_info)
        else:
            positions_data.append(positions_data[-1] if positions_data else {})
    
    # Create figure
    print('\nCreating animation figure...')
    fig = plt.figure(figsize=(16, 9), facecolor='#1a1a1a')
    ax_track = fig.add_axes([0.02, 0.08, 0.68, 0.88], facecolor='#1a1a1a')
    ax_board = fig.add_axes([0.72, 0.08, 0.26, 0.88], facecolor='#1a1a1a')
    
    # Draw track
    ax_track.plot(track_x, track_y, color='#2a2a2a', linewidth=16, 
                  solid_capstyle='round', zorder=1)
    ax_track.plot(track_x, track_y, color='#404040', linewidth=8, 
                  solid_capstyle='round', zorder=2)
    ax_track.set_aspect('equal')
    ax_track.set_xlim(track_x.min() - 500, track_x.max() + 500)
    ax_track.set_ylim(track_y.min() - 500, track_y.max() + 500)
    ax_track.axis('off')
    ax_track.set_title(f'{event_name} {year}', fontsize=14, fontweight='bold', 
                       color='white', pad=5)
    
    # Info text
    info_text = ax_track.text(0.02, 0.98, '', transform=ax_track.transAxes, 
                              fontsize=12, color='white', fontfamily='monospace',
                              ha='left', va='top',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='#222', alpha=0.95))
    
    # Leaderboard setup
    ax_board.set_xlim(0, 1)
    ax_board.set_ylim(0, 1)
    ax_board.axis('off')
    ax_board.text(0.5, 0.98, 'LEADERBOARD', fontsize=11, fontweight='bold', 
                  color='white', ha='center', va='top')
    
    board_entries = {}
    for i in range(20):
        y = 0.90 - i * 0.042
        pos_txt = ax_board.text(0.08, y, str(i+1), fontsize=9, color='white', 
                                ha='center', va='center', fontweight='bold')
        color_bar = ax_board.add_patch(
            FancyBboxPatch((0.12, y-0.015), 0.04, 0.028, boxstyle='round,pad=0.01',
                          facecolor='#444', transform=ax_board.transAxes))
        abbr_txt = ax_board.text(0.22, y, '---', fontsize=9, color='white', 
                                 ha='left', va='center', fontfamily='monospace', 
                                 fontweight='bold')
        gap_txt = ax_board.text(0.95, y, '', fontsize=8, color='#aaa', 
                                ha='right', va='center', fontfamily='monospace')
        board_entries[i] = {'pos': pos_txt, 'bar': color_bar, 'abbr': abbr_txt, 'gap': gap_txt}
    
    # Car markers
    car_markers = {}
    car_labels = {}
    abbr_to_num = {info['abbreviation']: num for num, info in driver_info.items()}
    
    for num in driver_data.keys():
        if num not in driver_info:
            continue
        color = driver_info[num]['color']
        m, = ax_track.plot([], [], 'o', markersize=10, color=color, 
                           markeredgecolor='white', markeredgewidth=1, zorder=100)
        car_markers[num] = m
        l = ax_track.text(0, 0, str(num), fontsize=6, fontweight='bold', 
                          color='white', ha='center', va='center', zorder=101,
                          bbox=dict(boxstyle='round,pad=0.12', facecolor=color, 
                                   edgecolor='white', linewidth=0.5, alpha=0.9))
        l.set_visible(False)
        car_labels[num] = l
    
    total_laps = int(laps['LapNumber'].max())
    
    def init():
        for m in car_markers.values():
            m.set_data([], [])
        for l in car_labels.values():
            l.set_visible(False)
        return list(car_markers.values())
    
    def update(frame):
        if frame >= len(time_points) or frame >= len(positions_data):
            return list(car_markers.values())
        
        t_s = time_seconds[frame]
        race_time = t_s - race_start_s
        mins, secs = int(race_time // 60), int(race_time % 60)
        
        pos_info = positions_data[frame]
        leader_lap = 1
        for d, p in pos_info.items():
            if p['position'] == 1:
                leader_lap = p['lap']
                break
        
        info_text.set_text(f'Lap {int(leader_lap)}/{total_laps}\n{mins:02d}:{secs:02d}')
        
        for num in driver_data.keys():
            if num not in driver_info:
                continue
            x, y = driver_data[num]['x'][frame], driver_data[num]['y'][frame]
            if np.isnan(x) or np.isnan(y):
                car_markers[num].set_data([], [])
                car_labels[num].set_visible(False)
            else:
                car_markers[num].set_data([x], [y])
                car_labels[num].set_position((x, y + 120))
                car_labels[num].set_visible(True)
        
        sorted_drivers = sorted(pos_info.items(), key=lambda x: x[1]['position'])
        for i, (abbr, pdata) in enumerate(sorted_drivers[:20]):
            entry = board_entries[i]
            num = abbr_to_num.get(abbr, abbr)
            if num in driver_info:
                entry['bar'].set_facecolor(driver_info[num]['color'])
                entry['abbr'].set_text(driver_info[num]['abbreviation'])
            if pdata['position'] == 1:
                entry['gap'].set_text('LEADER')
                entry['gap'].set_color('#0f0')
            else:
                g = pdata['gap']
                entry['gap'].set_text(f'+{g:.1f}s' if g < 60 else f'+{int(g/90)}L')
                entry['gap'].set_color('#aaa')
        
        return list(car_markers.values())
    
    total_frames = len(time_points)
    
    print(f'\nRendering {total_frames} frames...')
    anim = animation.FuncAnimation(fig, update, init_func=init, 
                                   frames=total_frames, interval=1000/actual_fps, blit=False)
    
    # Generate output filename
    if output_filename is None:
        speed_str = f'_{speed_multiplier}x' if speed_multiplier != 1 else '_realtime'
        output_filename = f'race_animation_{num_laps}laps{speed_str}.mp4'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    print(f'Saving to {output_path}...')
    print('This may take several minutes for longer animations...')
    
    writer = animation.FFMpegWriter(
        fps=actual_fps, 
        bitrate=5000,
        extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'medium']
    )
    anim.save(output_path, writer=writer, dpi=120)
    
    plt.close(fig)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    print('\n' + '=' * 60)
    print('Animation complete!')
    print('=' * 60)
    print(f'Output: {output_path}')
    print(f'Size: {file_size_mb:.1f} MB')
    print(f'Duration: {total_frames/actual_fps:.0f}s ({total_frames/actual_fps/60:.1f} min)')
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Create F1 Race Animation Video')
    parser.add_argument('--year', type=int, default=2025, help='Race year')
    parser.add_argument('--event', type=str, default='Abu Dhabi Grand Prix', 
                        help='Event name')
    parser.add_argument('--laps', type=int, default=15, 
                        help='Number of laps to animate')
    parser.add_argument('--fps', type=int, default=20, 
                        help='Frames per second (higher = smoother)')
    parser.add_argument('--speed', type=float, default=1.0, 
                        help='Playback speed multiplier (1 = real-time)')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output filename')
    
    args = parser.parse_args()
    
    create_race_video(
        year=args.year,
        event_name=args.event,
        num_laps=args.laps,
        fps=args.fps,
        speed_multiplier=args.speed,
        output_filename=args.output
    )


if __name__ == '__main__':
    main()
