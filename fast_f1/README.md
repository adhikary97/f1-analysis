# FastF1 Race Analysis

This folder contains scripts to fetch and visualize Formula 1 race data using the [FastF1](https://github.com/theOehrly/Fast-F1) library.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the complete analysis:
```bash
python run_all.py
```

## Scripts

### `fetch_latest_race.py`
Fetches comprehensive data from the latest completed F1 race and exports it to CSV files.

**Data exported:**
- `session_results.csv` - Final race classification
- `laps.csv` - All lap times for all drivers
- `drivers.csv` - Driver information
- `stints.csv` - Tire stint data
- `weather.csv` - Weather conditions throughout the race
- `race_control.csv` - Flags, safety cars, penalties
- `telemetry_fastest_laps.csv` - Sampled telemetry data
- `position_data.csv` - Car position data (sampled)
- `event_info.csv` - Race event metadata

### `visualize_race.py`
Creates comprehensive visualizations from the exported CSV data.

**Plots generated:**
1. `01_race_summary.png` - Infographic with key race stats
2. `02_final_positions.png` - Bar chart of finishing positions
3. `03_lap_times.png` - Lap time comparison for top drivers
4. `04_sector_times_heatmap.png` - Sector times heatmap
5. `05_tire_strategy.png` - Tire compound timeline
6. `06_weather_evolution.png` - Weather conditions over time
7. `07_pace_distribution.png` - Violin plot of pace
8. `08_telemetry_comparison.png` - Speed/throttle/brake comparison
9. `09_lap_time_evolution.png` - Tire degradation visualization
10. `10_race_control_timeline.png` - Flags and incidents

### `run_all.py`
Master script that runs the complete pipeline.

```bash
# Run complete pipeline
python run_all.py

# Fetch data only
python run_all.py --fetch

# Create visualizations only (uses cached data)
python run_all.py --viz
```

## Race Animation

### `run_animation.py`
Easy-to-use script to generate race animations with different options.

```bash
# Create HTML5 animation (no FFmpeg required)
python run_animation.py --html

# Quick test mode (fewer frames)
python run_animation.py --html --quick

# Create static snapshots only
python run_animation.py --snapshot

# Full MP4 animation (requires FFmpeg)
python run_animation.py

# Custom options
python run_animation.py --html --speed 20 --duration 600
```

### Animation Scripts

#### `race_animation_enhanced.py`
Full-featured race animation with:
- Cars moving around the track with team colors
- Driver number labels
- Real-time speed visualization
- Leaderboard panel
- Lap counter
- MP4/GIF output (requires FFmpeg for MP4)

```bash
python race_animation_enhanced.py --speed 15 --fps 30
python race_animation_enhanced.py --snapshot-only
```

#### `race_animation_html.py`
HTML5 interactive animation that works in any web browser:
- No FFmpeg required
- Interactive playback controls
- Team-colored cars with driver numbers

```bash
python race_animation_html.py
```

#### `race_animation.py`
Basic animation script with simpler features.

### Animation Output

Animations are saved to the `animations/` folder:
- `race_animation.mp4` - Video animation
- `race_animation.html` - Interactive HTML5 animation
- `race_animation.gif` - GIF animation (fallback)
- `snapshot_*.png` - Static snapshots at different race points

## Directory Structure

```
fast_f1/
├── README.md
├── requirements.txt
├── run_all.py
├── fetch_latest_race.py
├── visualize_race.py
├── run_animation.py            # Easy animation runner
├── race_animation.py           # Basic animation script
├── race_animation_enhanced.py  # Full-featured animation
├── race_animation_html.py      # HTML5 animation
├── cache/                      # FastF1 API cache
├── data/                       # Exported CSV files
│   └── YYYY-MM-DD_RaceName/
│       ├── session_results.csv
│       ├── laps.csv
│       └── ...
├── plots/                      # Static visualizations
│   ├── 01_race_summary.png
│   └── ...
└── animations/                 # Animation outputs
    ├── race_animation.mp4
    ├── race_animation.html
    └── snapshot_*.png
```

## FastF1 Features Used

- **Session data**: Results, laps, sectors, stints
- **Telemetry**: Speed, throttle, brake, gear, RPM, DRS
- **Weather**: Temperature, humidity, wind, rainfall
- **Race control**: Flags, safety cars, penalties

## Notes

- First run may take longer as FastF1 downloads and caches data
- Telemetry data is sampled to reduce file sizes
- The cache directory speeds up subsequent analyses

### FFmpeg for Video Output

For MP4 video animations, FFmpeg is required:

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

If FFmpeg is not available, use the HTML5 animation output (`--html` flag) which works without it.

## Links

- [FastF1 Documentation](https://docs.fastf1.dev)
- [FastF1 GitHub](https://github.com/theOehrly/Fast-F1)
