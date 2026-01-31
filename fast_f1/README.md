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

## Directory Structure

```
fast_f1/
├── README.md
├── requirements.txt
├── run_all.py
├── fetch_latest_race.py
├── visualize_race.py
├── cache/              # FastF1 API cache (speeds up subsequent runs)
├── data/               # Exported CSV files
│   └── YYYY-MM-DD_RaceName/
│       ├── session_results.csv
│       ├── laps.csv
│       └── ...
└── plots/              # Generated visualizations
    ├── 01_race_summary.png
    ├── 02_final_positions.png
    └── ...
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

## Links

- [FastF1 Documentation](https://docs.fastf1.dev)
- [FastF1 GitHub](https://github.com/theOehrly/Fast-F1)
