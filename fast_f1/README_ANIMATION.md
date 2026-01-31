# F1 Race Animation Video Generator

Create smooth, real-time race animations showing all cars moving around the track with team colors, driver labels, and a live leaderboard.

## Features

- **All 20 cars** moving around the actual track layout
- **Team colors** for each car (Red Bull blue, Ferrari red, McLaren orange, etc.)
- **Driver numbers** labeled on each car
- **Live leaderboard** showing positions and gaps to car ahead
- **Lap counter** and race timer
- **Real-time or accelerated** playback
- **Smooth animation** at configurable FPS

## Requirements

### Python Packages

```bash
pip install fastf1 matplotlib numpy pandas
```

### FFmpeg (for video output)

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

## Quick Start

```bash
cd /Users/paras.adhikary/Downloads/f1_analysis/fast_f1

# Create a 15-lap animation at real-time speed
python create_race_video.py

# Output: animations/race_animation_15laps_realtime.mp4
```

## Usage

### Basic Commands

```bash
# Default: 15 laps, real-time speed, 20 FPS
python create_race_video.py

# Specify number of laps
python create_race_video.py --laps 10      # First 10 laps
python create_race_video.py --laps 58      # Full race (all 58 laps)

# Adjust playback speed
python create_race_video.py --speed 2      # 2x speed (half the video length)
python create_race_video.py --speed 5      # 5x speed
python create_race_video.py --speed 10     # 10x speed

# Adjust quality (FPS)
python create_race_video.py --fps 30       # Smoother animation
python create_race_video.py --fps 10       # Smaller file size

# Custom output filename
python create_race_video.py --output my_race.mp4
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--year` | 2025 | Race year |
| `--event` | "Abu Dhabi Grand Prix" | Grand Prix name |
| `--laps` | 15 | Number of laps to animate |
| `--fps` | 20 | Frames per second |
| `--speed` | 1.0 | Playback speed multiplier |
| `--output` | auto | Output filename |

### Examples

```bash
# Full race at 10x speed (good for overview)
python create_race_video.py --laps 58 --speed 10

# First 5 laps in high quality
python create_race_video.py --laps 5 --fps 30

# Quick preview (3 laps at 5x speed)
python create_race_video.py --laps 3 --speed 5 --fps 15
```

## Output

Videos are saved to the `animations/` folder:

```
animations/
├── race_animation_15laps_realtime.mp4    # 15 laps at real-time
├── race_animation_58laps_10x.mp4         # Full race at 10x speed
└── race_animation_smooth.mp4             # Custom filename
```

### File Sizes (approximate)

| Laps | Speed | FPS | Duration | File Size |
|------|-------|-----|----------|-----------|
| 15 | 1x | 20 | 22 min | ~180 MB |
| 15 | 5x | 20 | 4.5 min | ~40 MB |
| 58 | 10x | 20 | 8.7 min | ~80 MB |
| 58 | 30x | 20 | 2.9 min | ~30 MB |

## How It Works

1. **Data Loading**: Uses FastF1 to load telemetry and position data from the official F1 timing system
2. **Track Extraction**: Gets the track shape from the fastest lap's telemetry
3. **Position Interpolation**: Interpolates car positions at regular time intervals for smooth animation
4. **Gap Calculation**: Computes race positions and gaps based on lap timing data
5. **Rendering**: Creates matplotlib animation frames and encodes to MP4 with FFmpeg

## Customization

### Changing the Race

Edit the `--year` and `--event` parameters:

```bash
# 2024 Monaco Grand Prix
python create_race_video.py --year 2024 --event "Monaco Grand Prix"

# 2025 British Grand Prix  
python create_race_video.py --year 2025 --event "British Grand Prix"
```

### Modifying Team Colors

Edit the `TEAM_COLORS` dictionary in `create_race_video.py`:

```python
TEAM_COLORS = {
    'Red Bull Racing': '#3671C6',
    'McLaren': '#FF8000',
    'Ferrari': '#E8002D',
    # ... add or modify colors
}
```

## Troubleshooting

### "ffmpeg not found"
Install FFmpeg using the commands above for your OS.

### Animation is choppy
Increase the FPS: `--fps 30`

### File is too large
- Reduce laps: `--laps 10`
- Increase speed: `--speed 5`
- Reduce FPS: `--fps 15`

### Takes too long to render
- Reduce laps
- Increase speed multiplier
- Use lower FPS

### No data for race
Make sure the race has occurred and FastF1 has data available. The first run will download and cache the data.

## Files

| File | Description |
|------|-------------|
| `create_race_video.py` | Main script to generate MP4 videos |
| `race_animation_html.py` | Alternative HTML5 animation generator |
| `run_animation.py` | Simple runner script with presets |
| `animations/` | Output folder for generated videos |
| `cache/` | FastF1 data cache (speeds up subsequent runs) |

## License

This project uses data from the official F1 timing system via FastF1.
