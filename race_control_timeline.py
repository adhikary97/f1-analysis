"""
Race Control Timeline Visualization using OpenF1 API.
Creates a clear timeline of race events (flags, safety cars, DRS, etc.)
"""

from datetime import datetime, timezone
from openf1_client import OpenF1Client
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os


# Color scheme for different event categories
CATEGORY_COLORS = {
    "Flag": "#FFD700",           # Gold for flags
    "SafetyCar": "#FF6B35",      # Orange for safety car
    "Drs": "#00D4AA",            # Teal for DRS
    "Other": "#8B8B8B",          # Gray for other
    "CarEvent": "#E74C3C",       # Red for car events
    "default": "#3498DB",        # Blue default
}

# Specific flag colors
FLAG_COLORS = {
    "GREEN": "#2ECC71",
    "YELLOW": "#F1C40F", 
    "DOUBLE YELLOW": "#E67E22",
    "RED": "#E74C3C",
    "BLUE": "#3498DB",
    "BLACK": "#2C3E50",
    "CHEQUERED": "#1A1A1A",
    "CLEAR": "#95A5A6",
}


def find_latest_completed_race(client):
    """Find the most recent race that has actually occurred."""
    sessions = client.sessions.list()
    
    if not sessions:
        return None
    
    race_sessions = [s for s in sessions if s.session_type == "Race"]
    
    if not race_sessions:
        return None
    
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


def get_event_color(category, message):
    """Get color based on category and message content."""
    message_upper = message.upper() if message else ""
    
    # Check for specific flag colors
    if category == "Flag":
        for flag_type, color in FLAG_COLORS.items():
            if flag_type in message_upper:
                return color
        return CATEGORY_COLORS["Flag"]
    
    return CATEGORY_COLORS.get(category, CATEGORY_COLORS["default"])


def parse_race_time(event_date, race_start):
    """Convert event timestamp to minutes from race start."""
    try:
        if isinstance(event_date, str):
            event_dt = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
        else:
            event_dt = event_date
            
        if isinstance(race_start, str):
            race_start_dt = datetime.fromisoformat(race_start.replace('Z', '+00:00'))
        else:
            race_start_dt = race_start
            
        delta = event_dt - race_start_dt
        return delta.total_seconds() / 60  # Return minutes
    except (ValueError, TypeError, AttributeError):
        return None


def create_timeline_plot(race_control_events, race_info):
    """Create a clear race control timeline visualization broken down by lap."""
    
    # Filter and prepare events - group by lap
    events_by_lap = {}
    
    for event in race_control_events:
        lap = getattr(event, "lap_number", None) or 0
        category = event.category or "Other"
        message = event.message or ""
        
        if lap not in events_by_lap:
            events_by_lap[lap] = []
        
        events_by_lap[lap].append({
            "lap": lap,
            "category": category,
            "message": message,
            "flag": getattr(event, "flag", None),
        })
    
    if not events_by_lap:
        print("No events found.")
        return None
    
    # Get max lap
    max_lap = max(events_by_lap.keys())
    
    # Define event types with better categorization
    event_types = {
        "Yellow Flag": {"keywords": ["YELLOW"], "color": "#F1C40F", "marker": "s"},
        "Green Flag": {"keywords": ["GREEN", "CLEAR"], "color": "#2ECC71", "marker": "^"},
        "Red Flag": {"keywords": ["RED FLAG"], "color": "#E74C3C", "marker": "X"},
        "Safety Car": {"keywords": ["SAFETY CAR", "VSC"], "color": "#FF6B35", "marker": "D"},
        "DRS": {"keywords": ["DRS"], "color": "#00D4AA", "marker": "o"},
        "Track Limits": {"keywords": ["TRACK LIMIT", "DELETED"], "color": "#9B59B6", "marker": "."},
        "Investigation": {"keywords": ["NOTED", "INVESTIGATION", "STEWARD"], "color": "#3498DB", "marker": "p"},
        "Incident": {"keywords": ["INCIDENT"], "color": "#E67E22", "marker": "h"},
        "Black & White": {"keywords": ["BLACK AND WHITE"], "color": "#95A5A6", "marker": "*"},
        "Chequered": {"keywords": ["CHEQUERED"], "color": "#1ABC9C", "marker": "8"},
    }
    
    def classify_event(message):
        msg_upper = message.upper()
        for event_type, config in event_types.items():
            if any(kw in msg_upper for kw in config["keywords"]):
                return event_type
        return "Other"
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), height_ratios=[2, 1])
    fig.patch.set_facecolor('#0d1117')
    
    # ========== SUBPLOT 1: Main Timeline by Lap ==========
    ax1.set_facecolor('#161b22')
    
    # Prepare data for plotting
    type_positions = {}
    all_types = list(event_types.keys()) + ["Other"]
    for i, t in enumerate(all_types):
        type_positions[t] = i
    
    # Track events for legend
    plotted_types = set()
    
    # Plot events
    for lap, events in sorted(events_by_lap.items()):
        for idx, event in enumerate(events):
            event_type = classify_event(event["message"])
            y_offset = idx * 0.15  # Offset if multiple events on same lap
            
            config = event_types.get(event_type, {"color": "#8B8B8B", "marker": "o"})
            color = config["color"]
            marker = config["marker"]
            
            y_pos = type_positions.get(event_type, len(all_types) - 1) + y_offset
            
            ax1.scatter(lap, y_pos, c=color, s=100, marker=marker, 
                       edgecolors='white', linewidths=0.5, alpha=0.85, zorder=3)
            
            plotted_types.add(event_type)
    
    # Styling for subplot 1
    ax1.set_xlim(-1, max_lap + 2)
    ax1.set_ylim(-0.5, len(all_types) - 0.5)
    
    # Y-axis labels
    ax1.set_yticks(range(len(all_types)))
    ax1.set_yticklabels(all_types, fontsize=10, color='white')
    
    # X-axis - show every 5 laps
    lap_ticks = list(range(0, max_lap + 1, 5))
    if max_lap not in lap_ticks:
        lap_ticks.append(max_lap)
    ax1.set_xticks(lap_ticks)
    ax1.set_xticklabels([f"Lap {l}" for l in lap_ticks], fontsize=9, color='white')
    
    ax1.set_xlabel('Lap Number', fontsize=12, color='white', fontweight='bold')
    ax1.set_ylabel('Event Type', fontsize=12, color='white', fontweight='bold')
    
    race_name = f"{race_info.location}, {race_info.country_name}" if race_info.location else "Race"
    ax1.set_title(f"Race Control Timeline by Lap\n{race_name}", 
                 fontsize=16, color='white', fontweight='bold', pad=15)
    
    # Grid
    ax1.grid(True, axis='x', alpha=0.2, color='white', linestyle='-', linewidth=0.5)
    ax1.grid(True, axis='y', alpha=0.1, color='white', linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # Add race phase shading
    ax1.axvspan(-1, 0, alpha=0.15, color='#2ECC71', label='Pre-Race')
    ax1.axvspan(max_lap, max_lap + 2, alpha=0.15, color='#E74C3C', label='Post-Race')
    
    # Spines
    for spine in ax1.spines.values():
        spine.set_color('#30363d')
    ax1.tick_params(colors='white', which='both')
    
    # Legend
    legend_handles = []
    for event_type in all_types:
        if event_type in plotted_types:
            config = event_types.get(event_type, {"color": "#8B8B8B", "marker": "o"})
            handle = Line2D([0], [0], marker=config["marker"], color='w', 
                          markerfacecolor=config["color"], markersize=8, 
                          label=event_type, linestyle='None')
            legend_handles.append(handle)
    
    ax1.legend(handles=legend_handles, loc='upper right', ncol=2,
              facecolor='#21262d', edgecolor='#30363d', labelcolor='white',
              fontsize=9, framealpha=0.9)
    
    # ========== SUBPLOT 2: Event Count by Lap ==========
    ax2.set_facecolor('#161b22')
    
    # Count events per lap
    lap_counts = {lap: len(events) for lap, events in events_by_lap.items()}
    laps = sorted(lap_counts.keys())
    counts = [lap_counts[l] for l in laps]
    
    # Color bars by intensity
    colors = ['#238636' if c <= 2 else '#d29922' if c <= 5 else '#da3633' for c in counts]
    
    bars = ax2.bar(laps, counts, color=colors, edgecolor='white', linewidth=0.3, alpha=0.8)
    
    ax2.set_xlim(-1, max_lap + 2)
    ax2.set_xlabel('Lap Number', fontsize=12, color='white', fontweight='bold')
    ax2.set_ylabel('Event Count', fontsize=12, color='white', fontweight='bold')
    ax2.set_title('Events per Lap', fontsize=14, color='white', fontweight='bold', pad=10)
    
    # X-axis ticks
    ax2.set_xticks(lap_ticks)
    ax2.set_xticklabels([str(l) for l in lap_ticks], fontsize=9, color='white')
    
    ax2.tick_params(colors='white', which='both')
    for spine in ax2.spines.values():
        spine.set_color('#30363d')
    
    ax2.grid(True, axis='y', alpha=0.2, color='white', linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)
    
    # Add legend for bar colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#238636', edgecolor='white', label='1-2 events'),
        Patch(facecolor='#d29922', edgecolor='white', label='3-5 events'),
        Patch(facecolor='#da3633', edgecolor='white', label='6+ events'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', 
              facecolor='#21262d', edgecolor='#30363d', labelcolor='white', fontsize=9)
    
    plt.tight_layout()
    
    return fig


def create_detailed_event_list(race_control_events, race_info):
    """Create a detailed list view of race control events."""
    
    race_start = race_info.date_start
    events_with_time = []
    
    for event in race_control_events:
        time_mins = parse_race_time(event.date, race_start)
        if time_mins is not None:
            events_with_time.append({
                "time": time_mins,
                "category": event.category or "Other",
                "message": event.message or "",
                "lap_number": getattr(event, "lap_number", None),
            })
    
    events_with_time.sort(key=lambda x: x["time"])
    
    # Create figure with event list
    n_events = len(events_with_time)
    fig_height = max(8, min(20, n_events * 0.3))
    
    fig, ax = plt.subplots(figsize=(14, fig_height))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    
    # Hide axes
    ax.axis('off')
    
    # Title
    race_name = f"{race_info.location}, {race_info.country_name}" if race_info.location else "Race"
    fig.suptitle(f"Race Control Events\n{race_name}", 
                fontsize=16, color='white', fontweight='bold', y=0.98)
    
    # Create table-like display
    y_start = 0.92
    y_step = 0.8 / max(n_events, 1)
    
    for i, event in enumerate(events_with_time[:50]):  # Limit to 50 events
        y_pos = y_start - (i * y_step)
        
        color = get_event_color(event["category"], event["message"])
        
        # Time column
        time_str = f"+{event['time']:.1f} min" if event['time'] >= 0 else f"{event['time']:.1f} min"
        ax.text(0.02, y_pos, time_str, transform=ax.transAxes, 
               fontsize=9, color='#ECF0F1', family='monospace', va='center')
        
        # Lap column
        lap_str = f"Lap {event['lap_number']}" if event['lap_number'] else ""
        ax.text(0.15, y_pos, lap_str, transform=ax.transAxes,
               fontsize=9, color='#BDC3C7', va='center')
        
        # Category badge
        ax.text(0.25, y_pos, f"[{event['category']}]", transform=ax.transAxes,
               fontsize=9, color=color, fontweight='bold', va='center')
        
        # Message
        message = event['message'][:80] + "..." if len(event['message']) > 80 else event['message']
        ax.text(0.38, y_pos, message, transform=ax.transAxes,
               fontsize=9, color='white', va='center')
        
        # Add separator line using plot instead of axhline
        ax.plot([0.02, 0.98], [y_pos - y_step/2, y_pos - y_step/2],
               color='white', alpha=0.1, linewidth=0.5, transform=ax.transAxes)
    
    plt.tight_layout()
    
    return fig


def main():
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    print("Fetching race control data from OpenF1 API...")
    
    with OpenF1Client() as client:
        # Find latest completed race
        latest_race = find_latest_completed_race(client)
        
        if not latest_race:
            print("No completed race found.")
            return
        
        print(f"\nRace: {latest_race.location}, {latest_race.country_name}")
        print(f"Date: {latest_race.date_start}")
        print(f"Session Key: {latest_race.session_key}")
        
        # Get race control events
        race_control = client.race_control.list(session_key=latest_race.session_key)
        
        if not race_control:
            print("No race control data available.")
            return
        
        print(f"\nFound {len(race_control)} race control events")
        
        # Print category breakdown
        categories = {}
        for event in race_control:
            cat = event.category or "Other"
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\nEvent breakdown:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")
        
        # Create timeline plot
        print("\nGenerating timeline plot...")
        fig1 = create_timeline_plot(race_control, latest_race)
        if fig1:
            fig1.savefig("plots/race_control_timeline.png", dpi=150, 
                        facecolor=fig1.get_facecolor(), edgecolor='none',
                        bbox_inches='tight')
            print("Saved: plots/race_control_timeline.png")
            plt.close(fig1)
        
        # Create detailed event list
        print("Generating event list...")
        fig2 = create_detailed_event_list(race_control, latest_race)
        if fig2:
            fig2.savefig("plots/race_control_events.png", dpi=150,
                        facecolor=fig2.get_facecolor(), edgecolor='none',
                        bbox_inches='tight')
            print("Saved: plots/race_control_events.png")
            plt.close(fig2)
        
        print("\n✅ Done! Check the plots/ directory for the visualizations.")


if __name__ == "__main__":
    main()
