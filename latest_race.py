"""
Script to fetch and display data about the latest F1 race using OpenF1 API.
"""

from datetime import datetime, timezone
from openf1_client import OpenF1Client


def find_latest_completed_race(client):
    """Find the most recent race that has actually occurred (has data)."""
    sessions = client.sessions.list()
    
    if not sessions:
        return None
    
    # Filter for race sessions
    race_sessions = [s for s in sessions if s.session_type == "Race"]
    
    if not race_sessions:
        return None
    
    now = datetime.now(timezone.utc)
    
    # Sort races by date (newest first) and find one with data
    # First, try to filter by date (races in the past)
    for race in reversed(race_sessions):
        # Check if race date is in the past
        if race.date_start:
            try:
                race_date = datetime.fromisoformat(str(race.date_start).replace('Z', '+00:00'))
                if race_date > now:
                    continue  # Skip future races
            except (ValueError, TypeError):
                pass
        
        # Verify the race has actual data by checking for lap data
        laps = client.laps.list(session_key=race.session_key)
        if laps:
            return race
    
    return None


def main():
    with OpenF1Client() as client:
        # Get the latest session (most recent race)
        print("Fetching latest completed race...\n")
        
        latest_race = find_latest_completed_race(client)
        
        if not latest_race:
            print("No completed race sessions with data found.")
            return
        
        session_key = latest_race.session_key
        
        print("=" * 60)
        print(f"🏁 LATEST RACE: {latest_race.session_name}")
        print(f"📍 Location: {latest_race.location}, {latest_race.country_name}")
        print(f"📅 Date: {latest_race.date_start}")
        print(f"🔑 Session Key: {session_key}")
        print("=" * 60)
        
        # Get drivers first for the driver map
        drivers = client.drivers.list(session_key=session_key)
        driver_map = {d.driver_number: d for d in drivers}
        
        # Get race results (final positions and times)
        print("\n🏆 RACE RESULTS:")
        print("-" * 60)
        results = client.session_result.list(session_key=session_key)
        
        if results:
            # Sort by position
            results.sort(key=lambda x: x.position if x.position else 999)
            
            for result in results[:20]:  # Top 20 finishers
                driver = driver_map.get(result.driver_number)
                driver_name = driver.name_acronym if driver else f"#{result.driver_number}"
                full_name = driver.full_name if driver else "Unknown"
                team = driver.team_name if driver else ""
                
                pos = result.position if result.position else "DNF"
                
                # Format the time/gap
                if result.position == 1:
                    time_str = f"{result.time}" if hasattr(result, 'time') and result.time else "Winner"
                else:
                    gap = getattr(result, 'gap_to_leader', None) or getattr(result, 'time', None)
                    time_str = f"+{gap}" if gap else ""
                
                # Check for DNF/DNS status
                status = getattr(result, 'status', None) or ""
                if status and status not in ["Finished", ""]:
                    time_str = status
                
                print(f"  {str(pos):>3}. {driver_name:3} | {full_name:22} | {team:20} | {time_str}")
            
            # Highlight the winner
            if results and results[0].position == 1:
                winner = driver_map.get(results[0].driver_number)
                if winner:
                    print(f"\n  🥇 WINNER: {winner.full_name} ({winner.team_name})")
        else:
            print("  Race results not available yet.")
        
        # Display drivers in this session
        print("\n👥 DRIVERS:")
        print("-" * 60)
        for driver in drivers:
            print(f"  #{driver.driver_number:2} | {driver.name_acronym:3} | {driver.full_name:25} | {driver.team_name}")
        
        # Get race results if available
        print("\n📊 RACE ANALYSIS:")
        print("-" * 60)
        
        # Get fastest laps for each driver
        print("\n⏱️ FASTEST LAPS:")
        fastest_laps = []
        for driver in drivers:
            fastest = client.laps.get_fastest_lap(
                session_key=session_key,
                driver_number=driver.driver_number,
            )
            if fastest and fastest.lap_duration:
                fastest_laps.append((driver, fastest))
        
        # Sort by lap duration
        fastest_laps.sort(key=lambda x: x[1].lap_duration)
        
        for i, (driver, lap) in enumerate(fastest_laps[:10], 1):
            print(f"  {i:2}. {driver.name_acronym:3} - {lap.lap_duration:.3f}s (Lap {lap.lap_number})")
        
        # Get pit stop counts
        print("\n🛞 PIT STOPS:")
        pit_data = []
        for driver in drivers:
            pit_count = client.pit.count(
                session_key=session_key,
                driver_number=driver.driver_number,
            )
            if pit_count > 0:
                pit_data.append((driver, pit_count))
        
        pit_data.sort(key=lambda x: x[1])
        for driver, count in pit_data:
            print(f"  {driver.name_acronym:3} - {count} stop(s)")
        
        # Get tyre strategies (for top 5 finishers if results available, else first 5 drivers)
        print("\n🔧 TYRE STRATEGIES:")
        top_drivers = []
        if results:
            sorted_results = sorted([r for r in results if r.position], key=lambda x: x.position)
            top_drivers = [driver_map.get(r.driver_number) for r in sorted_results[:5] if driver_map.get(r.driver_number)]
        if not top_drivers:
            top_drivers = drivers[:5]
        
        for driver in top_drivers:
            stints = client.stints.list(
                session_key=session_key,
                driver_number=driver.driver_number,
            )
            if stints:
                compounds = [s.compound for s in stints if s.compound]
                if compounds:
                    print(f"  {driver.name_acronym:3}: {' → '.join(compounds)}")
        
        # Get weather conditions
        print("\n🌤️ WEATHER CONDITIONS:")
        weather = client.weather.list(session_key=session_key)
        if weather:
            # Get first and last weather reading
            first_weather = weather[0]
            last_weather = weather[-1]
            
            print(f"  Start: 🌡️ Air {first_weather.air_temperature}°C | "
                  f"Track {first_weather.track_temperature}°C | "
                  f"💧 Humidity {first_weather.humidity}%")
            print(f"  End:   🌡️ Air {last_weather.air_temperature}°C | "
                  f"Track {last_weather.track_temperature}°C | "
                  f"💧 Humidity {last_weather.humidity}%")
            
            # Check for rain
            rain_readings = [w for w in weather if w.rainfall]
            if rain_readings:
                print(f"  🌧️ Rain detected during the race!")
            else:
                print(f"  ☀️ Dry conditions throughout")
        
        # Get any race control messages (flags, incidents)
        print("\n🚩 RACE CONTROL HIGHLIGHTS:")
        race_control = client.race_control.list(session_key=session_key)
        
        # Filter for important messages
        important_categories = ["Flag", "SafetyCar", "Drs", "Other"]
        highlights = [rc for rc in race_control if rc.category in important_categories][:10]
        
        for rc in highlights:
            print(f"  [{rc.category}] {rc.message}")
        
        print("\n" + "=" * 60)
        print("Data fetched successfully from OpenF1 API!")
        print("=" * 60)


if __name__ == "__main__":
    main()
