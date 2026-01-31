"""
Script to fetch data from all OpenF1 API endpoints for the latest race
and export each to CSV files.
"""

import os
from datetime import datetime
from openf1_client import OpenF1Client


def create_output_dir(race_name: str, date_str: str) -> str:
    """Create a directory for the race data exports."""
    # Clean the race name for use as directory name
    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in race_name)
    safe_name = safe_name.replace(" ", "_")
    dir_name = f"race_data_{safe_name}_{date_str}"
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    return dir_name


def write_csv(filename: str, csv_data: str, output_dir: str) -> bool:
    """Write CSV data to a file."""
    if not csv_data or csv_data.strip() == "":
        print(f"  ⚠️  No data for {filename}")
        return False
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(csv_data)
    
    # Count rows (excluding header)
    row_count = len(csv_data.strip().split("\n")) - 1
    print(f"  ✅ {filename}: {row_count} rows")
    return True


def export_all_endpoints(client: OpenF1Client, session_key: int, output_dir: str) -> dict:
    """Export data from all endpoints for a given session."""
    results = {}
    
    # Dictionary of all endpoints and their export configurations
    # Format: (endpoint_name, filename, has_driver_filter)
    endpoints = [
        ("drivers", "drivers.csv"),
        ("laps", "laps.csv"),
        ("stints", "stints.csv"),
        ("pit", "pit_stops.csv"),
        ("position", "positions.csv"),
        ("intervals", "intervals.csv"),
        ("race_control", "race_control.csv"),
        ("weather", "weather.csv"),
        ("team_radio", "team_radio.csv"),
        ("session_result", "session_results.csv"),
        ("starting_grid", "starting_grid.csv"),
        ("overtakes", "overtakes.csv"),
    ]
    
    print("\n📥 Exporting session-level data...")
    print("-" * 50)
    
    for endpoint_name, filename in endpoints:
        try:
            endpoint = getattr(client, endpoint_name)
            csv_data = endpoint.list_csv(session_key=session_key)
            success = write_csv(filename, csv_data, output_dir)
            results[endpoint_name] = success
        except Exception as e:
            print(f"  ❌ {filename}: Error - {e}")
            results[endpoint_name] = False
    
    # Export high-frequency telemetry data (car_data and location)
    # These can be very large, so we'll export them with a warning
    print("\n📥 Exporting high-frequency telemetry data...")
    print("-" * 50)
    print("  ⚠️  Note: Telemetry data can be very large (car_data ~3.7Hz, location ~3.7Hz)")
    
    # Get list of drivers first to sample telemetry
    try:
        drivers = client.drivers.list(session_key=session_key)
        if drivers:
            # Export car_data for all drivers (this can be very large)
            try:
                print("  📡 Fetching car_data (this may take a while)...")
                csv_data = client.car_data.list_csv(session_key=session_key)
                success = write_csv("car_data.csv", csv_data, output_dir)
                results["car_data"] = success
            except Exception as e:
                print(f"  ❌ car_data.csv: Error - {e}")
                results["car_data"] = False
            
            # Export location data for all drivers
            try:
                print("  📍 Fetching location data (this may take a while)...")
                csv_data = client.location.list_csv(session_key=session_key)
                success = write_csv("location.csv", csv_data, output_dir)
                results["location"] = success
            except Exception as e:
                print(f"  ❌ location.csv: Error - {e}")
                results["location"] = False
    except Exception as e:
        print(f"  ❌ Error fetching drivers: {e}")
        results["car_data"] = False
        results["location"] = False
    
    return results


def export_meeting_data(client: OpenF1Client, meeting_key: int, output_dir: str) -> bool:
    """Export meeting (Grand Prix) level data."""
    print("\n📥 Exporting meeting-level data...")
    print("-" * 50)
    
    try:
        csv_data = client.meetings.list_csv(meeting_key=meeting_key)
        success = write_csv("meeting.csv", csv_data, output_dir)
        return success
    except Exception as e:
        print(f"  ❌ meeting.csv: Error - {e}")
        return False


def export_sessions_for_meeting(client: OpenF1Client, meeting_key: int, output_dir: str) -> bool:
    """Export all sessions for a meeting (FP1, FP2, FP3, Quali, Race, etc.)."""
    try:
        csv_data = client.sessions.list_csv(meeting_key=meeting_key)
        success = write_csv("sessions.csv", csv_data, output_dir)
        return success
    except Exception as e:
        print(f"  ❌ sessions.csv: Error - {e}")
        return False


def main():
    print("=" * 60)
    print("🏎️  OpenF1 Race Data Exporter")
    print("=" * 60)
    
    with OpenF1Client() as client:
        # Get the latest race session with actual data
        print("\n🔍 Finding the latest race session with data...")
        
        sessions = client.sessions.list()
        
        if not sessions:
            print("❌ No sessions found.")
            return
        
        # Find race sessions
        race_sessions = [s for s in sessions if s.session_type == "Race"]
        
        if not race_sessions:
            print("❌ No race sessions found.")
            return
        
        # Find the most recent race that has actual data
        # Start from the end and work backwards
        latest_race = None
        for race in reversed(race_sessions):
            try:
                # Check if this race has driver data
                drivers = client.drivers.list(session_key=race.session_key)
                if drivers:
                    latest_race = race
                    break
            except Exception:
                continue
        
        if not latest_race:
            print("❌ No race sessions with data found.")
            return
        
        session_key = latest_race.session_key
        meeting_key = latest_race.meeting_key
        
        # Format date for directory name
        date_str = ""
        if latest_race.date_start:
            try:
                dt = datetime.fromisoformat(str(latest_race.date_start).replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d")
            except Exception:
                date_str = "unknown_date"
        
        print("\n" + "=" * 60)
        print(f"🏁 LATEST RACE: {latest_race.session_name}")
        print(f"📍 Location: {latest_race.location}, {latest_race.country_name}")
        print(f"📅 Date: {latest_race.date_start}")
        print(f"🔑 Session Key: {session_key}")
        print(f"🔑 Meeting Key: {meeting_key}")
        print("=" * 60)
        
        # Create output directory
        race_name = latest_race.session_name or f"Race_{session_key}"
        output_dir = create_output_dir(race_name, date_str)
        print(f"\n📁 Output directory: {output_dir}/")
        
        # Export meeting data
        export_meeting_data(client, meeting_key, output_dir)
        
        # Export all sessions for this meeting
        export_sessions_for_meeting(client, meeting_key, output_dir)
        
        # Export all endpoint data for the race session
        results = export_all_endpoints(client, session_key, output_dir)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 EXPORT SUMMARY")
        print("=" * 60)
        
        successful = sum(1 for v in results.values() if v)
        total = len(results)
        
        print(f"\n✅ Successfully exported: {successful}/{total} endpoints")
        
        if successful < total:
            failed = [k for k, v in results.items() if not v]
            print(f"⚠️  Failed/Empty endpoints: {', '.join(failed)}")
        
        print(f"\n📁 All files saved to: {output_dir}/")
        print("\n🏁 Export complete!")
        print("=" * 60)
        
        # List all exported files
        print("\n📄 Exported files:")
        for filename in sorted(os.listdir(output_dir)):
            filepath = os.path.join(output_dir, filename)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"   - {filename} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
