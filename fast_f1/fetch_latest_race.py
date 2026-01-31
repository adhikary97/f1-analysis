"""
Fetch data from the latest F1 race using FastF1 and export to CSV files.

FastF1 provides comprehensive F1 data including:
- Session results and lap times
- Telemetry data (speed, throttle, brake, gear, RPM, DRS)
- Weather data
- Tire/stint information
- Car positions and sector times
"""

import os
import warnings
from datetime import datetime

import fastf1
import pandas as pd

# Suppress some FastF1 warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Set up cache directory for FastF1
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Enable caching to speed up subsequent runs
fastf1.Cache.enable_cache(CACHE_DIR)


def get_latest_race_event():
    """Find the most recent completed race event."""
    current_year = datetime.now().year
    
    # Get the current season schedule
    schedule = fastf1.get_event_schedule(current_year)
    
    # Filter for events that have already occurred
    now = pd.Timestamp.now(tz='UTC')
    
    # Handle timezone-naive datetime columns by converting to tz-naive comparison
    event_dates = pd.to_datetime(schedule['EventDate'])
    if event_dates.dt.tz is None:
        # Convert now to tz-naive for comparison
        now_naive = now.tz_localize(None)
        past_events = schedule[event_dates < now_naive]
    else:
        past_events = schedule[event_dates < now]
    
    if past_events.empty:
        # Try previous year if no races this year yet
        schedule = fastf1.get_event_schedule(current_year - 1)
        event_dates = pd.to_datetime(schedule['EventDate'])
        if event_dates.dt.tz is None:
            now_naive = now.tz_localize(None)
            past_events = schedule[event_dates < now_naive]
        else:
            past_events = schedule[event_dates < now]
    
    if past_events.empty:
        raise ValueError("No completed race events found")
    
    # Get the most recent event
    latest_event = past_events.iloc[-1]
    return latest_event


def export_session_data(session, output_dir):
    """Export all available data from a session to CSV files."""
    results = {}
    
    print("\n📥 Exporting session data to CSV...")
    print("-" * 50)
    
    # 1. Session Results (final classification)
    try:
        if hasattr(session, 'results') and session.results is not None and not session.results.empty:
            filepath = os.path.join(output_dir, 'session_results.csv')
            session.results.to_csv(filepath, index=False)
            print(f"  ✅ session_results.csv: {len(session.results)} rows")
            results['session_results'] = True
        else:
            print("  ⚠️  session_results.csv: No data available")
            results['session_results'] = False
    except Exception as e:
        print(f"  ❌ session_results.csv: {e}")
        results['session_results'] = False
    
    # 2. Lap Data (all laps for all drivers)
    try:
        if hasattr(session, 'laps') and session.laps is not None and not session.laps.empty:
            laps_df = session.laps.copy()
            # Convert timedelta columns to seconds for easier analysis
            time_cols = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 
                        'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
                        'LapStartTime', 'Time', 'PitOutTime', 'PitInTime']
            for col in time_cols:
                if col in laps_df.columns:
                    laps_df[f'{col}_seconds'] = laps_df[col].dt.total_seconds()
            
            filepath = os.path.join(output_dir, 'laps.csv')
            laps_df.to_csv(filepath, index=False)
            print(f"  ✅ laps.csv: {len(laps_df)} rows")
            results['laps'] = True
        else:
            print("  ⚠️  laps.csv: No data available")
            results['laps'] = False
    except Exception as e:
        print(f"  ❌ laps.csv: {e}")
        results['laps'] = False
    
    # 3. Weather Data
    try:
        if hasattr(session, 'weather_data') and session.weather_data is not None and not session.weather_data.empty:
            filepath = os.path.join(output_dir, 'weather.csv')
            session.weather_data.to_csv(filepath, index=False)
            print(f"  ✅ weather.csv: {len(session.weather_data)} rows")
            results['weather'] = True
        else:
            print("  ⚠️  weather.csv: No data available")
            results['weather'] = False
    except Exception as e:
        print(f"  ❌ weather.csv: {e}")
        results['weather'] = False
    
    # 4. Race Control Messages (flags, penalties, etc.)
    try:
        if hasattr(session, 'race_control_messages') and session.race_control_messages is not None:
            rcm = session.race_control_messages
            if not rcm.empty:
                filepath = os.path.join(output_dir, 'race_control.csv')
                rcm.to_csv(filepath, index=False)
                print(f"  ✅ race_control.csv: {len(rcm)} rows")
                results['race_control'] = True
            else:
                print("  ⚠️  race_control.csv: No data available")
                results['race_control'] = False
        else:
            print("  ⚠️  race_control.csv: No data available")
            results['race_control'] = False
    except Exception as e:
        print(f"  ❌ race_control.csv: {e}")
        results['race_control'] = False
    
    # 5. Driver Information
    try:
        drivers_data = []
        for driver in session.drivers:
            try:
                driver_info = session.get_driver(driver)
                drivers_data.append({
                    'DriverNumber': driver,
                    'Abbreviation': driver_info.get('Abbreviation', ''),
                    'FirstName': driver_info.get('FirstName', ''),
                    'LastName': driver_info.get('LastName', ''),
                    'FullName': driver_info.get('FullName', ''),
                    'TeamName': driver_info.get('TeamName', ''),
                    'TeamColor': driver_info.get('TeamColor', ''),
                    'HeadshotUrl': driver_info.get('HeadshotUrl', ''),
                    'CountryCode': driver_info.get('CountryCode', ''),
                })
            except Exception:
                drivers_data.append({'DriverNumber': driver})
        
        if drivers_data:
            drivers_df = pd.DataFrame(drivers_data)
            filepath = os.path.join(output_dir, 'drivers.csv')
            drivers_df.to_csv(filepath, index=False)
            print(f"  ✅ drivers.csv: {len(drivers_df)} rows")
            results['drivers'] = True
        else:
            print("  ⚠️  drivers.csv: No data available")
            results['drivers'] = False
    except Exception as e:
        print(f"  ❌ drivers.csv: {e}")
        results['drivers'] = False
    
    # 6. Stint/Tire Data
    try:
        stints_data = []
        for driver in session.drivers:
            try:
                driver_laps = session.laps.pick_driver(driver)
                if hasattr(driver_laps, 'get_stints'):
                    driver_stints = driver_laps.get_stints()
                    for _, stint in driver_stints.iterrows():
                        stints_data.append({
                            'Driver': driver,
                            **stint.to_dict()
                        })
                else:
                    # Extract stint info from laps
                    if 'Stint' in driver_laps.columns and 'Compound' in driver_laps.columns:
                        for stint_num in driver_laps['Stint'].dropna().unique():
                            stint_laps = driver_laps[driver_laps['Stint'] == stint_num]
                            if not stint_laps.empty:
                                stints_data.append({
                                    'Driver': driver,
                                    'Stint': int(stint_num),
                                    'Compound': stint_laps['Compound'].iloc[0] if 'Compound' in stint_laps.columns else None,
                                    'TyreLife': stint_laps['TyreLife'].max() if 'TyreLife' in stint_laps.columns else None,
                                    'FreshTyre': stint_laps['FreshTyre'].iloc[0] if 'FreshTyre' in stint_laps.columns else None,
                                    'LapStart': stint_laps['LapNumber'].min(),
                                    'LapEnd': stint_laps['LapNumber'].max(),
                                    'NumLaps': len(stint_laps),
                                })
            except Exception as e:
                continue
        
        if stints_data:
            stints_df = pd.DataFrame(stints_data)
            filepath = os.path.join(output_dir, 'stints.csv')
            stints_df.to_csv(filepath, index=False)
            print(f"  ✅ stints.csv: {len(stints_df)} rows")
            results['stints'] = True
        else:
            print("  ⚠️  stints.csv: No data available")
            results['stints'] = False
    except Exception as e:
        print(f"  ❌ stints.csv: {e}")
        results['stints'] = False
    
    # 7. Telemetry Data (sampled - full telemetry is very large)
    try:
        telemetry_data = []
        print("  📡 Fetching telemetry data (sampling every 10th point)...")
        
        for driver in session.drivers[:5]:  # Limit to top 5 drivers to avoid huge files
            try:
                driver_laps = session.laps.pick_driver(driver)
                fastest_lap = driver_laps.pick_fastest()
                if fastest_lap is not None and hasattr(fastest_lap, 'get_telemetry'):
                    tel = fastest_lap.get_telemetry()
                    if tel is not None and not tel.empty:
                        # Sample every 10th point to reduce file size
                        tel_sampled = tel.iloc[::10].copy()
                        tel_sampled['Driver'] = driver
                        tel_sampled['LapNumber'] = fastest_lap['LapNumber'] if 'LapNumber' in fastest_lap else None
                        telemetry_data.append(tel_sampled)
            except Exception:
                continue
        
        if telemetry_data:
            telemetry_df = pd.concat(telemetry_data, ignore_index=True)
            # Convert timedelta columns
            if 'Time' in telemetry_df.columns:
                telemetry_df['Time_seconds'] = telemetry_df['Time'].dt.total_seconds()
            if 'SessionTime' in telemetry_df.columns:
                telemetry_df['SessionTime_seconds'] = telemetry_df['SessionTime'].dt.total_seconds()
            
            filepath = os.path.join(output_dir, 'telemetry_fastest_laps.csv')
            telemetry_df.to_csv(filepath, index=False)
            print(f"  ✅ telemetry_fastest_laps.csv: {len(telemetry_df)} rows (sampled)")
            results['telemetry'] = True
        else:
            print("  ⚠️  telemetry_fastest_laps.csv: No data available")
            results['telemetry'] = False
    except Exception as e:
        print(f"  ❌ telemetry_fastest_laps.csv: {e}")
        results['telemetry'] = False
    
    # 8. Position Data
    try:
        if hasattr(session, 'pos_data') and session.pos_data is not None:
            pos_data = []
            for driver, data in session.pos_data.items():
                if data is not None and not data.empty:
                    data_copy = data.copy()
                    data_copy['Driver'] = driver
                    pos_data.append(data_copy)
            
            if pos_data:
                pos_df = pd.concat(pos_data, ignore_index=True)
                # Sample to reduce file size
                pos_df_sampled = pos_df.iloc[::50]
                filepath = os.path.join(output_dir, 'position_data.csv')
                pos_df_sampled.to_csv(filepath, index=False)
                print(f"  ✅ position_data.csv: {len(pos_df_sampled)} rows (sampled)")
                results['position_data'] = True
            else:
                print("  ⚠️  position_data.csv: No data available")
                results['position_data'] = False
        else:
            print("  ⚠️  position_data.csv: No data available")
            results['position_data'] = False
    except Exception as e:
        print(f"  ❌ position_data.csv: {e}")
        results['position_data'] = False
    
    # 9. Event Schedule Info
    try:
        event_info = {
            'EventName': [session.event['EventName']],
            'EventDate': [session.event['EventDate']],
            'Country': [session.event['Country']],
            'Location': [session.event['Location']],
            'OfficialEventName': [session.event.get('OfficialEventName', '')],
            'EventFormat': [session.event.get('EventFormat', '')],
            'Session': [session.name],
            'SessionDate': [str(session.date) if hasattr(session, 'date') else ''],
        }
        event_df = pd.DataFrame(event_info)
        filepath = os.path.join(output_dir, 'event_info.csv')
        event_df.to_csv(filepath, index=False)
        print(f"  ✅ event_info.csv: 1 row")
        results['event_info'] = True
    except Exception as e:
        print(f"  ❌ event_info.csv: {e}")
        results['event_info'] = False
    
    return results


def main():
    print("=" * 60)
    print("🏎️  FastF1 Race Data Exporter")
    print("=" * 60)
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    try:
        # Get the latest race event
        print("\n🔍 Finding the latest race event...")
        event = get_latest_race_event()
        
        print("\n" + "=" * 60)
        print(f"🏁 LATEST RACE: {event['EventName']}")
        print(f"📍 Location: {event['Location']}, {event['Country']}")
        print(f"📅 Date: {event['EventDate']}")
        print("=" * 60)
        
        # Create output directory for this race
        date_str = pd.Timestamp(event['EventDate']).strftime('%Y-%m-%d')
        safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in str(event['EventName']))
        output_dir = os.path.join(DATA_DIR, f"{date_str}_{safe_name.replace(' ', '_')}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📁 Output directory: {output_dir}/")
        
        # Load the race session
        print("\n📥 Loading race session data (this may take a while on first run)...")
        session = fastf1.get_session(
            event['EventDate'].year,
            event['EventName'],
            'R'  # Race session
        )
        session.load()
        
        print(f"✅ Session loaded: {session.name}")
        print(f"   Total laps: {session.total_laps if hasattr(session, 'total_laps') else 'N/A'}")
        print(f"   Drivers: {len(session.drivers)}")
        
        # Export all data
        results = export_session_data(session, output_dir)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 EXPORT SUMMARY")
        print("=" * 60)
        
        successful = sum(1 for v in results.values() if v)
        total = len(results)
        
        print(f"\n✅ Successfully exported: {successful}/{total} data types")
        
        if successful < total:
            failed = [k for k, v in results.items() if not v]
            print(f"⚠️  Failed/Empty: {', '.join(failed)}")
        
        print(f"\n📁 All files saved to: {output_dir}/")
        
        # List all exported files
        print("\n📄 Exported files:")
        for filename in sorted(os.listdir(output_dir)):
            filepath = os.path.join(output_dir, filename)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"   - {filename} ({size_kb:.1f} KB)")
        
        print("\n🏁 Export complete!")
        print("=" * 60)
        
        return output_dir
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
