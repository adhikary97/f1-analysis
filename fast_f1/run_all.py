"""
Main script to fetch latest race data and create visualizations.

Usage:
    python run_all.py           # Fetch data and create plots
    python run_all.py --fetch   # Only fetch data
    python run_all.py --viz     # Only create visualizations (uses cached data)
"""

import sys
import os

def main():
    args = sys.argv[1:]
    
    fetch_only = '--fetch' in args
    viz_only = '--viz' in args
    
    if viz_only:
        print("📊 Creating visualizations only (using cached data)...\n")
        from visualize_race import main as viz_main
        viz_main()
    elif fetch_only:
        print("📥 Fetching data only...\n")
        from fetch_latest_race import main as fetch_main
        fetch_main()
    else:
        print("🏎️  FastF1 Complete Analysis Pipeline")
        print("=" * 60)
        print("\nStep 1: Fetching latest race data...")
        print("-" * 60)
        
        from fetch_latest_race import main as fetch_main
        data_dir = fetch_main()
        
        if data_dir:
            print("\n" + "=" * 60)
            print("\nStep 2: Creating visualizations...")
            print("-" * 60)
            
            from visualize_race import main as viz_main
            viz_main()
        else:
            print("\n❌ Data fetch failed. Skipping visualizations.")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
