#!/usr/bin/env python3
"""
Simple runner script for F1 race animations.

Usage:
    python run_animation.py              # Full animation (may take a while)
    python run_animation.py --quick      # Quick test (fewer frames)
    python run_animation.py --snapshot   # Just create snapshots
    python run_animation.py --html       # Create HTML5 animation (no FFmpeg needed)
"""

import sys
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Generate F1 Race Animation')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (fewer frames)')
    parser.add_argument('--snapshot', action='store_true',
                       help='Only create static snapshots')
    parser.add_argument('--html', action='store_true',
                       help='Create HTML5 animation (no FFmpeg required)')
    parser.add_argument('--speed', type=int, default=15,
                       help='Playback speed multiplier (default: 15)')
    parser.add_argument('--duration', type=int, default=300,
                       help='Duration in seconds to animate (default: 300 = 5 min)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("F1 Race Animation Generator")
    print("=" * 70)
    
    if args.html:
        print("\nUsing HTML5 output (no FFmpeg required)")
        from race_animation_html import (
            load_race_data, get_driver_info, 
            prepare_animation_data, create_html_animation
        )
        
        session = load_race_data()
        driver_info = get_driver_info(session)
        
        # Calculate number of laps
        num_laps = 5 if args.quick else max(10, args.duration // 90)
        max_frames = 300 if args.quick else 1200
        
        print(f"Animating {num_laps} laps...")
        
        anim_data = prepare_animation_data(
            session, 
            time_resolution_ms=250,
            num_laps=num_laps
        )
        
        create_html_animation(
            session, anim_data, driver_info,
            speed_multiplier=args.speed,
            fps=20,
            max_frames=max_frames,
            output_file='race_animation.html'
        )
        
    elif args.snapshot:
        print("\nCreating static snapshots only...")
        from race_animation_enhanced import RaceAnimator
        
        animator = RaceAnimator()
        animator.load_session()
        animator.extract_driver_info()
        animator.extract_track_shape()
        animator.prepare_position_data(time_resolution_ms=200)
        
        # Create snapshots at different race points
        animator.create_snapshot(time_seconds=3650, output_file='snapshot_lap1.png')
        animator.create_snapshot(time_seconds=3800, output_file='snapshot_lap3.png')
        animator.create_snapshot(time_seconds=4200, output_file='snapshot_lap10.png')
        animator.create_snapshot(time_seconds=5000, output_file='snapshot_mid_race.png')
        
    else:
        print("\nCreating full video animation...")
        print("Note: Requires FFmpeg for MP4 output. Will fall back to GIF if not available.")
        
        from race_animation_enhanced import RaceAnimator
        
        animator = RaceAnimator()
        animator.load_session()
        animator.extract_driver_info()
        animator.extract_track_shape()
        animator.prepare_position_data(time_resolution_ms=150)
        animator.calculate_positions_and_gaps()
        
        max_frames = 300 if args.quick else None
        
        animator.create_animation(
            output_file='race_animation.mp4',
            speed_multiplier=args.speed,
            fps=30,
            max_frames=max_frames
        )
        
        # Also create a snapshot
        animator.create_snapshot(time_seconds=4000, output_file='race_snapshot.png')
    
    print("\n" + "=" * 70)
    print("Done! Check the 'animations' folder for output files.")
    print("=" * 70)


if __name__ == "__main__":
    main()
