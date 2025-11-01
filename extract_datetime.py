import os
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def extract_datetime_from_filename(filename):
    """
    Extract date and time information from image filenames.
    
    Expected format: ann-img_YYYYMMDD-HHMMSS_...
    
    Args:
        filename (str): The filename to parse
        
    Returns:
        dict: A dictionary containing:
            - 'filename': original filename
            - 'date': datetime.date object
            - 'time': datetime.time object
            - 'datetime': datetime.datetime object
            - 'date_str': formatted date string (YYYY-MM-DD)
            - 'time_str': formatted time string (HH:MM:SS)
        Returns None if filename doesn't match the expected pattern
    """
    # Pattern: ann-img_YYYYMMDD-HHMMSS
    pattern = r'ann-img_(\d{8})-(\d{6})'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    date_code = match.group(1)  # YYYYMMDD
    time_code = match.group(2)  # HHMMSS
    
    try:
        # Parse date code
        year = int(date_code[0:4])
        month = int(date_code[4:6])
        day = int(date_code[6:8])
        
        # Parse time code
        hour = int(time_code[0:2])
        minute = int(time_code[2:4])
        second = int(time_code[4:6])
        
        # Create datetime object
        dt = datetime(year, month, day, hour, minute, second)
        
        return {
            'filename': filename,
            'date': dt.date(),
            'time': dt.time(),
            'datetime': dt,
            'date_str': dt.strftime('%Y-%m-%d'),
            'time_str': dt.strftime('%H:%M:%S')
        }
    except ValueError as e:
        print(f"Error parsing {filename}: {e}")
        return None


def read_images_from_folder(folder_path):
    """
    Read all image files from a folder and extract datetime information.
    
    Args:
        folder_path (str): Path to the images folder
        
    Returns:
        list: List of dictionaries containing extracted datetime info
    """
    results = []
    
    # Check if folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return results
    
    # Get all files in the folder
    files = sorted(os.listdir(folder_path))
    
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        
        # Skip if it's a directory
        if os.path.isdir(file_path):
            continue
        
        # Try to extract datetime
        datetime_info = extract_datetime_from_filename(filename)
        
        if datetime_info:
            results.append(datetime_info)
    
    return results


def print_results(datetime_list):
    """
    Print the extracted datetime information in a formatted table.
    
    Args:
        datetime_list (list): List of extracted datetime dictionaries
    """
    if not datetime_list:
        print("No images found or no valid filenames.")
        return
    
    print(f"\nFound {len(datetime_list)} images:\n")
    print(f"{'Filename':<50} {'Date':<12} {'Time':<10}")
    print("-" * 72)
    
    for item in datetime_list:
        print(f"{item['filename']:<50} {item['date_str']:<12} {item['time_str']:<10}")
    
    # Print summary statistics
    print("\n" + "=" * 72)
    print("Summary Statistics:")
    print(f"  Total images: {len(datetime_list)}")
    
    dates = [item['date'] for item in datetime_list]
    times = [item['time'] for item in datetime_list]
    
    print(f"  Date range: {min(dates)} to {max(dates)}")
    print(f"  Time range: {min(times)} to {max(times)}")


def analyze_by_weekday_and_time(datetime_list):
    """
    Analyze image distribution by weekday and time of day.
    
    Args:
        datetime_list (list): List of extracted datetime dictionaries
        
    Returns:
        dict: Analysis data with weekday distribution and hourly distribution per weekday
    """
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Initialize data structures
    weekday_counts = defaultdict(int)  # Count of images per weekday
    weekday_hours = defaultdict(lambda: defaultdict(int))  # Hour distribution per weekday
    
    for item in datetime_list:
        dt = item['datetime']
        weekday = dt.weekday()  # 0=Monday, 6=Sunday
        hour = dt.hour
        
        weekday_counts[weekday] += 1
        weekday_hours[weekday][hour] += 1
    
    return {
        'weekday_names': weekday_names,
        'weekday_counts': weekday_counts,
        'weekday_hours': weekday_hours
    }


def visualize_weekday_distribution(datetime_list, output_path=None):
    """
    Create a bar chart showing image distribution across weekdays.
    
    Args:
        datetime_list (list): List of extracted datetime dictionaries
        output_path (str): Optional path to save the figure
        
    Returns:
        matplotlib figure object
    """
    analysis = analyze_by_weekday_and_time(datetime_list)
    weekday_names = analysis['weekday_names']
    weekday_counts = analysis['weekday_counts']
    
    # Create data for all weekdays (0-6)
    counts = [weekday_counts.get(i, 0) for i in range(7)]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(7), counts, color='steelblue', edgecolor='navy', alpha=0.7)
    
    # Customize the chart
    ax.set_xlabel('Weekday', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('Surveillance Image Distribution by Weekday', fontsize=14, fontweight='bold')
    ax.set_xticks(range(7))
    ax.set_xticklabels(weekday_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Weekday distribution chart saved to '{output_path}'")
    
    return fig


def visualize_time_distribution_by_weekday(datetime_list, output_path=None):
    """
    Create heatmap showing hourly distribution for each weekday.
    
    Args:
        datetime_list (list): List of extracted datetime dictionaries
        output_path (str): Optional path to save the figure
        
    Returns:
        matplotlib figure object
    """
    analysis = analyze_by_weekday_and_time(datetime_list)
    weekday_names = analysis['weekday_names']
    weekday_hours = analysis['weekday_hours']
    
    # Create a 7x24 matrix (weekdays x hours)
    data = np.zeros((7, 24))
    
    for weekday in range(7):
        for hour in range(24):
            data[weekday][hour] = weekday_hours[weekday].get(hour, 0)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_yticks(range(7))
    ax.set_yticklabels(weekday_names)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right', fontsize=9)
    
    # Labels and title
    ax.set_xlabel('Time of Day (Hour)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weekday', fontsize=12, fontweight='bold')
    ax.set_title('Surveillance Image Distribution: Heatmap by Weekday and Hour', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Images', fontsize=11, fontweight='bold')
    
    # Add text annotations
    for i in range(7):
        for j in range(24):
            if data[i][j] > 0:
                text = ax.text(j, i, f'{int(data[i][j])}',
                             ha="center", va="center", color="black", fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to '{output_path}'")
    
    return fig


def visualize_hourly_distribution(datetime_list, output_path=None):
    """
    Create a line chart showing overall hourly distribution across all days.
    
    Args:
        datetime_list (list): List of extracted datetime dictionaries
        output_path (str): Optional path to save the figure
        
    Returns:
        matplotlib figure object
    """
    # Count images by hour
    hourly_counts = defaultdict(int)
    
    for item in datetime_list:
        hour = item['datetime'].hour
        hourly_counts[hour] += 1
    
    # Create data for all 24 hours
    hours = list(range(24))
    counts = [hourly_counts.get(h, 0) for h in hours]
    
    # Create line chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hours, counts, marker='o', linewidth=2, markersize=8, 
            color='darkgreen', markerfacecolor='lightgreen', markeredgewidth=2)
    
    # Customize the chart
    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('Overall Hourly Distribution of Surveillance Images', fontsize=14, fontweight='bold')
    ax.set_xticks(hours)
    ax.set_xticklabels([f'{h:02d}:00' for h in hours], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    for hour, count in zip(hours, counts):
        if count > 0:
            ax.text(hour, count, f'{int(count)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Hourly distribution chart saved to '{output_path}'")
    
    return fig


def print_analysis_summary(datetime_list):
    """
    Print detailed analysis summary of the surveillance image patterns.
    
    Args:
        datetime_list (list): List of extracted datetime dictionaries
    """
    analysis = analyze_by_weekday_and_time(datetime_list)
    weekday_names = analysis['weekday_names']
    weekday_counts = analysis['weekday_counts']
    weekday_hours = analysis['weekday_hours']
    
    print("\n" + "=" * 80)
    print("SURVEILLANCE IMAGE PATTERN ANALYSIS")
    print("=" * 80)
    
    # Weekday distribution
    print("\n1. DISTRIBUTION BY WEEKDAY:")
    print("-" * 80)
    for weekday in range(7):
        count = weekday_counts.get(weekday, 0)
        bar = '█' * count
        print(f"  {weekday_names[weekday]:<12} {count:>3} images  {bar}")
    
    # Busiest times per weekday
    print("\n2. PEAK HOURS BY WEEKDAY:")
    print("-" * 80)
    for weekday in range(7):
        hours = weekday_hours[weekday]
        if hours:
            peak_hour = max(hours, key=hours.get)
            peak_count = hours[peak_hour]
            hours_with_activity = [h for h in hours if hours[h] > 0]
            print(f"  {weekday_names[weekday]:<12} Peak: {peak_hour:02d}:00 ({peak_count} images) | "
                  f"Active hours: {sorted(hours_with_activity)}")
    
    # Overall statistics
    print("\n3. OVERALL STATISTICS:")
    print("-" * 80)
    hourly_counts = defaultdict(int)
    for item in datetime_list:
        hour = item['datetime'].hour
        hourly_counts[hour] += 1
    
    if hourly_counts:
        peak_hour = max(hourly_counts, key=hourly_counts.get)
        peak_count = hourly_counts[peak_hour]
        hours_with_activity = sorted([h for h in hourly_counts if hourly_counts[h] > 0])
        
        print(f"  Total images: {len(datetime_list)}")
        print(f"  Busiest hour (overall): {peak_hour:02d}:00 ({peak_count} images)")
        print(f"  Hours with activity: {', '.join([f'{h:02d}:00' for h in hours_with_activity])}")
        print(f"  Number of active hours: {len(hours_with_activity)}")
        
        # Busiest and quietest weekdays
        busiest_day = max(weekday_counts, key=weekday_counts.get)
        quietest_day = min(weekday_counts, key=weekday_counts.get)
        print(f"  Busiest weekday: {weekday_names[busiest_day]} ({weekday_counts[busiest_day]} images)")
        print(f"  Quietest weekday: {weekday_names[quietest_day]} ({weekday_counts[quietest_day]} images)")


if __name__ == "__main__":
    # Define the images folder path
    images_folder = os.path.join(os.path.dirname(__file__), 'images')
    output_dir = os.path.dirname(__file__)
    
    # Read and extract datetime information
    results = read_images_from_folder(images_folder)
    
    # Print the results
    print_results(results)
    
    if results:
        # Print analysis summary
        print_analysis_summary(results)
        
        # Save results to a CSV file
        import csv
        csv_path = os.path.join(output_dir, 'image_datetime.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'date', 'time', 'datetime']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for item in results:
                writer.writerow({
                    'filename': item['filename'],
                    'date': item['date_str'],
                    'time': item['time_str'],
                    'datetime': item['datetime'].strftime('%Y-%m-%d %H:%M:%S')
                })
        
        print(f"\nResults saved to '{csv_path}'")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        # Weekday distribution chart
        weekday_chart_path = os.path.join(output_dir, 'weekday_distribution.png')
        visualize_weekday_distribution(results, weekday_chart_path)
        
        # Heatmap by weekday and hour
        heatmap_path = os.path.join(output_dir, 'weekday_hourly_heatmap.png')
        visualize_time_distribution_by_weekday(results, heatmap_path)
        
        # Overall hourly distribution
        hourly_chart_path = os.path.join(output_dir, 'hourly_distribution.png')
        visualize_hourly_distribution(results, hourly_chart_path)
        
        print("\nAll visualizations have been generated!")
        print(f"  - {weekday_chart_path}")
        print(f"  - {heatmap_path}")
        print(f"  - {hourly_chart_path}")
        
        # Display the plots (comment out for non-interactive environments)
        # plt.show()
