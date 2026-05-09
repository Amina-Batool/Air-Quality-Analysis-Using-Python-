# EDA_Analysis.py - Focused on Required Deliverables
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('ggplot')
sns.set_palette("husl")

# Create directory for saved graphs
output_dir = "air_quality_eda"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Load preprocessed data
print("Loading preprocessed data...")
df = pd.read_csv('air_quality_preprocessed.csv')
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Display basic info
print("\n=== Dataset Overview ===")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Cities: {df['City'].nunique()}")
print(f"Countries: {df['Country'].nunique()}")
print(f"AQI range: {df['AQI'].min():.1f} to {df['AQI'].max():.1f}")

# Define columns for analysis
pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
meteorological = ['Temperature', 'Humidity', 'Wind Speed']
all_numerical = pollutants + meteorological + ['AQI']

print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("="*60)

# ============================================
# a. UNIVARIATE ANALYSIS
# ============================================
print("\na. UNIVARIATE ANALYSIS")
print("-" * 40)

# 1. Distribution plots for all numerical variables
print("1. Creating distribution plots...")

fig, axes = plt.subplots(4, 3, figsize=(15, 15))
axes = axes.flatten()

for i, col in enumerate(all_numerical):
    ax = axes[i]
    
    # Histogram with KDE
    sns.histplot(df[col], kde=True, ax=ax, bins=30)
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    
    # Add statistics
    mean_val = df[col].mean()
    median_val = df[col].median()
    ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
    ax.legend(fontsize=8)

# Hide empty subplots
for i in range(len(all_numerical), len(axes)):
    axes[i].set_visible(False)

plt.suptitle('Univariate Analysis: Distributions of All Variables', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/01_univariate_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir}/01_univariate_distributions.png")

# 2. Box plots for outliers detection
print("2. Creating box plots for outlier detection...")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i, col in enumerate(all_numerical):
    ax = axes[i]
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(f'{col} Box Plot')
    ax.set_ylabel(col)
    
    # Calculate outliers
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    
    ax.text(0.05, 0.95, f'Outliers: {len(outliers)}', 
            transform=ax.transAxes, fontsize=9, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Hide empty subplots
for i in range(len(all_numerical), len(axes)):
    axes[i].set_visible(False)

plt.suptitle('Univariate Analysis: Box Plots for Outlier Detection', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/02_box_plots_outliers.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir}/02_box_plots_outliers.png")

# 3. Statistical summary table
print("3. Generating statistical summary...")

stats_summary = df[all_numerical].describe().T
stats_summary['skewness'] = df[all_numerical].apply(lambda x: x.skew())
stats_summary['kurtosis'] = df[all_numerical].apply(lambda x: x.kurtosis())

print("\nStatistical Summary:")
print(stats_summary.round(2))

# Save to CSV
stats_summary.to_csv(f'{output_dir}/statistical_summary.csv')
print(f"✓ Saved: {output_dir}/statistical_summary.csv")

# ============================================
# b. BIVARIATE ANALYSIS
# ============================================
print("\nb. BIVARIATE ANALYSIS")
print("-" * 40)

# 1. Scatter plots: Pollutants vs AQI
print("1. Creating scatter plots: Pollutants vs AQI...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, pollutant in enumerate(pollutants):
    ax = axes[i]
    scatter = ax.scatter(df[pollutant], df['AQI'], alpha=0.5, c=df['AQI'], cmap='viridis')
    ax.set_xlabel(f'{pollutant} Concentration')
    ax.set_ylabel('AQI')
    ax.set_title(f'{pollutant} vs AQI')
    
    # Add correlation coefficient
    corr = df[pollutant].corr(df['AQI'])
    ax.text(0.05, 0.95, f'Corr: {corr:.3f}', 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.colorbar(scatter, ax=axes[-1], label='AQI')
plt.suptitle('Bivariate Analysis: Pollutants vs AQI', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/03_pollutants_vs_aqi.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir}/03_pollutants_vs_aqi.png")

# 2. Meteorological factors vs AQI
print("2. Creating scatter plots: Meteorological factors vs AQI...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, factor in enumerate(meteorological):
    ax = axes[i]
    scatter = ax.scatter(df[factor], df['AQI'], alpha=0.5, c=df['AQI'], cmap='plasma')
    ax.set_xlabel(factor)
    ax.set_ylabel('AQI')
    ax.set_title(f'{factor} vs AQI')
    
    # Add correlation coefficient
    corr = df[factor].corr(df['AQI'])
    ax.text(0.05, 0.95, f'Corr: {corr:.3f}', 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.colorbar(scatter, ax=axes[-1], label='AQI')
plt.suptitle('Bivariate Analysis: Meteorological Factors vs AQI', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/04_meteorological_vs_aqi.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir}/04_meteorological_vs_aqi.png")

# 3. Pairwise relationships among pollutants
print("3. Creating pairwise scatter matrix for pollutants...")

# Sample data for readability (first 500 points)
sample_df = df.sample(min(500, len(df)), random_state=42)

pair_grid = sns.PairGrid(sample_df[pollutants], height=2)
pair_grid.map_upper(sns.scatterplot, alpha=0.5)
pair_grid.map_diag(sns.histplot, kde=True)
pair_grid.map_lower(sns.kdeplot, cmap='Blues', fill=True)

pair_grid.fig.suptitle('Pairwise Relationships Among Pollutants', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/05_pollutants_pairplot.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir}/05_pollutants_pairplot.png")

# ============================================
# c. CORRELATION ANALYSIS
# ============================================
print("\nc. CORRELATION ANALYSIS")
print("-" * 40)

# 1. Correlation matrix heatmap
print("1. Creating correlation matrix heatmap...")

# Calculate correlation matrix
corr_matrix = df[all_numerical].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of All Numerical Variables', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{output_dir}/06_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir}/06_correlation_matrix.png")

# 2. AQI correlation with all variables
print("2. Analyzing AQI correlations...")

aqi_correlations = corr_matrix['AQI'].sort_values(ascending=False)
print("\nAQI Correlations with Other Variables:")
print(aqi_correlations.round(3))

# Visualize top correlations with AQI
top_n = 8
top_corrs = aqi_correlations.drop('AQI').head(top_n)

plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(top_corrs)), top_corrs.values)
plt.yticks(range(len(top_corrs)), top_corrs.index)
plt.xlabel('Correlation Coefficient')
plt.title(f'Top {top_n} Variables Correlated with AQI', fontsize=14)

# Color bars based on correlation strength
for bar, val in zip(bars, top_corrs.values):
    if val > 0:
        bar.set_color('red')
    else:
        bar.set_color('blue')
    
    # Add value labels
    plt.text(val + (0.01 if val > 0 else -0.05), bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', ha='left' if val > 0 else 'right',
             fontsize=9)

plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/07_top_aqi_correlations.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir}/07_top_aqi_correlations.png")

# 3. Significant correlations table
print("3. Identifying significant correlations...")

# Find significant correlations (|corr| > 0.5)
significant_corrs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.5:
            significant_corrs.append({
                'Variable1': corr_matrix.columns[i],
                'Variable2': corr_matrix.columns[j],
                'Correlation': corr_val
            })

if significant_corrs:
    sig_corr_df = pd.DataFrame(significant_corrs)
    sig_corr_df = sig_corr_df.sort_values('Correlation', key=abs, ascending=False)
    print("\nSignificant Correlations (|corr| > 0.5):")
    print(sig_corr_df.round(3))
    sig_corr_df.to_csv(f'{output_dir}/significant_correlations.csv', index=False)
    print(f"✓ Saved: {output_dir}/significant_correlations.csv")
else:
    print("No significant correlations found (|corr| > 0.5)")

# ============================================
# d. COMPARATIVE ANALYSIS
# ============================================
print("\nd. COMPARATIVE ANALYSIS")
print("-" * 40)

# 1. Compare AQI across cities
print("1. Comparing AQI across cities...")

# Get top 10 cities by number of records
top_cities = df['City'].value_counts().head(10).index.tolist()
city_aqi_data = df[df['City'].isin(top_cities)]

plt.figure(figsize=(12, 6))
sns.boxplot(x='City', y='AQI', data=city_aqi_data, palette='Set3')
plt.title('AQI Distribution Across Top 10 Cities', fontsize=14)
plt.xlabel('City')
plt.ylabel('AQI')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{output_dir}/08_city_aqi_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir}/08_city_aqi_comparison.png")

# Calculate city statistics
city_stats = df.groupby('City').agg({
    'AQI': ['mean', 'median', 'std', 'count'],
    'PM2.5': 'mean',
    'PM10': 'mean'
}).round(2)

city_stats.columns = ['_'.join(col).strip() for col in city_stats.columns.values]
city_stats = city_stats.sort_values('AQI_mean', ascending=False)

print("\nTop 5 Cities by Average AQI:")
print(city_stats.head(5))
city_stats.to_csv(f'{output_dir}/city_aqi_statistics.csv')
print(f"✓ Saved: {output_dir}/city_aqi_statistics.csv")

# 2. Compare AQI categories
print("2. Analyzing AQI categories...")

plt.figure(figsize=(10, 6))
category_order = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 
                  'Unhealthy', 'Very Unhealthy', 'Hazardous']
sns.countplot(x='AQI_Category', data=df, order=category_order, palette='RdYlGn_r')
plt.title('Distribution of AQI Categories', fontsize=14)
plt.xlabel('AQI Category')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{output_dir}/09_aqi_category_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir}/09_aqi_category_distribution.png")

# 3. Compare pollutants by AQI category
print("3. Comparing pollutants across AQI categories...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, pollutant in enumerate(pollutants[:6]):
    ax = axes[i]
    sns.boxplot(x='AQI_Category', y=pollutant, data=df, 
                order=category_order, ax=ax, palette='RdYlGn_r')
    ax.set_title(f'{pollutant} by AQI Category')
    ax.set_xlabel('')
    ax.set_ylabel(pollutant)
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Pollutant Concentrations by AQI Category', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/10_pollutants_by_category.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir}/10_pollutants_by_category.png")

# 4. AQI driver analysis
print("4. Analyzing which pollutants drive AQI...")

driver_counts = df['AQI_Driver'].value_counts()

plt.figure(figsize=(10, 6))
colors = plt.cm.Set3(np.arange(len(driver_counts)) / len(driver_counts))
bars = plt.bar(range(len(driver_counts)), driver_counts.values, color=colors)
plt.xticks(range(len(driver_counts)), driver_counts.index, rotation=45, ha='right')
plt.xlabel('Pollutant')
plt.ylabel('Frequency')
plt.title('Frequency of Pollutants Driving AQI', fontsize=14)

# Add percentage labels
total = len(df)
for i, (bar, count) in enumerate(zip(bars, driver_counts.values)):
    percentage = (count / total) * 100
    plt.text(i, bar.get_height() + max(driver_counts.values)*0.01, 
             f'{percentage:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}/11_aqi_driver_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir}/11_aqi_driver_analysis.png")

print("\nAQI Driver Statistics:")
for pollutant, count in driver_counts.items():
    percentage = (count / total) * 100
    print(f"{pollutant}: {count} records ({percentage:.1f}%)")

# ============================================
# e. IDENTIFY CYCLES
# ============================================
print("\ne. IDENTIFYING CYCLES")
print("-" * 40)

# 1. Monthly cycles
print("1. Analyzing monthly cycles...")

if 'Month' in df.columns:
    monthly_avg = df.groupby('Month').agg({
        'AQI': 'mean',
        'PM2.5': 'mean',
        'PM10': 'mean',
        'Temperature': 'mean'
    }).round(2)
    
    monthly_avg.index = pd.to_datetime(monthly_avg.index, format='%m').month_name()
    
    plt.figure(figsize=(12, 8))
    
    # Plot AQI
    plt.subplot(2, 1, 1)
    plt.plot(monthly_avg.index, monthly_avg['AQI'], marker='o', linewidth=2, markersize=8)
    plt.fill_between(monthly_avg.index, monthly_avg['AQI'], alpha=0.3)
    plt.title('Monthly AQI Cycle', fontsize=14)
    plt.ylabel('Average AQI')
    plt.grid(True, alpha=0.3)
    
    # Add peak detection
    max_month = monthly_avg['AQI'].idxmax()
    max_value = monthly_avg['AQI'].max()
    min_month = monthly_avg['AQI'].idxmin()
    min_value = monthly_avg['AQI'].min()
    
    plt.annotate(f'Peak: {max_month}\n({max_value:.1f})', 
                xy=(max_month, max_value),
                xytext=(0.7, 0.9), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot PM2.5 and Temperature
    plt.subplot(2, 1, 2)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    ax1.plot(monthly_avg.index, monthly_avg['PM2.5'], 'g-', marker='s', 
             label='PM2.5', linewidth=2, markersize=6)
    ax2.plot(monthly_avg.index, monthly_avg['Temperature'], 'r-', marker='^', 
             label='Temperature', linewidth=2, markersize=6)
    
    ax1.set_xlabel('Month')
    ax1.set_ylabel('PM2.5 (µg/m³)', color='g')
    ax2.set_ylabel('Temperature (°C)', color='r')
    ax1.tick_params(axis='y', labelcolor='g')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Monthly PM2.5 and Temperature Cycles', fontsize=14)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/12_monthly_cycles.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/12_monthly_cycles.png")
    
    print("\nMonthly Cycle Analysis:")
    print(monthly_avg)
    monthly_avg.to_csv(f'{output_dir}/monthly_cycles.csv')
    print(f"✓ Saved: {output_dir}/monthly_cycles.csv")
    
    # Calculate seasonality
    aqi_range = monthly_avg['AQI'].max() - monthly_avg['AQI'].min()
    seasonality_strength = (aqi_range / monthly_avg['AQI'].mean()) * 100
    print(f"\nSeasonality Strength: {seasonality_strength:.1f}%")
    
    if seasonality_strength > 20:
        print("Strong seasonal pattern detected")
    elif seasonality_strength > 10:
        print("Moderate seasonal pattern detected")
    else:
        print("Weak or no seasonal pattern")

# 2. Day of week cycles
print("2. Analyzing day-of-week cycles...")

if 'Day_of_Week' in df.columns:
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['Day_Name'] = df['Day_of_Week'].map(lambda x: day_names[int(x)] if not pd.isna(x) else 'Unknown')
    
    weekly_avg = df.groupby('Day_Name').agg({
        'AQI': 'mean',
        'PM2.5': 'mean',
        'NO2': 'mean'
    }).round(2)
    
    # Reorder to proper week sequence
    weekly_avg = weekly_avg.reindex(day_names)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(range(len(weekly_avg)), weekly_avg['AQI'])
    plt.xticks(range(len(weekly_avg)), weekly_avg.index, rotation=45, ha='right')
    plt.xlabel('Day of Week')
    plt.ylabel('Average AQI')
    plt.title('Weekly AQI Cycle', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Highlight weekends
    weekend_indices = [4, 5]  # Friday, Saturday (0-indexed)
    for idx in weekend_indices:
        bars[idx].set_color('orange')
    
    plt.subplot(1, 2, 2)
    plt.plot(weekly_avg.index, weekly_avg['PM2.5'], 'b-', marker='o', label='PM2.5')
    plt.plot(weekly_avg.index, weekly_avg['NO2'], 'r-', marker='s', label='NO2')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Day of Week')
    plt.ylabel('Concentration (µg/m³)')
    plt.title('Weekly Pollutant Cycles', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/13_weekly_cycles.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/13_weekly_cycles.png")
    
    print("\nWeekly Cycle Analysis:")
    print(weekly_avg)
    weekly_avg.to_csv(f'{output_dir}/weekly_cycles.csv')
    print(f"✓ Saved: {output_dir}/weekly_cycles.csv")
    
    # Check for weekday-weekend pattern
    weekday_avg = weekly_avg.iloc[:5]['AQI'].mean()
    weekend_avg = weekly_avg.iloc[5:]['AQI'].mean()
    weekend_effect = ((weekend_avg - weekday_avg) / weekday_avg) * 100
    
    print(f"\nWeekday AQI average: {weekday_avg:.2f}")
    print(f"Weekend AQI average: {weekend_avg:.2f}")
    print(f"Weekend effect: {weekend_effect:.1f}% {'increase' if weekend_effect > 0 else 'decrease'}")

# 3. Identify pollution episodes
print("3. Identifying pollution episodes...")

# Define pollution episodes (consecutive days with AQI > 150)
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df_sorted = df.sort_values('Date')
    
    # Create episode flag
    df_sorted['High_Pollution'] = df_sorted['AQI'] > 150
    df_sorted['Episode_Group'] = (df_sorted['High_Pollution'] != df_sorted['High_Pollution'].shift()).cumsum()
    
    # Count episodes
    episodes = df_sorted[df_sorted['High_Pollution']].groupby('Episode_Group').agg({
        'Date': ['min', 'max', 'count'],
        'AQI': 'mean'
    })
    
    episodes.columns = ['Start_Date', 'End_Date', 'Duration', 'Avg_AQI']
    episodes = episodes[episodes['Duration'] >= 2]  # At least 2 consecutive days
    
    if not episodes.empty:
        print(f"\nFound {len(episodes)} pollution episodes (AQI > 150 for ≥2 days):")
        print(episodes.round(2))
        episodes.to_csv(f'{output_dir}/pollution_episodes.csv')
        print(f"✓ Saved: {output_dir}/pollution_episodes.csv")
        
        # Plot episodes
        plt.figure(figsize=(14, 6))
        
        # Plot AQI over time
        plt.plot(df_sorted['Date'], df_sorted['AQI'], 'b-', alpha=0.7, linewidth=1)
        
        # Highlight episodes
        for _, episode in episodes.iterrows():
            plt.axvspan(episode['Start_Date'], episode['End_Date'], 
                       alpha=0.3, color='red', label='Pollution Episode' if _ == 0 else "")
        
        plt.axhline(y=150, color='orange', linestyle='--', label='Unhealthy Threshold (AQI=150)')
        plt.xlabel('Date')
        plt.ylabel('AQI')
        plt.title('Pollution Episodes Over Time', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/14_pollution_episodes.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_dir}/14_pollution_episodes.png")
    else:
        print("No significant pollution episodes found (AQI > 150 for ≥2 consecutive days)")

# ============================================
# SUMMARY REPORT
# ============================================
print("\n" + "="*60)
print("EDA COMPLETE - SUMMARY")
print("="*60)

# Count files created
png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]

print(f"\nTotal visualizations saved: {len(png_files)} PNG files")
print(f"Total data files saved: {len(csv_files)} CSV files")

print("\nKey Findings:")
print("-" * 50)

# Extract key findings from analysis
print("1. DATA OVERVIEW:")
print(f"   • Records: {df.shape[0]}, Cities: {df['City'].nunique()}")
print(f"   • AQI Range: {df['AQI'].min():.1f} - {df['AQI'].max():.1f}")
print(f"   • Average AQI: {df['AQI'].mean():.1f}")

print("\n2. TOP POLLUTANT DRIVERS:")
driver_stats = df['AQI_Driver'].value_counts()
for pollutant, count in driver_stats.head(3).items():
    percentage = (count / len(df)) * 100
    print(f"   • {pollutant}: {percentage:.1f}% of records")

print("\n3. AQI CATEGORY DISTRIBUTION:")
for category in category_order:
    if category in df['AQI_Category'].unique():
        count = (df['AQI_Category'] == category).sum()
        percentage = (count / len(df)) * 100
        print(f"   • {category}: {percentage:.1f}%")

print("\n4. KEY CORRELATIONS:")
top_corr = aqi_correlations.drop('AQI').head(3)
for var, corr in top_corr.items():
    print(f"   • {var} vs AQI: {corr:.3f}")

if 'Month' in df.columns and monthly_avg is not None:
    print("\n5. SEASONAL PATTERNS:")
    worst_month = monthly_avg['AQI'].idxmax()
    best_month = monthly_avg['AQI'].idxmin()
    print(f"   • Worst month: {worst_month} (AQI: {monthly_avg.loc[worst_month, 'AQI']:.1f})")
    print(f"   • Best month: {best_month} (AQI: {monthly_avg.loc[best_month, 'AQI']:.1f})")

print("\n6. CITY ANALYSIS:")
print(f"   • Highest AQI city: {city_stats.index[0]} (Avg AQI: {city_stats.iloc[0]['AQI_mean']:.1f})")
print(f"   • Lowest AQI city: {city_stats.index[-1]} (Avg AQI: {city_stats.iloc[-1]['AQI_mean']:.1f})")

print("\n" + "="*60)
print("NEXT STEPS:")
print("="*60)
print("1. EDA completed with all required deliverables")
print("2. Key insights documented in visualizations")
print("3. Proceed to Model_Building.py for machine learning")