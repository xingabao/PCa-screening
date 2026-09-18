# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
from matplotlib import font_manager
from clearn.data_preprocessing import compute_missing

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Load Environment Configurations
    # -------------------------------------------------------------------------
    import configparser
    CF = configparser.ConfigParser()
    CF.read('E:/BaiduSyncdisk/005.Bioinformatics/SCI/005/PCa-screening/Scripts/.config.txt', encoding = 'utf-8')
    rtdir = CF.get('ENV', 'rtdir')
    dtdir = CF.get('ENV', 'dtdir')
    fgdir = f'{rtdir}/Outputs/Fig_Missing_Data_Rate'
    # -------------------------------------------------------------------------
    
    # Load raw dataset
    DFraw = pd.read_excel(f'{rtdir}/{dtdir}/dat.final.xlsx')
    
    # Delete non-feature columns
    del DFraw['Group']
    del DFraw['diagnose']

    # Compute missingness using custom preprocessing module
    compute_missing(DFraw, normalize = True)
    
    # Calculate the number of missing values per column
    missing_counts = DFraw.isnull().sum()
    
    # Calculate the percentage of missing values per column
    missing_percentage = (missing_counts / len(DFraw)) * 100
    
    # Create a summary DataFrame for missing data
    missing_data_df = pd.DataFrame({
        'Feature': missing_counts.index,
        'Missing Count': missing_counts,
        'Missing Percentage (%)': missing_percentage.round(1)
    })
    
    # Sort features by missing percentage in descending order
    missing_data_df = missing_data_df.sort_values(by = 'Missing Percentage (%)', ascending = False)
    
    # Calculate cumulative percentage of missing values based on the sorted order
    missing_data_df['Cumulative Percentage'] = (missing_data_df['Missing Count'] / missing_data_df['Missing Count'].sum()).cumsum() * 100
    
    # Filter features for inspection
    missing_data_df[missing_data_df['Missing Percentage (%)'] >= 50][['Missing Percentage (%)']]
    "', '".join(list(missing_data_df[missing_data_df['Missing Percentage (%)'] <= 5]['Feature']))
    
    if True:
        
        # Define color thresholds based on missing percentage
        def get_color(percentage):
            if percentage <= 5:
                return '#FFFACD', 'Good (<= 5%)'                               # Color #FFFACD
            elif percentage <= 10:
                return '#FFDEAD', 'OK (<= 10%)'                                # Color #FFDEAD
            elif percentage <= 20:
                return '#F4A460', 'NotBad (<= 20%)'                            # Color #F4A460
            elif percentage <= 50:
                return '#CFCFCF', 'Bad (<= 50%)'                               # Color #CFCFCF
            else:
                return '#828282', 'Remove (<= 100%)'                           # Color #828282
        
        # Initialize a dual-axis plot
        fig, ax1 = plt.subplots(figsize = (13.531269740998106, 5))
        
        # Plot bar chart representing missing frequency for each feature
        bars = []
        for i in range(len(missing_data_df)):
            perc = missing_data_df['Missing Percentage (%)'].iloc[i]
            count = missing_data_df['Missing Count'].iloc[i]
            feature = missing_data_df['Feature'].iloc[i]
            
            color, label = get_color(perc)
            bar = ax1.bar(feature, count, color = color, label = label if i == 0 else "")
            bars.append(bar)
        
        # Set primary Y-axis (Bar chart) labels and grid lines
        ax1.set_ylabel('Frequency of Missing Values', fontsize = 12, fontweight = 'bold')
        ax1.tick_params(axis = 'x', rotation = 90, labelsize = 12, width = 1.5)
        ax1.grid(True, which = 'both', linestyle = '--', linewidth = 0.5)
        
        # Add percentage labels on top of each bar
        for i, bar in enumerate(bars):
            height = bar[0].get_height()  # Get height of each bar
            perc = missing_data_df['Missing Percentage (%)'].iloc[i]
            ax1.text(bar[0].get_x() + bar[0].get_width() / 2 + 0.6, height + 7.5, f'{perc}%', ha = 'center', fontsize = 8, rotation = 45, color = 'black')
        
        # Create secondary Y-axis for cumulative percentage line
        ax2 = ax1.twinx()
        ax2.plot(missing_data_df['Feature'], missing_data_df['Cumulative Percentage'], color = 'r', marker = 'o', label = 'Cumulative Percentage')
        ax2.set_ylabel('Cumulative Percentage (%)', fontsize = 14, fontweight = 'bold')
        ax2.grid(True, which = 'both', linestyle = '--', linewidth = 0.5)
        
        # Hide top and right spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
       
        # Remove empty margins on X-axis edges
        ax1.margins(x = 0)
        
        # Define custom legends matching the color thresholds
        handles = [
            mpatches.Patch(color = '#FFFACD', label = 'Good\n(<= 5%)'),
            mpatches.Patch(color = '#FFDEAD', label = 'OK\n(<= 10%)'),
            mpatches.Patch(color = '#F4A460', label = 'NotBad\n(<= 20%)'),
            mpatches.Patch(color = '#CFCFCF', label = 'Bad\n(<= 50%)'),
            mpatches.Patch(color = '#828282', label = 'Remove\n(<= 100%)')
        ]
        
        # Format and place the legend at the top of the figure
        legend_font = font_manager.FontProperties(weight = 'bold', size = 12)
        ax1.legend(handles = handles, bbox_to_anchor = (0.5, 1.2), loc = 'upper center', ncol = 5, frameon = False, prop = legend_font)
        plt.savefig(f"{fgdir}/na.value.png", bbox_inches = 'tight', dpi = 300)
        plt.savefig(f"{fgdir}/na.value.pdf", bbox_inches = 'tight', dpi = 300)
       
        # Display plot
        plt.tight_layout()
        plt.show()