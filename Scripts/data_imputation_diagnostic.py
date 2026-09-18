# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot(DFraw, DFfil, col = 'BMI', suffix = 'tt'):
    
    # Get original data and imputed data
    original_data = DFraw[col]
    imputed_data = DFfil[col] 
    
    # Set figure size and DPI
    plt.figure(figsize = (3, 4), dpi = 300)
    
    # Plot KDE of imputed data (red)
    sns.kdeplot(imputed_data, color = 'red', legend = False)
    
    # Plot KDE of original data (blue), placed on top
    sns.kdeplot(original_data.dropna(), color = 'blue', legend = False)
    
    # Set labels and title with bold font
    plt.xlabel('BMI', fontsize = 12, fontweight = 'bold')
    plt.ylabel('Density', fontsize = 12, fontweight = 'bold')
    plt.title('BMI: Original vs Imputed Data', fontsize = 14, fontweight = 'bold')
    plt.savefig(f"Figs/OriginalvsImputed.{suffix}.pdf", format = 'pdf', bbox_inches = 'tight')
    plt.show()
    
def plotall(DFraw, DFfil, cols, otdir, suffix = 'tt'):

    # Set figure size and DPI
    fig, axes = plt.subplots(5, 5, figsize = (12, 12), dpi = 300)
    
    # Flatten 2D axes array to 1D for easy iteration
    axes = axes.flatten()
    
    # Iterate through all features
    for i, col in enumerate(cols):
        
        original_data = DFraw[col]
        imputed_data = DFfil[col]
    
        # Plot KDE of imputed data (red)
        sns.kdeplot(imputed_data, color = 'red', ax = axes[i], legend = False)
        
        # Plot KDE of original data (blue), placed on top
        sns.kdeplot(original_data.dropna(), color = 'blue', ax = axes[i], legend = False)
    
        # Set title and labels for each subplot with bold font
        axes[i].set_xlabel(None, fontsize = 12, fontweight = 'bold')
        axes[i].set_ylabel('Density', fontsize = 12, fontweight = 'bold')
        axes[i].set_title(f'{col}', fontsize = 14, fontweight = 'bold')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(f"{otdir}/OriginalvsImputed.{suffix}.png", bbox_inches = 'tight')
    plt.savefig(f"{otdir}/OriginalvsImputed.{suffix}.pdf", bbox_inches = 'tight')
    
    # Close the plot to release memory
    plt.close()
    
if __name__ == '__main__':
    
    # -------------------------------------------------------------------------
    # Load Environment Configurations
    # -------------------------------------------------------------------------
    import configparser
    CF = configparser.ConfigParser()
    CF.read('E:/BaiduSyncdisk/005.Bioinformatics/SCI/005/PCa-screening/Scripts/.config.txt', encoding = 'utf-8')
    rtdir = CF.get('ENV', 'rtdir')
    dtdir = CF.get('ENV', 'dtdir')
    fgdir = f'{rtdir}/Outputs/Fig_Data_Imputation_Diagnostic'
    # -------------------------------------------------------------------------
    
    cols = ['AFP', 'APOE', 'MCHC', 'RDW-CV', 'HCT', 'BASO#', 'MCH', 'MCV', 'HGB', 'PLT', 'NEUT%', 'LY%', 'MPV', 'EO#', 'MONO%', 'BASO%', 'LY#', 'MONO#', 'NEUT#', 'PDW', 'CREA', 'ALT', 'AST', 'Urea']
    
    DFraw = pd.read_excel(f'{rtdir}/{dtdir}/dat.final.xlsx')
    DFrawtt = DFraw[DFraw['Group'].isin(['GZ'])]
    DFrawvn = DFraw[DFraw['Group'].isin(['AH'])]
    
    DFfil = pd.read_excel(f'{rtdir}/{dtdir}/train-test-data.xlsx')
    plotall(DFraw = DFrawtt, DFfil = DFfil, cols = cols, otdir = fgdir, suffix = 'tt')
    
    DFfil = pd.read_excel(f'{rtdir}/{dtdir}/validation-data.xlsx')
    plotall(DFraw = DFrawvn, DFfil = DFfil, cols = cols, otdir = fgdir, suffix = 'vn')