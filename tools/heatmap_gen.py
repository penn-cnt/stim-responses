#!/usr/bin/env python3
"""
High-Quality PDF Heatmap Generator for Pre-Post Merged Data

This script creates publication-ready PDF heatmaps based on the interactive heatmap functionality.
Supports multiple patients, value columns, and conditions with SOZ highlighting.

Usage:
    python generate_pdf_heatmaps.py

Features:
- High-quality PDF output using project matplotlib style
- SOZ channel highlighting in red
- Multiple value columns: change, log_change, baseline_spike_rate, stim_spike_rate
- Support for both 'pre' and 'post' conditions
- Automatic colormap scaling
- Batch processing for multiple patients
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Import project configuration
from config import CONFIG

# Constants
MM_TO_IN = 1/25.4
ospj = os.path.join

class PDFHeatmapGenerator:
    """
    Generator for high-quality PDF heatmaps based on pre-post merged data.
    """
    
    def __init__(self, use_project_style=True):
        """Initialize the PDF heatmap generator."""
        
        # Available value columns (limited to specific options from interactive script)
        self.available_value_columns = ['change', 'log_change', 'baseline_spike_rate', 'stim_spike_rate']
        
        # User-friendly display names for value columns
        self.value_column_nicknames = {
            'change': 'Change from Baseline',
            'log_change': 'Log Change from Baseline',
            'baseline_spike_rate': 'Baseline Spike Rate (Hz)',
            'stim_spike_rate': 'Stimulation Spike Rate (Hz)'
        }
        
        # Set up matplotlib style
        if use_project_style:
            style_path = ospj(CONFIG.results_dir, '../code/figures.mplstyle')
            if os.path.exists(style_path):
                plt.style.use(style_path)
                print(f"Using project matplotlib style: {style_path}")
            else:
                print("Project style not found, using high-quality defaults")
                self.setup_high_quality_defaults()
        else:
            self.setup_high_quality_defaults()
        
        self.setup_data()
        
    def setup_high_quality_defaults(self):
        """Set up high-quality matplotlib defaults for PDF output."""
        plt.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 600,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 9,
            'figure.autolayout': True,
            'pdf.fonttype': 42,  # Embed fonts in PDF
            'font.family': 'DejaVu Sans'
        })
        
    def setup_data(self):
        """Load the pre-post merged dataset."""
        print("Loading pre-post merged data...")
        self.pre_post_data = pd.read_csv(ospj(CONFIG.dataset_dir, "pre_post_merged_data.csv"))
        print(f"Loaded full dataset: {len(self.pre_post_data)} rows")
        
        # Separate pre and post conditions
        self.pre_data = self.pre_post_data[self.pre_post_data['condition'] == 'pre'].copy()
        self.post_data = self.pre_post_data[self.pre_post_data['condition'] == 'post'].copy()
        
        print(f"Pre condition data: {len(self.pre_data)} rows, {len(self.pre_data['hup_id'].unique())} patients")
        print(f"Post condition data: {len(self.post_data)} rows, {len(self.post_data['hup_id'].unique())} patients")
        
        # Check data availability for our specific value columns
        print("\nData availability by column:")
        for col in self.available_value_columns:
            if col in self.pre_post_data.columns:
                pre_count = self.pre_data[col].notna().sum()
                post_count = self.post_data[col].notna().sum()
                print(f"  {col}: {pre_count} pre values, {post_count} post values")
                
    def get_soz_info(self, patient_data):
        """Get SOZ information for a patient."""
        soz_stim_channels = set()
        soz_rec_channels = set()
        
        if patient_data is not None and len(patient_data) > 0:
            # Get unique SOZ stimulation channels
            soz_stim_mask = (patient_data['soz_stim_ch'] == True) | (patient_data['soz_stim_ch'] == 'True')
            soz_stim = patient_data[soz_stim_mask]['stim_ch'].unique()
            soz_stim_channels = set(soz_stim)
            
            # Get unique SOZ recording channels
            soz_rec_mask = (patient_data['soz_rec_ch'] == True) | (patient_data['soz_rec_ch'] == 'True')
            soz_rec = patient_data[soz_rec_mask]['recording_ch'].unique()
            soz_rec_channels = set(soz_rec)
        
        return soz_stim_channels, soz_rec_channels
        
    def get_colormap_limits(self, pivot_data, value_column):
        """Get appropriate colormap limits based on the value column."""
        if pivot_data is None or pivot_data.empty:
            return -1, 1
            
        data_min = pivot_data.min().min()
        data_max = pivot_data.max().max()
        
        if value_column in ['log_stim_spike_rate', 'log_change']:
            # For log values, use symmetric range around 0
            abs_max = max(abs(data_min), abs(data_max))
            return -abs_max, abs_max
        elif value_column in ['stim_spike_rate', 'baseline_spike_rate']:
            # For raw rates, use 0 as minimum
            return max(0, data_min), data_max
        elif value_column == 'change':
            # For change values, center around 0
            abs_max = max(abs(data_min), abs(data_max))
            return -abs_max, abs_max
        else:
            return data_min, data_max
            
    def create_patient_heatmap(self, patient_id, condition='pre', value_column='change', 
                              figsize=(12, 10), save_pdf=True, output_dir=None,
                              show_xlabel=True, show_ylabel=True, 
                              show_xticklabels=True, show_yticklabels=True,
                              show_xticks=True, show_yticks=True):
        """
        Create a high-quality heatmap for a specific patient.
        
        Parameters:
        -----------
        patient_id : int
            Patient ID to create heatmap for
        condition : str
            'pre' or 'post' condition
        value_column : str
            Column to plot ('change', 'log_change', 'baseline_spike_rate', 'stim_spike_rate')
        figsize : tuple
            Figure size in inches
        save_pdf : bool
            Whether to save as PDF
        output_dir : str
            Output directory (defaults to CONFIG.fig_dir)
        show_xlabel : bool
            Whether to show x-axis label (default: True)
        show_ylabel : bool
            Whether to show y-axis label (default: True)
        show_xticklabels : bool
            Whether to show x-axis tick labels (default: True)
        show_yticklabels : bool
            Whether to show y-axis tick labels (default: True)
        show_xticks : bool
            Whether to show x-axis tick marks (default: True)
        show_yticks : bool
            Whether to show y-axis tick marks (default: True)
        
        Returns:
        --------
        matplotlib.figure.Figure or None
        """
        
        if output_dir is None:
            output_dir = CONFIG.fig_dir
            
        # Select appropriate dataset
        if condition == 'pre':
            data = self.pre_data
        elif condition == 'post':
            data = self.post_data
        else:
            raise ValueError("Condition must be 'pre' or 'post'")
            
        # Filter data for the specific patient
        patient_data = data[data['hup_id'] == patient_id].copy()
        
        if len(patient_data) < 5:
            print(f"Not enough data for patient {patient_id} in {condition} condition")
            return None
            
        # Filter out rows with missing value column data
        filtered_data = patient_data.dropna(subset=[value_column])
        
        if len(filtered_data) == 0:
            print(f"No valid {value_column} data for patient {patient_id} in {condition} condition")
            return None
            
        # Create a pivot table for the heatmap
        pivot_data = filtered_data.pivot_table(
            values=value_column, 
            index='stim_ch', 
            columns='recording_ch',
            aggfunc='mean'
        )
        
        # Sort the index and columns for consistent ordering
        pivot_data = pivot_data.reindex(index=sorted(pivot_data.index))
        pivot_data = pivot_data.reindex(columns=sorted(pivot_data.columns))
        
        # Get SOZ information for coloring
        soz_stim_channels, soz_rec_channels = self.get_soz_info(patient_data)
        
        # Create the figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine appropriate colormap limits
        vmin, vmax = self.get_colormap_limits(pivot_data, value_column)
        
        # Create the heatmap
        heatmap = sns.heatmap(
            pivot_data, 
            cmap='coolwarm', 
            center=0 if vmin < 0 < vmax else None,
            vmin=vmin,
            vmax=vmax,
            annot=False, 
            linewidths=0.5,
            cbar_kws={'label': self.value_column_nicknames.get(value_column, value_column)},
            mask=pivot_data.isna(), 
            xticklabels=pivot_data.columns,
            yticklabels=pivot_data.index,
            ax=ax
        )
        
        # Highlight SOZ channels in red
        # Color SOZ stimulation channel labels (y-axis) red
        for i, label in enumerate(ax.get_yticklabels()):
            if label.get_text() in soz_stim_channels:
                label.set_color('red')
                label.set_fontweight('bold')
        
        # Color SOZ recording channel labels (x-axis) red  
        for i, label in enumerate(ax.get_xticklabels()):
            if label.get_text() in soz_rec_channels:
                label.set_color('red')
                label.set_fontweight('bold')
        
        # Set title and labels
        title = f'Patient {patient_id}: {self.value_column_nicknames.get(value_column, value_column)} ({condition.upper()} condition)'
        if len(soz_stim_channels) > 0 or len(soz_rec_channels) > 0:
            title += '\n(Red labels = SOZ channels)'
        ax.set_title(title, fontsize=14, pad=20)
        
        # Set axis labels based on parameters
        if show_xlabel:
            ax.set_xlabel('Recording Channel', fontsize=12)
        else:
            ax.set_xlabel('')
            
        if show_ylabel:
            ax.set_ylabel('Stimulation Channel', fontsize=12)
        else:
            ax.set_ylabel('')
        
        # Control tick labels and tick marks
        if show_xticklabels:
            # Rotate labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8, ha='center')
        else:
            ax.set_xticklabels([])
            
        if show_yticklabels:
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8, va='center')
        else:
            ax.set_yticklabels([])
            
        # Control tick marks
        if not show_xticks:
            ax.tick_params(axis='x', which='both', length=0)
            
        if not show_yticks:
            ax.tick_params(axis='y', which='both', length=0)
        
        # Tight layout
        plt.tight_layout()
        
        # Save as PDF if requested
        if save_pdf:
            # Create filename
            filename = f'patient_{patient_id}_{condition}_{value_column}_heatmap.pdf'
            filepath = ospj(output_dir, filename)
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Save with high quality settings
            plt.savefig(filepath, format='pdf', dpi=600, bbox_inches='tight')
            print(f"Saved: {filepath}")
            
        return fig
        
    def generate_all_heatmaps_for_patient(self, patient_id, conditions=['pre', 'post'], 
                                        value_columns=None, output_dir=None, **heatmap_kwargs):
        """
        Generate all combinations of heatmaps for a patient.
        
        Parameters:
        -----------
        patient_id : int
            Patient ID
        conditions : list
            List of conditions to generate ['pre', 'post']
        value_columns : list
            List of value columns to generate (defaults to all available)
        output_dir : str
            Output directory
        **heatmap_kwargs : dict
            Additional keyword arguments to pass to create_patient_heatmap
            (e.g., figsize, show_xlabel, show_yticklabels, etc.)
        """
        
        if value_columns is None:
            value_columns = self.available_value_columns
            
        if output_dir is None:
            output_dir = ospj(CONFIG.fig_dir, f'patient_{patient_id}_heatmaps')
            
        print(f"\nGenerating heatmaps for patient {patient_id}...")
        
        generated_count = 0
        for condition in conditions:
            for value_column in value_columns:
                # Check if data exists
                data = self.pre_data if condition == 'pre' else self.post_data
                patient_data = data[data['hup_id'] == patient_id]
                
                if len(patient_data) > 0 and patient_data[value_column].notna().sum() > 0:
                    try:
                        fig = self.create_patient_heatmap(
                            patient_id, condition, value_column, 
                            save_pdf=True, output_dir=output_dir,
                            **heatmap_kwargs
                        )
                        if fig is not None:
                            generated_count += 1
                            plt.close(fig)  # Close to save memory
                    except Exception as e:
                        print(f"  Error generating {condition} {value_column}: {e}")
                else:
                    print(f"  Skipping {condition} {value_column}: no data")
                    
        print(f"Generated {generated_count} heatmaps for patient {patient_id}")
        
    def generate_batch_heatmaps(self, patient_ids=None, conditions=['pre', 'post'], 
                              value_columns=None, output_dir=None, **heatmap_kwargs):
        """
        Generate heatmaps for multiple patients.
        
        Parameters:
        -----------
        patient_ids : list
            List of patient IDs (defaults to all available)
        conditions : list
            List of conditions ['pre', 'post']
        value_columns : list
            List of value columns (defaults to all available)
        output_dir : str
            Base output directory
        **heatmap_kwargs : dict
            Additional keyword arguments to pass to create_patient_heatmap
            (e.g., figsize, show_xlabel, show_yticklabels, etc.)
        """
        
        if patient_ids is None:
            # Get all patients with data
            pre_patients = set(self.pre_data['hup_id'].unique())
            post_patients = set(self.post_data['hup_id'].unique())
            patient_ids = sorted(list(pre_patients.union(post_patients)))
            
        if value_columns is None:
            value_columns = self.available_value_columns
            
        if output_dir is None:
            output_dir = ospj(CONFIG.fig_dir, 'batch_heatmaps')
            
        print(f"Starting batch generation for {len(patient_ids)} patients...")
        print(f"Conditions: {conditions}")
        print(f"Value columns: {value_columns}")
        print(f"Output directory: {output_dir}")
        
        total_generated = 0
        for patient_id in patient_ids:
            patient_output_dir = ospj(output_dir, f'patient_{patient_id}')
            self.generate_all_heatmaps_for_patient(
                patient_id, conditions, value_columns, patient_output_dir,
                **heatmap_kwargs
            )
            
        print(f"\nBatch generation complete!")
        
    def get_available_patients(self, condition=None):
        """Get list of available patients with data."""
        if condition == 'pre':
            return sorted(self.pre_data['hup_id'].unique())
        elif condition == 'post':
            return sorted(self.post_data['hup_id'].unique())
        else:
            pre_patients = set(self.pre_data['hup_id'].unique())
            post_patients = set(self.post_data['hup_id'].unique())
            return sorted(list(pre_patients.union(post_patients)))
            
    def get_data_summary(self):
        """Print a summary of available data."""
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        
        print(f"Total rows in dataset: {len(self.pre_post_data)}")
        print(f"Pre condition: {len(self.pre_data)} rows")
        print(f"Post condition: {len(self.post_data)} rows")
        
        pre_patients = sorted(self.pre_data['hup_id'].unique())
        post_patients = sorted(self.post_data['hup_id'].unique())
        
        print(f"\nAvailable patients:")
        print(f"  Pre condition: {len(pre_patients)} patients {pre_patients}")
        print(f"  Post condition: {len(post_patients)} patients {post_patients}")
        
        print(f"\nValue columns with data:")
        for col in self.available_value_columns:
            if col in self.pre_post_data.columns:
                pre_count = self.pre_data[col].notna().sum()
                post_count = self.post_data[col].notna().sum()
                pre_pts = len(self.pre_data[self.pre_data[col].notna()]['hup_id'].unique())
                post_pts = len(self.post_data[self.post_data[col].notna()]['hup_id'].unique())
                print(f"  {col}: {pre_count} pre values ({pre_pts} pts), {post_count} post values ({post_pts} pts)")


def main():
    """Main function demonstrating usage."""
    print("High-Quality PDF Heatmap Generator")
    print("=" * 50)
    
    # Create generator
    generator = PDFHeatmapGenerator()
    
    # Print data summary
    generator.get_data_summary()
    
    # Example 1: Generate a single heatmap
    print("\n" + "="*50)
    print("EXAMPLE 1: Single Heatmap")
    print("="*50)
    
    # Check which patients have good data
    available_patients = generator.get_available_patients('pre')
    if len(available_patients) > 0:
        example_patient = available_patients[0]
        print(f"Generating example heatmap for patient {example_patient}...")
        
        fig = generator.create_patient_heatmap(
            patient_id=example_patient,
            condition='pre',
            value_column='change',
            save_pdf=True
        )
        
        if fig is not None:
            plt.show()  # Display the plot
            plt.close(fig)
        
    # Example 2: Generate all heatmaps for a specific patient
    print("\n" + "="*50)
    print("EXAMPLE 2: All Heatmaps for One Patient")
    print("="*50)
    
    if len(available_patients) > 0:
        example_patient = available_patients[0]
        print(f"Generating all heatmaps for patient {example_patient}...")
        
        generator.generate_all_heatmaps_for_patient(
            patient_id=example_patient,
            conditions=['pre'],  # Just pre for this example
            value_columns=['change', 'baseline_spike_rate']  # Just a couple columns
        )
    
    # Example 3: Batch generation (commented out to avoid generating too many files)
    print("\n" + "="*50)
    print("EXAMPLE 3: Batch Generation (Commented Out)")
    print("="*50)
    print("To generate heatmaps for all patients, uncomment the following:")
    print("# generator.generate_batch_heatmaps(")
    print("#     patient_ids=available_patients[:3],  # First 3 patients")
    print("#     conditions=['pre'],")
    print("#     value_columns=['change']")
    print("# )")
    
    # Uncomment the following lines to actually run batch generation:
    # generator.generate_batch_heatmaps(
    #     patient_ids=available_patients[:3],  # First 3 patients only
    #     conditions=['pre'],
    #     value_columns=['change']
    # )
    
    print(f"\nPDFs will be saved to: {CONFIG.fig_dir}")
    print("Use the generator object to create custom heatmaps as needed.")


if __name__ == "__main__":
    main()