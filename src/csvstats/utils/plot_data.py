import os
import tempfile
from typing import Union

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.pdfgen import canvas

from .plot_helper import get_image_dims

def plot_data(df: pd.DataFrame, 
              c: canvas.Canvas,
              data_column: str,
              group_column: str, 
              width: float,
              height: float,
              margin: float,
              repeated_measures_column: Union[str, None] = None              
              ) -> None:
    """
    Plot a violin plot of the data from a CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data to plot
    c : canvas.Canvas
        ReportLab canvas to draw the plot on
    data_column : str
        Name of the column containing the values to plot
    group_column : str
        Name of the column containing group labels
    width : float
        Width of the plot area in the PDF
    height : float
        Height of the plot area in the PDF
    margin : float
        Margin around the plot area in the PDF
    repeated_measures_column : Union[str, None], optional
        Name of the column identifying repeated measures (e.g., subject ID).
        If provided, each unique level will be colored differently.
    """

    c.showPage() # Start a new page for the plot

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        temp_filename = tmp_file.name
        # Set style for better-looking plots
        sns.set_style("whitegrid")

        fig_width = 12
        fig_height = 7
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create violin plot
        sns.violinplot(data=df, x=group_column, y=data_column, ax=ax, 
                    inner=None, alpha=0.3)
        
        # Overlay individual data points with coloring
        if repeated_measures_column is not None:
            # Color by repeated measures
            sns.stripplot(data=df, x=group_column, y=data_column, 
                        hue=repeated_measures_column, ax=ax,
                        size=6, alpha=1, legend=True)
            
            # Move legend below plot and make it horizontal
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                    ncol=5, frameon=False)
        else:
            # Just show individual points without coloring
            sns.stripplot(data=df, x=group_column, y=data_column, ax=ax,
                        size=6, alpha=1, color='black')
        
        # Customize labels and title
        ax.set_xlabel(group_column.replace('_', ' '), fontsize=12)
        ax.set_ylabel(data_column.replace('_', ' '), fontsize=12)
        ax.set_title(f'Distribution of {data_column.replace("_", " ")} by {group_column.replace("_", " ")}',
                    fontsize=14, pad=20)
        
        # Improve layout
        plt.tight_layout()
        
        plt.savefig(temp_filename, format='png', dpi=300, bbox_inches='tight')
        plt.close()

        image_dims = get_image_dims(
            fig_width=fig_width,
            fig_height=fig_height,
            width=width,
            height=height,
            margin=margin
        )

        c.drawImage(
            temp_filename, 
            image_dims['x'],
            image_dims['y'],
            width=image_dims['img_width'], 
            height=image_dims['img_height']
        )

    # Clean up temp file
    os.unlink(temp_filename)