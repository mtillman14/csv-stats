import tempfile
import os
import json
from typing import Union, Any
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from .plot_data import plot_data
from .plot_helper import get_image_dims

def dict_to_pdf(df: pd.DataFrame, data: dict, data_to_plot: Union[dict, None] = None, filename: Union[str, None] = 'output.pdf'):
    """
    Convert dictionary to PDF and optionally add bell curve plot.
    
    Args:
        data: The dictionary to convert
        data_to_plot: Optional dict with 'means' and 'std_devs' for bell curves
        filename: Output PDF filename
    """
    if filename is None:
        return # Don't save if no filename provided
    
    filename = str(filename)
        
    # Create the PDF canvas
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    margin = 0.5 * inch
    y_position = height - margin
    line_height = 0.15 * inch
    
    # Write dictionary content
    json_str = json.dumps(data, indent=2)
    c.setFont("Courier", 9)
    
    for line in json_str.split('\n'):
        if y_position < margin:
            c.showPage()
            c.setFont("Courier", 9)
            y_position = height - margin
        c.drawString(margin, y_position, line)
        y_position -= line_height
    
    # Add plot if provided
    if data_to_plot:       
        means = data_to_plot['means']
        std_devs = data_to_plot['std_devs']
        x_range = data_to_plot.get('x_range', None) 
        draw_bell_curve(means, std_devs, c, width, height, margin, x_range)

        # Plot the data in the dataframe
        plot_data(df, 
                    c,
                    data_column=data['data_column'],
                    group_column=data['group_column'],
                    width=width,
                    height=height,
                    margin=margin,
                    repeated_measures_column=data['repeated_measures_column']
                  )
    
    c.save()
    print(f"PDF saved as '{filename}'")


def convert_types(obj: Any) -> Any:
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_types(item) for item in obj]
    elif hasattr(obj, 'item'):  # NumPy scalar types
        return obj.item()
    return obj


def get_plot_data(summary_stats: dict, render_plot: bool) -> dict:
    """Format the means and std_devs for plotting bell curves."""
    if not render_plot:
        return None
    
    plot_data = {
        "means": summary_stats['grouped']['mean'],
        "std_devs": summary_stats['grouped']['std_dev']
    }
    return plot_data


def dict_to_json(result: dict, filename: Union[str, Path]) -> str:
    """Save a result dict to a specified JSON file."""
    
    str_result = json.dumps(result)
    with open(filename, 'w') as f:
        # TODO: Use the `str_result` to write the string directly to file without second dump
        json.dump(f, result)

    return str_result


def save_handler(df: pd.DataFrame, result: dict, filename: Union[str, Path, None], render_plot: bool = False) -> Union[dict, str]:
    """Called by each of the tests to determine how to save the data given the save file path and other parameters"""

    if filename is None:
        return result
    
    filename = str(filename)

    converted_result = convert_types(result)

    if filename.endswith(".pdf"):
        data_to_plot = get_plot_data(converted_result["summary_stats"], render_plot=render_plot)
        returned = dict_to_pdf(df, converted_result, data_to_plot=data_to_plot, filename=filename)
    elif filename.endswith(".json"):
        returned = dict_to_json(converted_result, filename=filename)

    return returned


def draw_bell_curve(means: dict, std_devs: dict, c: canvas.Canvas, width: float, height: float, margin: float, x_range: list = None) -> None:
    """
    Draw bell curves for given means and standard deviations on the provided PDF canvas.
    Args:
        means: Dictionary of group labels to means
        std_devs: Dictionary of group labels to standard deviations
        x_range: Optional list [x_min, x_max] for x-axis range
        c: ReportLab canvas to draw on
        width: Width of the PDF page
        height: Height of the PDF page
        margin: Margin to leave around the plot    
    """
    c.showPage()  # Start new page for plot    

    # Create bell curves plot
    if x_range is None:
        all_means = list(means.values())
        all_stds = list(std_devs.values())
        x_min = min(all_means) - 4 * max(all_stds)
        x_max = max(all_means) + 4 * max(all_stds)
    else:
        x_min, x_max = x_range

    # Create temporary file for plot
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        temp_filename = tmp_file.name

        fig_width = 12
        fig_height = 7
        
        plt.figure(figsize=(fig_width, fig_height))                                
    
        x = np.linspace(x_min, x_max, 1000)
        colors = plt.cm.Set2(np.linspace(0, 1, len(means)))
        
        for idx, (label, mean) in enumerate(means.items()):
            std_dev = std_devs[label]
            y = norm.pdf(x, mean, std_dev)
            plt.plot(x, y, linewidth=2.5, label=f'{label} (μ={round(mean, 4)}, σ={round(std_dev, 4)})', color=colors[idx])
            plt.fill_between(x, y, alpha=0.2, color=colors[idx])
            plt.axvline(mean, color=colors[idx], linestyle='--', linewidth=1.5, alpha=0.7)
        
        plt.xlabel('x', fontsize=13)
        plt.ylabel('Probability Density', fontsize=13)
        plt.title('Normal Distribution Comparison', fontsize=15, fontweight='bold')
        plt.legend(fontsize=11, loc='best', framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save to temp file instead of BytesIO
        plt.savefig(temp_filename, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        image_dims = get_image_dims(
            fig_width=fig_width,
            fig_height=fig_height,
            width=width,
            height=height,
            margin=margin
        )
        
        # Draw image from temp file with centered positioning
        c.drawImage(
            temp_filename, 
            image_dims['x'],
            image_dims['y'],
            width=image_dims['img_width'], 
            height=image_dims['img_height']
        )
    
    # Clean up temp file
    os.unlink(temp_filename)