import json
from typing import Union, Any

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def dict_to_pdf(data: dict, filename: Union[str, None] = 'output.pdf'):
    """
    Convert a dictionary to a formatted JSON string and save as PDF.
    
    Args:
        data: The dictionary to convert
        filename: Output PDF filename (default: 'output.pdf')
    """
    # Convert NumPy types to native Python types for JSON serialization
    def convert_types(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # NumPy scalar types
            return obj.item()
        return obj
    
    if filename is None:
        return
    
    filename = str(filename)
    
    data = convert_types(data)

    # Convert dictionary to formatted JSON string
    json_str = json.dumps(data, indent=2)
    
    # Create PDF
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Margins and setup
    margin = 0.5 * inch
    y_position = height - margin
    line_height = 0.15 * inch
    
    # Use monospace font for better JSON formatting
    c.setFont("Courier", 9)
    
    # Draw each line of the JSON
    for line in json_str.split('\n'):
        # Check if we need a new page
        if y_position < margin:
            c.showPage()
            c.setFont("Courier", 9)
            y_position = height - margin
        
        # Draw the line
        c.drawString(margin, y_position, line)
        y_position -= line_height
    
    # Save the PDF
    c.save()
    print(f"PDF saved as {filename}")