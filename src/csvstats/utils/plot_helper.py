

def get_image_dims(
    fig_width: float,
    fig_height: float,
    width: float,
    height: float,
    margin: float
):
    """
    Calculate image dimensions and position to fit within page size while preserving aspect ratio.
    Args:
        fig_width: Original figure width in inches
        fig_height: Original figure height in inches
        width: Page width in points
        height: Page height in points
        margin: Margin in points
    """
    # Calculate dimensions preserving aspect ratio
    aspect_ratio = fig_width/fig_height  # Original figure size ratio
    if (width - 2*margin) / (height - 2*margin) > aspect_ratio:
        # Height limited
        img_height = height - 2*margin
        img_width = img_height * aspect_ratio
    else:
        # Width limited
        img_width = width - 2*margin
        img_height = img_width / aspect_ratio
    
    # Center the image on page
    x = margin + (width - 2*margin - img_width) / 2
    y = margin + (height - 2*margin - img_height) / 2

    result = {
        "img_width": img_width,
        "img_height": img_height,
        "x": x,
        "y": y
    }
    return result