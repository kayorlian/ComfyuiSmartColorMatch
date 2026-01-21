# ComfyUI Smart Color Match

A ComfyUI custom node for intelligent color matching between reference and generated images, with support for masked regions.

## Features

- **Two Color Matching Algorithms**:
  - `mkl_neutral`: White balance matching (aligns mean values only)
  - `reinhard_lab`: Full color transfer (aligns both mean and standard deviation)
- **Mask Support**: Ignore specific regions (e.g., clothing) when computing color statistics
- **Blend Control**: Adjust the strength of color matching (0.0 to 1.0)
- **LAB Color Space**: Works in LAB color space for more natural color transfer

## Installation

### Via ComfyUI Manager

1. Open ComfyUI Manager
2. Go to "Install Custom Nodes"
3. Search for "Smart Color Match" or paste the GitHub URL
4. Click "Install"

### Manual Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/comfyui-smart-color-match.git
```

2. Restart ComfyUI

## Usage

### Inputs

- **image_ref** (IMAGE): Reference image (source of color style)
- **image_gen** (IMAGE): Generated image to be color-corrected
- **method** (COMBO): Choose between `mkl_neutral` or `reinhard_lab`
- **blend_factor** (FLOAT): Blending strength (0.0-1.0, default: 1.0)
- **ignore_mask** (MASK, optional): Mask indicating regions to ignore (1=clothes, 0=background)

### Output

- **IMAGE**: Color-corrected image

## How It Works

1. Converts both images to LAB color space
2. Computes color statistics (mean and std) only on valid regions (where mask < 0.5)
3. Applies color transfer to the entire image for natural blending
4. Blends the result with the original image based on blend_factor

## Requirements

- torch
- numpy
- opencv-python (cv2)

## License

MIT License