# Quick Start - Visualization Viewer

## How to run it?

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate sample data**:
   ```bash
   python run.py --gen-samples
   ```

3. **Run the application**:
   ```bash
   python run.py
   ```

## Keyboard Shortcuts

- **Next image**: `→` or "Next" button
- **Previous image**: `←` or "Previous" button
- **Open in external program**: "Open in program" button
- **Exit**: `Esc` or close the window

## Brief Overview of Visualizations

### Histograms
- **histogram_normal.png**: Normal distribution (bell-shaped)
- **histogram_poisson.png**: Distribution of rare events

### Plots
- **plot_sin.png**: Sine function (wave)
- **plot_quadratic.png**: Quadratic function (parabola)

### Heat Maps
- **heatmap_random.png**: Random data
- **heatmap_sincos.png**: Mathematical pattern sin*cos

## What's Next?

1. **Detailed documentation**: See [README_ALGORITHMS.md](README_ALGORITHMS.md)
2. **Application tests**: Run `python run.py --test`
3. **Custom data**: Create your own visualizations and open the directory with `python run.py your_directory`

## Need help?

If something doesn't work:
1. Check if you have all dependencies installed
2. Make sure the data directory exists
3. Check file read permissions
4. Look at the full documentation in README.md 