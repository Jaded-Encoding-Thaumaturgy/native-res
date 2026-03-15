# vsview-nativeres

`vsview-nativeres` is the [VSView](https://github.com/Jaded-Encoding-Thaumaturgy/vs-view) frontend for [`nativeres`](https://github.com/Jaded-Encoding-Thaumaturgy/nativeres).

It adds an interactive `Native Resolution` tool to VSView so you can run descale analysis directly on the current output and frame instead of switching back to the CLI.

## Features

- `Get Native` tab for plotting descale error across a width or height range
- `Get Scaler` tab for ranking kernels against a chosen target dimension
- `Get Frequencies` tab for DCT-based frequency inspection on the current frame
- Global plugin settings for kernel presets and chart styling
- Import support for previously exported `getnative` JSON and CSV result files
- Automatic CSV dumps for computed `Get Native` plots in VSView local storage

## Installation

Install the plugin package directly:

```bash
pip install vsview-nativeres
```

Or install it through the main package extras:

```bash
pip install nativeres[plugin]
```

The plugin registers both as a VSView tool panel and tool dock under `Native Resolution`.

## Usage

Open VSView, load a source or VapourSynth script, and open the `Native Resolution` tool.

The plugin always works on the currently selected VSView output and displayed frame.

### Get Native

This tab is the interactive version of `nativeres getnative`.

You can:

- Analyze either `Width` or `Height`
- Choose a min/max range and fractional step size
- Select the descale kernel to test
- Choose the error metric (`MAE`, `MSE`, or `RMSE`)

Computed results are shown as an interactive plot. Each plot is also added to a history list:

- Single-click switches back to that plot
- Double-click also seeks VSView back to the frame used for that computation
- Right-click opens the plot context menu, including `Reset Zoom` and copy/export actions

The tab can also import saved results from `JSON` and `CSV` files.

When a calculation finishes, the plugin writes the plot data as CSV into VSView local storage if that storage path is available.

### Get Scaler

This tab is the interactive version of `nativeres getscaler`.

You can:

- Test either a target `Width` or `Height`
- Compare all kernels from the global plugin settings
- Choose the error metric (`MAE`, `MSE`, or `RMSE`)
- Optionally apply an edge-detection mask to reduce noise influence

Results are shown as a sorted comparison table with:

- Kernel name
- Relative error percentage against the best match
- Raw error value

### Get Frequencies

This tab is the interactive version of `nativeres getfreq`.

You can:

- Compute horizontal and vertical DCT distributions
- Adjust the `Cull Rate` to focus analysis more toward the image center
- Adjust the spike detection `Radius`
- Right-click the plot for `Reset Zoom` and copy/export actions

## Settings

### Global settings

Global settings affect all sessions:

- Chart theme
- Frequency plot colors (Only for the default theme)
- Kernel list used by `Get Native` and `Get Scaler`

### Local settings

Local settings persist recent UI choices for `Get Native` and `Get Scaler`.

## Notes

- `Get Native` warns before launching very large scans with more than 2000 tested dimensions.
- Plot context menus support `Reset Zoom` plus copy/export actions for `PNG`, `SVG`, `JSON`, and `CSV`.
- As with the CLI tools, results should be validated across multiple frames before deciding on a descale setup.
