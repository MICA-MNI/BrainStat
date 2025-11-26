# Issue Summaries and PR Descriptions

## Issue #370: Incompatible dependencies (netneurotools.civet missing)

### Summary
Fixed the `netneurotools` dependency issue by constraining the version to `<0.3.0`.

### Changes
- Updated `requirements.txt`: `netneurotools<0.3.0`
- Updated `setup.py`: `netneurotools<0.3.0`

### Why
`netneurotools>=0.3.0` removed the `civet` submodule that BrainStat depends on in `brainstat/datasets/base.py`. By constraining to `<0.3.0`, pip will install version `0.2.5` which includes the required `civet` module.

### Testing
All 274 tests pass locally with `netneurotools==0.2.5`.

### Branch
`370-incompatible-dependencies-netneurotoolscivet-missing-in-latest-release-030-and-numpy-20-incompatibility-in-brainstat-024`

---

## Issue #371: CI Failure - Dependabot Jinja2 Update

### Summary
Fixed the Dependabot CI failure by updating Sphinx and Jinja2 versions in docs requirements.

### Changes
- Updated `docs/requirements.txt`:
  - `Sphinx==3.5.4` → `Sphinx>=4.0,<8.0`
  - `jinja2<3.1` → `jinja2>=3.0`

### Why
The Dependabot workflow was failing because it couldn't update Jinja2 with the old constraint. Sphinx 3.5.4 is incompatible with newer Jinja2 versions, so both needed updating together.

### Branch
`371-fix-dependabot-jinja2`

---

## Issue #369: Incorrect Histology Profile Downloaded (fs_LR_64k instead of fs_LR_32k)

### Summary
Added warnings for users about the known data resolution mismatch and improved download reliability.

### Changes Made

#### 1. Added User Warnings
- Warning in `read_histology_profile()` when template is `fslr32k`
- Warning in `download_histology_profiles()` when downloading `fslr32k` data
- Both warnings reference issue #369 so users can track the status

#### 2. Fixed Test Skip
- Removed the skip for `fslr32k` test in `test_histology.py`
- Test now passes (netneurotools issue was fixed by #370)

#### 3. Added Download Retry Logic
- Retry mechanism with 3 attempts and exponential backoff
- Handles `RemoteDisconnected`, `URLError`, and `TimeoutError` exceptions
- Added 30-second timeout to prevent hanging
- Improves CI test reliability

### Note
This is a **server-side data issue** - the file on `box.bic.mni.mcgill.ca` for `fslr32k` contains 64k resolution data instead of 32k. The correct fix requires uploading the proper 32k resolution data file to the server.

### Branch
`369-fix-histology-fslr32k-url`

---

## Issue #351: surface_genetic_expression is broken

### Summary
Fixed the float64 to float32 dtype issue that was causing GIFTI writing to fail.
Also fixed compatibility issues with `abagen` 0.1.3 running on Python 3.13 with `pandas` 2.x and `nibabel` 5.x.

### Changes
- Cast surface `Points` to `float32` before writing to GIFTI format
- Added dtype check to avoid unnecessary conversions
- Make a copy of the surface to avoid modifying the original data
- **New**: Modified `surface_genetic_expression` to pass file paths instead of `GiftiImage` objects to `abagen` (fixes `nibabel` 5.x compatibility).
- **New**: Added monkeypatches for `pandas.DataFrame.append` and `set_axis(inplace=...)` (fixes `pandas` 2.0 compatibility).
- **New**: Patched `abagen.utils.labeltable_to_df` to handle missing background label (fixes `KeyError: [0]`).

### Why
The `surface_genetic_expression` function was failing because:
1. Surface coordinate data was in float64 format (GIFTI requires float32).
2. `abagen` 0.1.3 is incompatible with `nibabel` 5.x when passing `GiftiImage` objects directly.
3. `abagen` 0.1.3 uses `pandas.DataFrame.append` and `set_axis(inplace=...)` which were removed in `pandas` 2.0.

### Testing
The fix ensures that surface data is properly converted to GIFTI-compatible format before writing, and that `abagen` runs correctly with modern pandas/nibabel versions.

### Branch
`351-surface_genetic_expression-is-broken-with-current-versions-of`

---

## Issue #366: Kernel Crash on Google Colab / Remote Jupyter

### Suggested Reply (no code changes needed)

Hi @dancebean,

Thank you for reporting this issue! You're correct - this is related to the **VTK/Qt rendering backend** used by BrainSpace's `plot_hemispheres` function, which doesn't work well in headless/remote environments like Google Colab or SSH-tunneled Jupyter servers.

### The Problem
`plot_hemispheres` from BrainSpace uses VTK for 3D rendering, which requires a display backend. In remote environments without a proper display (X11/Xwindows), the kernel crashes when VTK tries to initialize the rendering context.

### Workarounds

**Option 1: Use `screenshot=True` (Recommended for Colab)**
```python
fig = plot_hemispheres(
    pial_left, pial_right, 
    np.mean(thickness, axis=0), 
    # ... other parameters ...
    screenshot=True  # <-- Add this
)
```

**Option 2: Use OSMesa for offscreen rendering (Linux servers)**
```python
import os
os.environ['VTK_OFFSCREEN'] = '1'
# Then import brainspace/brainstat
```

**Option 3: Use Nilearn for visualization instead**
```python
from nilearn import plotting
plotting.plot_surf_stat_map(
    surf_mesh=pial_left,
    stat_map=np.mean(thickness, axis=0)[:len(pial_left.coordinates)],
    hemi='left', view='lateral', colorbar=True, cmap='viridis'
)
```

**Option 4: For Google Colab specifically**
```python
!apt-get install -y xvfb
!pip install pyvirtualdisplay

from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 200))
display.start()
```

### Note
This is a known limitation of VTK-based visualization in headless environments and is not specific to BrainStat. The statistical analysis functions in BrainStat (SLM, etc.) work perfectly fine in remote environments - only the visualization functions from BrainSpace are affected.

Hope this helps!
