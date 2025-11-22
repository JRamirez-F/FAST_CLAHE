# Numba-CLAHE for FIJI

This repository contains three Python scripts implementing an optimized [**CLAHE (Contrast Limited Adaptive Histogram Equalization)**](https://imagej.net/plugins/clahe) workflow in FIJI using [Numba](https://numba.pydata.org/). These scripts accelerate the CLAHE process on images and stacks, supporting grayscale and RGB images, while integrating with the FIJI GUI and Python mode.

---

## Features

* Optimized CLAHE using **Numba** for parallelized CPU computation.
* Supports both **grayscale and RGB images**.
* Works with **single images or stacks**.
* Allows interactive parameter selection in FIJI:

  * `blockRadius`, `bins`, `slope`
  * Choice of **current image** or **file on disk**
* Substantially faster than FIJI's default CLAHE:

  * Example: 1024×1024×100 stack

    * FIJI CLAHE: ~290 seconds
    * Numba CLAHE: ~35 seconds
* RGB mode options:

  * Apply CLAHE on **luminance only** (closer to FIJI result)
  * Apply CLAHE **per channel** (R, G, B)
* Scripts include options to **display processed image** in FIJI or save to disk.

---

## Requirements

* FIJI with **Python mode** enabled.
* Python environment including, at least, the following packages:

  ```text
  PyQt6, QtPy, scikit-image, napari, napari-imagej, numba, numpy, scipy, ndv[qt], pyimagej>=1.7.0, appose
  ```

---

## Scripts

| Script       | Description                        | Notes                                                                                   |
| ------------ | ---------------------------------- | --------------------------------------------------------------------------------------- |
| `Fast_CLAHE_I.py` | Direct Python mode CLAHE in FIJI (GUI)   | Runs without saving to disk; may slow down after repeated runs in same GUI session.     |
| `Fast_CLAHE_II.py` | Python mode CLAHE with disk save (GUI)  | Saves images to disk before opening in FIJI; avoids slowdowns.                          |
| `Fast_CLAHE_III.py` | Pure Python script using ImageJ-Py | Runs outside FIJI GUI; allows explicit thread control and full parameter customization. |

---

## Usage

1. Open FIJI with Python mode enabled.
2. Load one of the scripts in the FIJI script editor.
3. Run the script.
4. For Script 2 and 3, specify an output path for saving processed images.

---

## Performance Notes

* Huge speedup with Numba on stacks and large images.
* Slight differences in pixel values compared to standard FIJI CLAHE are observed:

  * Differences are mostly minor (pixel-wise ±1 in 32-bit subtraction).
* Current scripts support **8-bit and RGB images**.
* 16-bit images are not fully supported; conversion to 8-bit is possible.

---

## Limitations

* Pip installation via the FIJI Python environment may be slow and silent. Wait until completion before restarting FIJI.
* RGB processing may not perfectly match FIJI CLAHE results.
* Displaying images directly in FIJI using `ImagePlus` may slow down repeated iterations; saving to disk avoids this.

---

## License

[Specify your license here, e.g., MIT License]
