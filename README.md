# Differential image auto-alignment tool for MOKE microscope (Used for magnetic domain observation)
(MOKE=Magneto-Optical Kerr effect)
> This is a tool for a certain MOKE microscope, therefore cannot be directly used for other MOKE microscopes. However, you can edit part of this code to utilize the auto-alignment and differential image generation.
# Files and its usage
- diff.py
  - Main script
- diff.bat
  - Drag and drop single video file for processing
- diff_multiple.bat
  - Drag and drop multiple video file for sequential processing
# backend files and its usage
- ocr.py
  - Used to read the characters in the video frames.
  - `get_strings_multiprocessing` is a multiprocessing version of `get_strings`. This will utilize all the cores therefore, with enough calculation power, `get_strings_multiprocessing` will be faster.
- plot.py
  - Plots current alignment result and the square area for contrast measurement.
  - You can change the square area how many times as you please and this script will output the contrast result.
# Environment setup
## [pip] Install the following
- numpy
- matplotlib
- pytesseract
- Pillow
- opencv-contrib-python
## [System] Install the following
- tesseract
## [Path] Set the environment variable (for Windows)
- directroy of tesseract.exe (usually C:\Program Files\Tesseract-OCR)
## Reference materials
- tesseract install guide
  - https://medium.com/quantrium-tech/installing-and-using-tesseract-4-on-windows-10-4f7930313f82
