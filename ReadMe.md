# Force-Time Analysis Application
by Kyle Sands and Dr. Jacob Goodin

# Overview
This application analyzes raw vertical ground-reaction force data
from VALD ForceDecks to evaluate countermovement jump (CMJ) and
squat jump (SJ) performance.

The application was developed as a capstone project
to provide force-time analysis with transparent methodology.

# Features
- CMJ and SJ analysis
- Raw ForceDecks CSV import
- Force-time signal processing
- Movement onset, takeoff, and landing detection
- Jump height and takeoff velocity calculations
- Force, power, impulse, and phase duration metrics
- Rate of force development (RFD) analysis
- Landing force and loading rate analysis
- Force-time visualizations
- CSV and Excel report exports

# Requirements
- Python 3.10 or newer
- Required Python packages listed in requirements.txt

# Installation

1. Open a terminal in the project folder.

2. Create a virtual environment:

   python3 -m venv .venv

3. Activate the virtual environment (Mac/Linux):

   source .venv/bin/activate

   Windows:

   .venv\Scripts\activate

4. Install dependencies:

   pip install -r requirements.txt

# Running the Application

From the project folder, run:

   streamlit run app.py

The application will open in your default web browser.

# Using the Application

1. Select countermovement jump (CMJ) or squat jump (SJ).
2. Upload a compatible raw ForceDecks CSV file.
3. Review the calculated performance metrics.
4. Inspect the force-time plots and detected events.
5. Export the results as CSV or Excel.

# Methodology

The application uses:
- 15Hz low-pass filtering for movement mechanics
- 85Hz low-pass filtering for landing-impact metrics
- Raw force for takeoff and landing contact detection
- A 20ms moving window for RFD and loading rate

Detailed biomechanical calculations, event detection methods,
and supporting literature are provided in the accompanying
Force-Time Analysis Methodology document.

# Project Files

- app.py: Streamlit user interface
- analysis.py: Force-time processing and calculations
- requirements.txt: Python dependencies
- README.md: Installation and usage instructions

# Notes

This application is intended for educational and sports science
analysis purposes. Its outputs should be interpreted alongside
the methodological limitations described in the accompanying reference manual.