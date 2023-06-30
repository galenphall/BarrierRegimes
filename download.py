"""
Download the dataset from Google Drive or Harvard Dataverse
Author: Galen Hall
Email: galen.p.hall@gmail.com

This script downloads the dataset for "Climate Coalitions and Anti-Coalitions in the United States"
(Hall, Culhane, Roberts 2023) from Google Drive or Harvard Dataverse

Dataverse Location (unpublished):
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FQDH4GU&version=DRAFT#
Google Drive Location:
    https://drive.google.com/drive/u/0/folders/1K6YG1bR4BnInpn_JdWBunoNLMnokLCOO

Date created: 2023-07-30
Date last modified: 2023-07-30
Python version: 3.9
License: MIT License
"""

# Import libraries
import os
import sys


def download_from_google_drive():
    google_drive_loc = "1K6YG1bR4BnInpn_JdWBunoNLMnokLCOO"

    # check whether gdown is installed
    try:
        import gdown
    except ImportError:
        print("gdown is not installed. Install and try again.")
        exit(1)

    # use gdown to download the dataset folder
    os.system(f"gdown --folder --id {google_drive_loc} -O data --folder")


if __name__ == '__main__':
    # Create a folder to store the data
    if not os.path.exists('data'):
        print("Creating a folder to store the data")
        os.makedirs('data')
        os.makedirs('data/SBM')

    # Check if the command line arguments contain a location option ('google_drive' or 'dataverse')
    # If not, ask the user to specify a location
    if len(sys.argv) < 2:
        print("Please specify a location to download the data from.")
        print("Options: 'google_drive' or 'dataverse'")
        print("Example: python download.py google_drive")
        exit(1)
    else:
        location = sys.argv[1]
        assert location in ['google_drive', 'dataverse'], "Please specify a valid location: 'google_drive' or 'dataverse'"

        if location == 'google_drive':
            download_from_google_drive()
        else:
            print("Harvard Dataverse location is not yet available. Please download from Google Drive.")
            exit(1)



