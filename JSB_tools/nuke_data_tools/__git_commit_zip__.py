"""
Run this script to re-zip the nuclear data in case changes were made
"""
from pathlib import Path
import shutil
import pickle


pwd = Path(__file__).parent
nuke_data_dir = pwd/'data'

latest_nuke_data_zip_size_path = pwd / '__nuke_zip_size__.txt'

def check_for_change_in_zipped_file():
    if
    #  .stat().st_size


zip_data()




