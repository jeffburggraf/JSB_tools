"""
re-zip the nuke data in case changes were made
"""
from pathlib import Path
import shutil

pwd = Path(__file__).parent
nuke_data_dir = pwd/'data'

shutil.make_archive(pwd/'data', 'zip', nuke_data_dir)


