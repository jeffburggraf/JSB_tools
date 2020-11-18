"""
re-zip the nuke data in case changes were made
"""
from zipfile import ZipFile
from pathlib import Path
import shutil

pwd = Path(__file__).parent
nuke_data_dir = pwd/'data'

# ZipFile(nuke_data_dir, 'w')
shutil.make_archive(pwd/'data', 'zip', nuke_data_dir)


