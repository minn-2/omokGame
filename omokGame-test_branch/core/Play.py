import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from omok_gui import OmokGUI

if __name__ == '__main__':
    gui = OmokGUI()
    gui.run()