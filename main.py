import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont
from importlib import import_module

class MainApp(QWidget):
    
    def __init__(self):
        super().__init__()
        self.setGeometry(0,0,400,600)
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.setWindowTitle("Feats")
        self.stim_filename = None
        self.meeg_filename = None
        self.files_loaded = False
        self.font = QFont()
        self.font.setPointSize(24)
        
        self.layout.setHorizontalSpacing(25)
        self.loadlabel = QLabel("Load")
        self.loadlabel.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.loadlabel.setFont(self.font)
        self.launchlabel = QLabel("Launch")
        self.launchlabel.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.launchlabel.setFont(self.font)
        self.layout.addWidget(self.loadlabel,0,0)
        self.layout.addWidget(self.launchlabel,0,1)
        butts = [["Stimuli","M/EEG","Definitions"],["Explore","Collect","Infer"]]

        self.butt_dict = {}
        for col_idx,col in enumerate(butts):
            for row_idx,butt in enumerate(col):
                tempbutt = QPushButton(butt)
                tempbutt.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
                tempbutt.setFont(self.font)
                tempbutt.setEnabled(False)
                self.layout.addWidget(tempbutt,row_idx+1,col_idx)
                self.butt_dict[butt] = tempbutt

        self.butt_dict["Stimuli"].clicked.connect(self.stim_click)
        self.butt_dict["Stimuli"].setEnabled(True)
        self.butt_dict["M/EEG"].clicked.connect(self.meeg_click)
        self.butt_dict["M/EEG"].setEnabled(True)
        self.butt_dict["Definitions"].clicked.connect(self.def_click)
        
        self.show()

    def check_files(self):
        if self.stim_filename and self.meeg_filename:
            self.files_loaded = True
            self.butt_dict["Definitions"].setEnabled(True)
        else:
            self.files_loaded = False
    
    def stim_click(self):
        self.stim_filename, _ = QFileDialog.getOpenFileName()
        self.check_files()
    
    def meeg_click(self):
        self.meeg_filename, _ = QFileDialog.getOpenFileName()
        self.check_files()
        
    def def_click(self):
        self.def_filename, _ = QFileDialog.getOpenFileName()
        defs = importlib.import_module(self.def_filename)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainapp = MainApp()
    sys.exit(app.exec_())
    