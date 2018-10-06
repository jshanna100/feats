import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont

app = QApplication(sys.argv)
font = QFont()
font.setPointSize(24)
window = QWidget()
layout = QGridLayout()
layout.setHorizontalSpacing(25)
loadlabel = QLabel("Load")
loadlabel.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
loadlabel.setFont(font)
launchlabel = QLabel("Launch")
launchlabel.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
launchlabel.setFont(font)
layout.addWidget(loadlabel,0,0)
layout.addWidget(launchlabel,0,1)
butts = [["Stimuli","M/EEG","Definitions"],["Explore","Collect","Infer"]]

for col_idx,col in enumerate(butts):
    for row_idx,butt in enumerate(col):
        tempbutt = QPushButton(butt)
        tempbutt.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        tempbutt.setFont(font)
        layout.addWidget(tempbutt,row_idx+1,col_idx)


window.setGeometry(0,0,400,600)
window.setLayout(layout)
window.setWindowTitle("Feats")
window.show()


if __name__ == "__main__":
    sys.exit(app.exec_())
    