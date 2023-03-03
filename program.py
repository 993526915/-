import os
import sys
from collections import Counter
from math import ceil
from mmcls.apis import init_model, inference_model
from PyQt5.QtCore import pyqtSignal, QThread, QObject
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QListWidgetItem
from tqdm import tqdm, trange
import Kfbreader.kfbReader as kr
from models import cut_t0_40 as cut
from qt.AI import *
import cv2


class signal(QObject):
    signalSaveFile = pyqtSignal(str)
    signalProgressBar = pyqtSignal(int)
    signalProgressBarIF = pyqtSignal(int)
    signalIfRes = pyqtSignal(str)

    def sendSaveFileSignal(self, fileName):
        self.signalSaveFile.emit(fileName)

    def sendProgressBar(self, numStep):
        self.signalProgressBar.emit(numStep)

    def sendProgressBarIF(self, numStep):
        self.signalProgressBarIF.emit(numStep)

    def sendInterfaceRes(self, result):
        self.signalIfRes.emit(result)


signal = signal()


class cutPicThread(QThread):
    def __init__(self, savePath, folder, filename):
        super().__init__()
        self.save_path = savePath
        self.folder = folder
        self.filename = filename
        self.signal = signal

    def run(self):
        treshold = 200000
        file_prefix = self.filename.split('/')[-1].split('.')[0]
        size = 1000
        reader = kr.reader()
        scale = 40
        print(os.path.join(self.folder, file_prefix + '.kfb'))
        kr.reader.ReadInfo(reader, os.path.join(self.folder, file_prefix + '.kfb'), scale, False)
        reader.setReadScale(scale)
        width, Height = reader.getWidth(), reader.getHeight()
        num_h, num_w = Height // size, width // size
        for i in trange(num_w):
            self.signal.sendProgressBar(num_w)
            for j in range(num_h):
                img = reader.ReadRoi(i * size, j * size, size, size, scale)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                if cut.filtercut(gray) > treshold and img.shape[0] == 1000 and img.shape[1] == 1000:  # 切图
                    save_name = file_prefix + '_' + 'annoname' + '_total_' + str(i * j) + ".jpg"
                    cv2.imwrite(os.path.join(self.save_path, save_name), img)
                    self.signal.sendSaveFileSignal(save_name)


class interfaceThread(QThread):
    def __init__(self, cfg_file, best_epoch, test_dir):
        super().__init__()
        self.cfg_file = cfg_file
        self.best_epoch = best_epoch
        self.test_dir = test_dir
        self.signal = signal

    def run(self):
        model = init_model(self.cfg_file, self.best_epoch)
        files = os.listdir(self.test_dir)
        pred_class = []
        for f in tqdm(files):
            img = os.path.join(self.test_dir, f)
            res = inference_model(model, img)
            pred_class.append(res["pred_class"])
            self.signal.sendProgressBarIF(len(files))
            print(
                "{:s}: label {:d} [conf: {:.3f}]:class {:s}".format(f, res["pred_label"], res["pred_score"],
                                                                    res["pred_class"]))
        result_kfb = list(Counter(pred_class).keys())[0]
        self.signal.sendInterfaceRes(result_kfb)


class removeFilesThread(QThread):
    def __init__(self, filePath):
        super().__init__()
        self.filePath = filePath

    def run(self):
        listDir = os.listdir(self.filePath)
        for file in listDir:
            if os.path.isfile(file):
                os.remove(self.filePath + '/' + file)
        print("finish")


class MainWindow(QMainWindow, Ui_MainWindow):
    signal = signal

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.folder = "./dataset/kfb"
        self.save_path = './dataset/tis'
        self.cfg_file = "./models/CancerMedice3060.py"
        self.best_epoch = './models/epoch_25.pth'
        self.label_AI.setPixmap(QPixmap("./img/pic.png"))
        self.label_info.setText("请选择图片...")
        self.removeFilesThread = removeFilesThread(self.save_path)
        self.progressBar.setValue(0)
        self.pushButton_loadPic.clicked.connect(self.clickedLoadImg)
        self.signal.signalSaveFile.connect(self.signalSaveFileCall)
        self.signal.signalProgressBar.connect(self.load_progress_bar)
        self.listWidget.itemDoubleClicked.connect(self.showListItem)
        self.pushButton_findResult.clicked.connect(self.clickedFindResult)
        self.pushButton_reset.clicked.connect(self.clickedReset)
        self.signal.signalProgressBarIF.connect(self.load_progress_bar)
        self.signal.signalIfRes.connect(self.showResult)

    def showResult(self, res):
        self.label_info.setText("分析图片完成： ")
        self.label_result.setText(res)
    def clickedReset(self):
        self.label_info.clear()
        self.label_result.clear()
        self.label_showCutPic.clear()
        self.progressBar.setValue(0)
        self.listWidget.clear()
        self.removeFilesThread.start()

    def clickedFindResult(self):
        self.progressBar.setValue(0)
        self.label_result.clear()
        self.label_info.setText("分析图片中...")
        self.threadRes = interfaceThread(self.cfg_file, self.best_epoch, self.save_path)
        self.threadRes.start()

    def load_progress_bar(self, numDir):
        stepNum = ceil(100 / numDir)
        self.progressBar.setValue(self.progressBar.value() + stepNum)
        if self.progressBar.value() >= 100:
            print("finish")

    def signalSaveFileCall(self, fileName):
        picName = fileName.split('/')[-1]
        self.listWidget.addItem(QListWidgetItem(picName))
        print("signal1 emit : " + fileName)

    def clickedLoadImg(self):
        self.progressBar.setValue(0)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        filename = QFileDialog.getOpenFileNames(self, '选择图像', self.folder, "Text Files(*.kfb)")[0]
        if len(filename)==0:
            return
        filename=filename[0]
        self.label_info.setText("切割图片中...")
        self.threadLoadImg = cutPicThread(self.save_path, self.folder, filename)
        self.threadLoadImg.start()
        self.threadLoadImg.finished.connect(lambda: self.label_info.setText("切割图片完成： "))


    def showListItem(self, item):
        self.label_showCutPic.setPixmap(QPixmap("./dataset/tis/" + item.text()))
        print("datasets/tis/" + item.text())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())
