from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, \
    QVBoxLayout, QWidget, QGridLayout
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import sys
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import cm
import matplotlib


class HSlider(QWidget):
    def __init__(self, minimum, maximum, parent=None):
        super(HSlider, self).__init__(parent=parent)
        self.horizontalLayout = QHBoxLayout(self)
        self.label = QLabel(self)
        self.horizontalLayout.addWidget(self.label)
        self.verticalLayout = QVBoxLayout()
        spacerItem = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem)
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Horizontal)
        self.verticalLayout.addWidget(self.slider)
        spacerItem1 = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem1)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.resize(self.sizeHint())

        self.minimum = minimum
        self.maximum = maximum
        self.slider.valueChanged.connect(self.setLabelValue)
        self.x = None
        self.setLabelValue(self.slider.value())

    def setLabelValue(self, value):
        self.x = self.minimum + (float(value) / (self.slider.maximum() - self.slider.minimum())) * (
        self.maximum - self.minimum)
        self.label.setText("{0:.2f}".format(self.x))

class VSlider(QWidget):
    def __init__(self, minimum, maximum, parent=None):
        super(VSlider, self).__init__(parent=parent)
        self.verticalLayout = QVBoxLayout(self)
        self.label = QLabel(self)
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout = QHBoxLayout()
        spacerItem = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Vertical)
        self.horizontalLayout.addWidget(self.slider)
        spacerItem1 = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.resize(self.sizeHint())

        self.minimum = minimum
        self.maximum = maximum
        self.slider.valueChanged.connect(self.setLabelValue)
        self.x = None
        self.setLabelValue(self.slider.value())

    def setLabelValue(self, value):
        self.x = self.minimum + (float(value) / (self.slider.maximum() - self.slider.minimum())) * (
        self.maximum - self.minimum)
        self.label.setText("{0:.2f}".format(self.x))

class Widget(QWidget):
    def __init__(self, parent=None):
        super(Widget, self).__init__(parent=parent)
        self.widgetlayout = QGridLayout(self)

        ## define covariance matrices
        cov_inv = np.diag([2., 2., 2.])
        cov_inv[0,1] = -1
        cov_inv[1,2] = -1
        cov_inv[1,0] = -1
        cov_inv[2,1] = -1
        self.cov = np.linalg.inv(cov_inv)
        print(self.cov)
        print(cov_inv)
        # for conditionals
        grid = np.linspace(-3, 3, 51)
        g1, g2 = np.meshgrid(grid, grid)
        self.grid = np.stack((g1.flatten(), g2.flatten()))

        ## add plots
        self.create_3d_plot()
        self.widgetlayout.addWidget(self.glvw, 0,1,9,8)

        self.p1 = pg.PlotWidget()
        self.p1.hideAxis('left')
        self.p1.hideAxis('bottom')
        self.im1 = pg.ImageItem()
        # Get the colormap
        colormap = matplotlib.colormaps["jet"] # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt

        self.im1.setLookupTable(lut)
        self.p1.addItem(self.im1)

        self.widgetlayout.addWidget(self.p1, 0,9,3,8)

        self.p2 = pg.PlotWidget()
        self.p2.hideAxis('left')
        self.p2.hideAxis('bottom')
        self.im2 = pg.ImageItem()
        self.im2.setLookupTable(lut)
        self.p2.addItem(self.im2)

        self.widgetlayout.addWidget(self.p2, 3,9,3,8)

        self.p3 = pg.PlotWidget()
        self.p3.hideAxis('left')
        self.p3.hideAxis('bottom')
        self.im3 = pg.ImageItem()
        self.im3.setLookupTable(lut)
        self.p3.addItem(self.im3)

        self.widgetlayout.addWidget(self.p3, 6,9,3,8)


        self.p1.sizeHint = lambda: pg.QtCore.QSize(900, 450)
        self.p2.sizeHint = lambda: pg.QtCore.QSize(900, 450)
        self.p3.sizeHint = lambda: pg.QtCore.QSize(900, 450)

        self.glvw.sizeHint = lambda: pg.QtCore.QSize(1200, 900)
        self.glvw.setSizePolicy(self.p1.sizePolicy())

        self.slider1 = VSlider(-3, 3)
        self.widgetlayout.addWidget(self.slider1, 0,0,8,1)

        self.slider2 = HSlider(-3, 3)
        self.slider2.slider.setValue(100)
        self.widgetlayout.addWidget(self.slider2, 9,1,1,8)

        self.slider3 = HSlider(-3, 3)
        #self.slider3.slider.setValue(100)
        self.widgetlayout.addWidget(self.slider3, 10,1,1,8)


        self.slider1.slider.valueChanged.connect(self.update_slices)
        self.slider2.slider.valueChanged.connect(self.update_slices)
        self.slider3.slider.valueChanged.connect(self.update_slices)
        self.slider1.slider.valueChanged.connect(self.update_conditional)
        self.slider2.slider.valueChanged.connect(self.update_conditional)
        self.slider3.slider.valueChanged.connect(self.update_conditional)


        self.x_prev = -3.
        self.y_prev = 3.
        self.z_prev = -3.



    def create_3d_plot(self):
        self.glvw= gl.GLViewWidget()
        self.glvw.setWindowTitle('3D Gaussian viewer')
        self.glvw.setCameraPosition(distance=15)

        ######################
        # Create grids
        ######################
        gxy = gl.GLGridItem()
        gxy.setSize(x=6., y=6., z=6.)
        gxy.rotate(180,0,0,1)
        gxy.translate(0,0,-3)
        self.glvw.addItem(gxy)

        gxz = gl.GLGridItem()
        gxz.setSize(x=6., y=6., z=6.)
        gxz.rotate(90, 1, 0, 0)
        gxz.translate(0,3,0)
        self.glvw.addItem(gxz)

        gzy = gl.GLGridItem()
        gzy.setSize(x=6., y=6., z=6.)
        gzy.rotate(90,0,1,0)
        gzy.translate(-3,0,0)
        self.glvw.addItem(gzy)


        ##############################
        # Plot Gaussian point cloud
        ##############################

        v, w = np.linalg.eig(self.cov)
        samples = multivariate_normal.rvs(cov=self.cov, size=20000)
        plt = gl.GLScatterPlotItem(pos=samples, color=(0.5, 0.5, 0.5, 0.7), size=2.0)
        self.glvw.addItem(plt)

        #########################
        # Draw eigenvector lines
        #########################
        v1 = np.array([[0., 0., 0.], 2*np.sqrt(v[0]) * w[:,0]])
        plt = gl.GLLinePlotItem(pos=v1, color=(1,0,0,1), width=3.)
        self.glvw.addItem(plt)

        v2 = np.array([[0., 0., 0.], 2*np.sqrt(v[1]) * w[:,1]])
        plt = gl.GLLinePlotItem(pos=v2, color=(0,1,0,1), width=3.)
        self.glvw.addItem(plt)

        v3 = np.array([[0., 0., 0.], 2*np.sqrt(v[2]) * w[:,2]])
        plt = gl.GLLinePlotItem(pos=v3, color=(0,0,1,1), width=3.)
        self.glvw.addItem(plt)

        #######################################
        # Draw lines indicating x, y, z axes
        #######################################
        xline = np.array([[0., -3., -3.],
                          [1.5, -3., -3.]])
        plt = gl.GLLinePlotItem(pos=xline, color=(1,0,0,1), width=2.)
        self.glvw.addItem(plt)

        yline = np.array([[0., -3., -3.],
                          [0., -1.5, -3.]])
        plt = gl.GLLinePlotItem(pos=yline, color=(0,0,1,1), width=2.)
        self.glvw.addItem(plt)

        zline = np.array([[0., -3., -3.],
                          [0., -3., -1.5]])
        plt = gl.GLLinePlotItem(pos=zline, color=(0,1,0,1), width=2.)
        self.glvw.addItem(plt)

        ##########################################
        # Draw intersecting planes
        ##########################################


        verts = np.array([[-3., -3., -3.],
                          [-3.,  3., -3.],
                          [ 3.,  3., -3.],
                          [ 3., -3., -3.]])
        faces = np.array([[0, 1, 2],
                          [0, 2, 3]])
        self.zmeshData = gl.MeshData(vertexes=verts, faces=faces)
        self.zPlane = gl.GLMeshItem(meshdata=self.zmeshData, color=(0,1,0,0.0))
        self.zPlane.setGLOptions('additive')
        self.glvw.addItem(self.zPlane)

        verts = np.array([[-3.,  3., -3.],
                          [-3.,  3.,  3.],
                          [ 3.,  3.,  3.],
                          [ 3.,  3., -3.]])
        faces = np.array([[0, 1, 2],
                          [0, 2, 3]])
        self.ymeshData = gl.MeshData(vertexes=verts, faces=faces)
        self.yPlane = gl.GLMeshItem(meshdata=self.ymeshData, color=(0,0,1,0.0))
        self.yPlane.setGLOptions('additive')
        self.glvw.addItem(self.yPlane)

        verts = np.array([[-3., -3., -3.],
                          [-3., -3.,  3.],
                          [-3.,  3.,  3.],
                          [-3.,  3., -3.]])

        faces = np.array([[0, 1, 2],
                          [0, 2, 3]])
        self.xmeshData = gl.MeshData(vertexes=verts, faces=faces)
        self.xPlane = gl.GLMeshItem(meshdata=self.xmeshData, color=(1,0,0,0.0))
        self.xPlane.setGLOptions('additive')
        self.glvw.addItem(self.xPlane)

        #### REMOVE AFTER
        # xline = np.array([[-3., 1., 1.],
        #                   [ 3., 1., 1.]])
        # plt = gl.GLLinePlotItem(pos=xline, color=(1,0,0,1), width=10.)
        # self.glvw.addItem(plt)

    def update_conditional(self):
        y_cond_mean = self.cov[::2,1] / self.cov[1,1] * self.slider2.x
        y_cond_cov = self.cov[::2,::2] - self.cov[::2,1:2].dot(self.cov[::2,1:2].T) / self.cov[1,1]
        pdf = multivariate_normal.pdf(self.grid.T, mean=y_cond_mean, cov=y_cond_cov)
        self.im1.setImage(pdf.reshape((51,51)))

        z_cond_mean = self.cov[:2,2] / self.cov[2,2] * self.slider1.x
        z_cond_cov = self.cov[:2,:2] - self.cov[:2,2:].dot(self.cov[:2,2:].T) / self.cov[2,2]
        pdf = multivariate_normal.pdf(self.grid.T, mean=z_cond_mean, cov=z_cond_cov)
        self.im2.setImage(pdf.reshape((51,51)))

        x_cond_mean = self.cov[1:,0] / self.cov[0,0] * self.slider3.x
        x_cond_cov = self.cov[1:,1:] - self.cov[1:,:1].dot(self.cov[1:,:1].T) / self.cov[0,0]
        pdf = multivariate_normal.pdf(self.grid.T, mean=x_cond_mean, cov=x_cond_cov)
        self.im3.setImage(pdf.reshape((51,51)))

    def update_slices(self):
        s1 = self.slider1.x
        s2 = self.slider2.x
        s3 = self.slider3.x

        if s1 != self.z_prev:
            if s1 < -2.8:
                self.zPlane.setColor((0.,1.,0.,0.))
            else:
                verts = np.array([[-3., -3., s1],
                                  [-3.,  3., s1],
                                  [ 3.,  3., s1],
                                  [ 3., -3., s1]])
                self.zmeshData.setVertexes(verts)
                self.zPlane.setColor((0.,1.,0.,0.2))
                self.zPlane.meshDataChanged()
                self.z_prev = s1



        if s2 != self.y_prev:
            if s2 > 2.8:
                self.yPlane.setColor((0.,0.,1.,0.))
            else:
                verts = np.array([[-3., s2, -3.],
                                  [-3., s2,  3.],
                                  [ 3., s2,  3.],
                                  [ 3., s2, -3.]])
                self.ymeshData.setVertexes(verts)

                self.yPlane.setColor((0.,0.,1.,0.2))
                self.yPlane.meshDataChanged()
                self.y_prev = s2

        if s3 != self.x_prev:
            if s3 < -2.8:
                self.xPlane.setColor((1.,0.,0.,0.))
            else:
                verts = np.array([[s3, -3., -3.],
                                  [s3, -3.,  3.],
                                  [s3,  3.,  3.],
                                  [s3,  3., -3.]])

                self.xmeshData.setVertexes(verts)
                self.xPlane.setColor((1.,0.,0.,0.2))
                self.xPlane.meshDataChanged()
                self.x_prev = s3

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())
