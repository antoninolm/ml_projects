'''
import numpy as np
import matplotlib.pyplot as plt


def make_meshgrid(ax, h=.02):
    # x_min, x_max = x.min() - 1, x.max() + 1
    # y_min, y_max = y.min() - 1, y.max() + 1
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def draw_boundary(ax, clf):

    xx, yy = make_meshgrid(ax)
    return plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.5)
'''
import numpy as np
import matplotlib.pyplot as plt

def make_meshgrid(ax, h=0.02, pad=0.25):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xx, yy = np.meshgrid(
        np.arange(x_min - pad, x_max + pad, h),
        np.arange(y_min - pad, y_max + pad, h)
    )
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Ensure two regions for binary classes 0/1
    out = ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], **params)
    # Draw the decision boundary line at 0.5
    ax.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=1)
    return out

def draw_boundary(ax, clf, h=0.02, pad=0.25):
    # IMPORTANT: call this AFTER your scatter, or set ax limits before.
    xx, yy = make_meshgrid(ax, h=h, pad=pad)
    return plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.3)

