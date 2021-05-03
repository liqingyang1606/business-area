import numpy as np
import matplotlib.pyplot as plt


# remember to modify the paths
def get_matrix_out(d, h):
    path = "D:\\Projects\\model_lstm\\test2\\model4out_%02d%02d.npy" % (d, h)
    return np.load(path)[0]


def get_matrix_tar(d, h):
    path = "D:\\Projects\\model_lstm\\test2\\model4tar_%02d%02d.npy" % (d, h)
    return np.load(path)[0]


def draw_and_save(matrix, path):
	"""draw contour figures and save it to path"""
    x = range(0, 40)
    y = range(0, 40)
    plt.contourf(x, y, matrix)
    plt.savefig(path)
    plt.show()


def main():
    day = 1
    hour = 21
    o = get_matrix_out(day, hour)
    t = get_matrix_tar(day, hour)
    path_out = "..\\graphs\\model4out_%02d%02d.png" % (day, hour)
    path_tar = "..\\graphs\\model4tar_%02d%02d.png" % (day, hour)
    draw_and_save(o, path_out)
    draw_and_save(t, path_tar)


if __name__ == '__main__':
	main()
