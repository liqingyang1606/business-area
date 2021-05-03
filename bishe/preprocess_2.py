import numpy as np


def time_within(he, me, h, m):
	"""examine whether (h,m) is within the time window"""
    if h == he and me <= m < me+10:
        return True
    else:
        return False


def parse_line_2(str_object, he, me, mat):
	"""parse one line, result stored in param mat"""
    spl = str_object.rstrip().split(",")
    h1 = int(spl[2])
    m1 = int(spl[3])
    if time_within(he, me, h1, m1):
        y = int(float(spl[8]) * 2)
        x = int(float(spl[9]) * 2)
        if y > 0 and x > 0:
            mat[y][x][1] += 1
    h2 = int(spl[4])
    m2 = int(spl[5])
    if time_within(he, me, h2, m2):
        y = int(float(spl[12]) * 2)
        x = int(float(spl[13]) * 2)
        if y > 0 and x > 0:
            mat[y][x][2] += 1


def handle_2(he, me, d):
	"""turn the (400,300) matrix into (40, 40)"""
    mat = np.zeros((400, 300, 3), dtype=int)
    file_name = "..\\data_paired\\paired_%02d.txt" % d
    with open(file_name, "r") as file_object:
        lines = file_object.readlines()
    for line in lines:
        parse_line_2(line, he, me, mat)
    for i in range(0, 400):
        for j in range(0, 300):
            mat[i][j][0] = mat[i][j][1] + mat[i][j][2]
    ret = mat[234:274, 140:180, :]
    #print(ret.shape)
    return ret


def main():
	"""generate input data, mat01.npy"""
    data = np.zeros((144, 40, 40, 3), dtype=int)
    index = 0
    day = 20
    for i in range(0, 24):
        for j in range(0, 60, 10):
            data[index] = handle_2(i, j, day)
            index += 1
        print("# Finish hour %02d" % i)
    path = "..\\data2\\mat%02d.npy" % day
    np.save(path, data)


def get_hot(access):
    if access == 0:
        return 0
    elif 0 < access <= 6:
        return 1
    elif 6 < access <= 12:
        return 2
    else:
        return 3


def parse_matrix(m):
    r = np.zeros((40, 40), dtype=int)
    for i in range(0, 40):
        for j in range(0, 40):
            r[i][j] = get_hot(m[i][j])
    return r


def main2(d):
	"""generate target data, tar01.npy"""
	res = np.zeros((144, 40, 40), dtype=int)
    i = 0
    inp = np.load("..\\data2\\mat%02d.npy" % d)
    for h in range(0, 24):
        for m in range(0, 60, 10):
            a = inp[i, :, :, 0]
            b = parse_matrix(a)
            res[i] = b
            i += 1
    np.save("..\\targets\\tar%02d.npy" % d, res)


if __name__ == '__main__':
    main()
