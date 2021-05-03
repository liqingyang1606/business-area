import numpy as np
import time


# convert "order_01.txt" into "paired_01.txt"
# turn to function process() for more information
def poi_in_axis():
	"""use 2 arrays as the 2 axis"""
    index_y = np.zeros(400)
    index_x = np.zeros(300)
    with open("..\\poi_3915.txt", "r") as file_object:
        lines = file_object.readlines()
    for line in lines:
        sp = line.rstrip().split(", ")
        i = int(float(sp[0]) * 2)
        j = int(float(sp[1]) * 2)
        lon = float(sp[2])
        lat = float(sp[3])
        if index_y[i] == 0.0 or index_y[i] > lat:
            index_y[i] = lat
        if index_x[j] == 0.0 or index_x[j] > lon:
            index_x[j] = lon
    return index_y, index_x


class RecordGrid:
    def __init__(self):
        self.ry, self.rx = poi_in_axis()
        self.rgy = 400
        self.rgx = 300

    def find_gridy(self, lat):
        """find the nearest y_grid for current location"""
        for i in range(0, self.rgy):
            if self.ry[i] > lat:
                if self.ry[i - 1] != 0:
                    return (i - 1) / 2
                j = i - 1
                start = self.ry[i] - 0.005
                while self.ry[j] == 0:
                    if start < lat:
                        self.ry[j] = start
                        return j / 2
                    j -= 1
                    start -= 0.005
                return j / 2
        return 0

    def find_gridx(self, lon):
        """find the nearest x_grid for current location"""
        for i in range(0, self.rgx):
            if self.rx[i] > lon:
                if self.rx[i - 1] != 0:
                    return (i - 1) / 2
                j = i - 1
                value = self.rx[i] - 0.005
                while self.rx[j] == 0:
                    if value < lon:
                        self.rx[j] = value
                        return j / 2
                    j -= 1
                    value -= 0.005
                return j / 2
        return 0


def process_time(t):
	"""extract day, hour and minute"""
    time_local = time.localtime(int(t))
    return time_local.tm_mday, time_local.tm_hour, time_local.tm_min


def process(line, rg):
	"""processing one line of trajectory"""
    spl = line.rstrip().split(",")
    driver_id = spl[0]
    d, uh, um = process_time(spl[1])
    d, oh, om = process_time(spl[2])
    times = ",%d,%d,%d,%d,%d" % (d, uh, um, oh, om)
    # date, pickup_h, pickup_m, dropoff_h, dropoff_m
    lon = float(spl[3])
    lat = float(spl[4])
    uy = rg.find_gridy(lat)
    ux = rg.find_gridx(lon)
    locate_up = ",%f,%f,%.1f,%.1f" % (lon, lat, uy, ux)
    # pickup_lon, pickup_lat, pickup_y, pickup_x
    lon = float(spl[5])
    lat = float(spl[6])
    oy = rg.find_gridy(lat)
    ox = rg.find_gridx(lon)
    locate_off = ",%f,%f,%.1f,%.1f" % (lon, lat, oy, ox)
    # dropoff_lon, dropoff_lat, dropoff_y, dropoff_x
    return driver_id + times + locate_up + locate_off


def main():
    """preprocess trajectory data"""
    r = RecordGrid()
    # Change the path below:
    with open("..\\data_chengdu\\order_01.txt", "r") as file_object:
        lines = file_object.readlines()
    with open("paired_01.txt", "w") as file_object:
        i = 0
        for line in lines:
            nl = process(line, r)
            file_object.write(nl + "\n")
            i += 1
            if i % 1000 == 1:
                print("# Written %d lines." % i)


if __name__ == '__main__':
	main()
