"""
Grid map library in python
based on the work of Atsushi Sakai
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
from scipy import ndimage
from scipy.interpolate.fitpack2 import RectBivariateSpline
from matplotlib import cm
from dhflocalization.gridmap.processpgm import PgmProcesser
from dhflocalization.rawdata.yamlhandler import YamlReader


class GridMap:
    """
    GridMap class
    """

    def __init__(self, raw_map_data, resolution, center_x, center_y):
        self.resolution = resolution

        # apply symmetric padding
        self.padding = int(0.1 * raw_map_data.shape[0])  # grid cell
        self.width = raw_map_data.shape[0] + 2 * self.padding  # grid cell
        self.height = raw_map_data.shape[1] + 2 * self.padding  # grid cell

        # place original map in the center of the padded one
        padded_map_data = np.zeros((self.width, self.height))
        padded_map_data[
            self.padding : -self.padding, self.padding : -self.padding
        ] = raw_map_data

        self.left_lower_x = center_x - self.padding * self.resolution  # meter
        self.left_lower_y = center_y - self.padding * self.resolution  # meter

        self.ndata = self.width * self.height
        self.data = list(np.flipud(padded_map_data).flatten())

        self.distance_transform = None
        self.distance_transform_interp = None
        self.distance_transform_dx = None
        self.distance_transform_dy = None
        # how many times finer grid for derivative discretization
        # than the map grid
        self.dt_derivative_resolution = 1

        self._init_distance_transform()

    @classmethod
    def load_map_from_config(cls, config_filename):
        config_path = "../resources/maps/" + config_filename + ".yaml"
        map_config = YamlReader.read(config_path)

        raw_map_data = PgmProcesser.read_pgm(map_config["image"])
        resolution = map_config["resolution"]  # meter/cell
        center_x = map_config["origin"][0]
        center_y = map_config["origin"][1]

        return cls(raw_map_data, resolution, center_x, center_y)

    def _init_distance_transform(self):
        self.distance_transform = self.calc_distance_transform()

        self.distance_transform_interp = RectBivariateSpline(
            np.arange(self.width) * self.resolution,
            np.arange(self.height) * self.resolution,
            self.distance_transform * self.resolution,
        )
        (
            distance_transform_dx,
            distance_transform_dy,
            step_size,
        ) = self.discretize_dt_derivative()

        self.distance_transform_dx = distance_transform_dx
        self.distance_transform_dy = distance_transform_dy
        self.dt_derivative_stepsize = step_size  # TODO might not needed

    def discretize_dt_derivative(self):
        stepnum = self.width * self.dt_derivative_resolution
        gridx, step_size = np.linspace(
            0, self.width * self.resolution, stepnum, retstep=True
        )
        gridy = np.linspace(0, self.height * self.resolution, stepnum)

        dx = self.distance_transform_interp(gridy, gridx, 0, 1)
        dy = self.distance_transform_interp(gridy, gridx, 1, 0)

        return dx, dy, step_size

    def get_value_from_xy_index(self, x_ind, y_ind):
        """get_value_from_xy_index
        when the index is out of grid map area, return None
        :param x_ind: x index
        :param y_ind: y index
        """

        grid_ind = self.calc_grid_index_from_xy_index(x_ind, y_ind)

        if 0 <= grid_ind < self.ndata:
            return self.data[grid_ind]
        else:
            return None

    def get_xy_index_from_xy_pos(self, x_pos, y_pos):
        """get_xy_index_from_xy_pos
        returns the closest cell, if position is out of boundary
        :param x_pos: x position [m]
        :param y_pos: y position [m]
        """
        x_ind = self.calc_xy_index_from_position(x_pos, self.left_lower_x, self.width)
        y_ind = self.calc_xy_index_from_position(y_pos, self.left_lower_y, self.height)

        return x_ind, y_ind

    def calc_xy_index_from_position(self, pos, lower_pos, array_size):
        # returns the closest cell, if pos is out of boundary

        index = np.floor((pos - lower_pos) / self.resolution)
        if type(pos) is np.ndarray:
            # position is less than lower_pos
            index[index < 0] = 0
            index[index >= array_size] = array_size - 1
            return index.astype(int)
        else:
            index = 0 if index < 0 else index
            index = array_size - 1 if index >= array_size else index
            return int(index)

    def set_value_from_xy_pos(self, x_pos, y_pos, val):
        """set_value_from_xy_pos
        return bool flag, which means setting value is succeeded or not
        :param x_pos: x position [m]
        :param y_pos: y position [m]
        :param val: grid value
        """

        x_ind, y_ind = self.get_xy_index_from_xy_pos(x_pos, y_pos)

        if (not x_ind) or (not y_ind):
            return False  # NG

        flag = self.set_value_from_xy_index(x_ind, y_ind, val)

        return flag

    def set_value_from_xy_index(self, x_ind, y_ind, val):
        """set_value_from_xy_index
        return bool flag, which means setting value is succeeded or not
        :param x_ind: x index
        :param y_ind: y index
        :param val: grid value
        """

        if (x_ind is None) or (y_ind is None):
            return False

        grid_ind = int(y_ind * self.width + x_ind)

        if 0 <= grid_ind < self.ndata:
            self.data[grid_ind] = val
            return True  # OK
        else:
            return False  # NG

    def calc_grid_index_from_xy_index(self, x_ind, y_ind):
        grid_ind = int(y_ind * self.width + x_ind)
        return grid_ind

    def calc_grid_central_xy_position_from_xy_index(self, x_ind, y_ind):
        x_pos = self.calc_grid_central_xy_position_from_index(x_ind, self.left_lower_x)
        y_pos = self.calc_grid_central_xy_position_from_index(y_ind, self.left_lower_y)

        return x_pos, y_pos

    def calc_grid_central_xy_position_from_index(self, index, lower_pos):
        return lower_pos + index * self.resolution + self.resolution / 2.0

    def calc_distance_transform(self):
        grid_data = np.reshape(np.array(self.data), (self.height, self.width))
        edt = ndimage.distance_transform_edt(1 - grid_data)
        return edt

    def calc_distance_transform_xy_pos(self, xy):
        edt_interp = self.distance_transform_interp
        x = xy[:, 0]
        y = xy[:, 1]
        # zero at the middle of the cell
        return edt_interp.ev(
            y - self.left_lower_y - self.resolution / 2.0,
            x - self.left_lower_x - self.resolution / 2.0,
        )

    def calc_distance_function_derivate_interp(self, xy_pos):
        """xy is in the coord system of the map
        (shifted, so the origin is around the middle of the map).
        However, the DT of the map has the origin in the bottom left corner.
        So the transformation of xy is needed:
        x: [-10,9.2) -> [0,19.2)
        y: [-10.05,9.15) -> [0,19.2).
        Also, the middle of the cell is considered, instead of the bottom left corner.
        """

        x_transf = xy_pos[:, 0] - self.left_lower_x - self.resolution / 2.0
        y_transf = xy_pos[:, 1] - self.left_lower_y - self.resolution / 2.0

        rounded = np.round(
            (np.array([x_transf, y_transf]).T / self.dt_derivative_stepsize)
        )
        rounded_int = rounded.astype(int)

        df_d_x = self.distance_transform_dx[rounded_int[:, 1], rounded_int[:, 0]]
        df_d_y = self.distance_transform_dy[rounded_int[:, 1], rounded_int[:, 0]]
        return df_d_x, df_d_y

    def check_occupied_from_xy_index(self, xind, yind, occupied_val=1.0):
        val = self.get_value_from_xy_index(xind, yind)

        if val is None or val >= occupied_val:
            return True
        else:
            return False

    def plot_grid_map(self, ax=None, zorder=1):
        grid_data = np.reshape(np.array(self.data), (self.height, self.width))

        # plot tick labels in meters, so that (0,0) is at the origin of the map
        extent = [
            self.left_lower_x,
            self.left_lower_x + self.width * self.resolution,
            self.left_lower_y,
            self.left_lower_y + self.height * self.resolution,
        ]
        if not ax:
            fig, ax = plt.subplots()
        ax.imshow(
            grid_data,
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
            zorder=zorder,
            origin="lower",
            extent=extent,
        )
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(AutoMinorLocator(20))
        ax.yaxis.set_minor_locator(AutoMinorLocator(20))
        # hide ticks
        ax.tick_params(which="minor", bottom=False, left=False)
