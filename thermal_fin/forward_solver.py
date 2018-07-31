from matplotlib import tri, cm
import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt
from numpy.random import rand
import tensorflow as tf

class ForwardSolver:
    def __init__(self, grid_x, grid_y, batch_size):
        x = np.linspace(-3.0, 3.0, grid_x)
        y = np.linspace(0.0, 4.0, grid_y)
        self.xx, self.yy = np.meshgrid(x, y)
        self.Aq_s, self.Fh, self.nodes, coor, theta_tri = self.load_FEM()
        self.triangulation = tri.Triangulation(coor[:, 0], coor[:, 1], theta_tri)
        self.batch_size = batch_size
        self.grid_x = grid_x
        self.grid_y = grid_y

        #  Aq_tf_s = []
        #  for Aq in self.Aq_s:
            #  indices = np.mat([Aq.row, Aq.col]).transpose()
            #  Aq_tf_s.append(tf.SparseTensor(indices, Aq.data, Aq.shape))

        #  self.Aq_s_tf = tf.stack(Aq_tf_s)

    def solve(self, params):
        '''
        Performs a forward solve with the given parameters and returns
        a 2D matrix representing the temperature distribution of a thermal fin
        with values interpolated from a finite element grid

        Arguments:
            params: Array of conductivities [k1, k2, k3, k4, Biot, k5]

        Returns
            theta : Temperature distribution on a grid, x ∈ [-3, 3] and y ∈ [0, 4]  
        '''

        Ah = coo_matrix((self.nodes, self.nodes))
        for param, Aq in zip(params, self.Aq_s):
            Ah = Ah + param * Aq

        uh = spsolve(Ah, self.Fh)
        interpolator = tri.CubicTriInterpolator(self.triangulation, uh)
        uh_interpolated = interpolator(self.xx, self.yy)

        return np.ma.fix_invalid(uh_interpolated, fill_value = 0.0).data

    def solve_noiterp(self, params):
        '''
        Performs a forward solve with the given parameters and returns
        a vector of nodal values of a finite element discretization of 
        the temperature distribution of a thermal fin

        Arguments:
            params: Array of conductivities [k1, k2, k3, k4, Biot, k5]

        Returns
            theta : Temperature distribution on a grid, x ∈ [-3, 3] and y ∈ [0, 4]  
        '''
        Ah = coo_matrix((self.nodes, self.nodes))
        for param, Aq in zip(params, self.Aq_s):
            Ah = Ah + param * Aq

        uh = spsolve(Ah, self.Fh)
        return uh

    #  def tf_solve(self, params):
        #  #  Ah = coo_matrix((self.nodes, self.nodes))
        #  #  for param, Aq in zip(params, self.Aq_s):
            #  #  Ah = Ah + param * Aq

        #  Ah = tf.reduce_sum(tf.multiply(tf.expand_dims(params), self.Aq_s_tf), axis=0)
        #  #  indices = np.mat([Ah.row, Ah.col]).transpose()
        #  #  Ah_tf = tf.SparseTensor(indices, Ah.data, Ah.shape)
        #  return 2.0

    def train_input_fn(self):
        fin_params = np.zeros((self.batch_size, 6))
        uh_s = np.zeros((self.batch_size, self.grid_x, self.grid_y))
        for i in range(self.batch_size):
            fin_params[i,:] = [rand()*8, rand()*8, rand()*8, rand()*8, 1, rand()*2]
            uh_s[i,:,:] = self.solve(fin_params[i,:])
        return ({'x':tf.convert_to_tensor(uh_s)}, tf.convert_to_tensor(fin_params))

    def eval_input_fn(self):
        fin_params = [rand()*8, rand()*8, rand()*8, rand()*8, 1, rand()*2]
        uh = self.solve(fin_params)
        return ({'x':tf.convert_to_tensor(uh, dtype=tf.float64)}, 
                tf.convert_to_tensor(fin_params, dtype=tf.float64))

    def load_FEM(self):
        '''
        Loads the FEM matrices in sparse format for the forward solve
        Only 6 RHS matrices corresponding to the parameters loaded currently.
        Data generated from MATLAB, so the indices need to be subtracted by 1.

        Returns:
            Aq_s : Array of Aq sparse matrices each with dimension (nodes,nodes)
            Fh   : Load vector for FEM with dimension (nodes,1)
            nodes: Number of FEM nodes 
            coor: coordinates for plottting
            theta_tri: coordinate indices for the triangulation with dimension (triangles, 3)
        '''

        Aq_s = []

        data_dir = 'matlab_data/'
        Fh = np.loadtxt(data_dir + 'Fh.csv')
        nodes = Fh.shape[0]
        coor = np.loadtxt(data_dir + 'coarse_coor.csv', delimiter=',')
        theta_tri = np.loadtxt(data_dir + 'theta_tri.csv',
                               delimiter=",", unpack=True)

        for i in range(1, 7):
            col, row, value = np.loadtxt(
                data_dir + 'Aq' + str(i) + '.csv', delimiter="\t", unpack=True)
            Aq = coo_matrix((value, (row-1, col-1)), shape=(nodes, nodes))
            Aq_s.append(Aq)

        return Aq_s, Fh, nodes, coor, (theta_tri-1).T

    def plot_solution(self, uh_interpolated, filepath):
        pl = plt.pcolormesh(self.xx, self.yy, uh_interpolated, linewidth=0.0, rasterized=True)
        pl.set_edgecolor('face')
        plt.savefig(filepath, dpi=400)
        #  plt.tripcolor(self.triangulation, uh)
