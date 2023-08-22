"""Tools for analyzing tracking data files created by magnocube tracker.

Objects
-------
TrackingTrial
    Loads an h5 dataset representing a TrackingTrial from the magnocube library.
TrialsDataset
    Loads a whole set of TrackingTrial instances, allowing queries and
    statistical analyses along experimental parameters.
"""
import copy
import numpy as np
import numpy
numpy.float = np.float64
numpy.int = numpy.int_

import h5py                             # for handling h5 tracking datasets
import math
from matplotlib import pyplot as plt    # for plotting data
import matplotlib
import os
import pickle
import scipy
from scipy import optimize
import seaborn as sbn
import skimage
from skvideo import io
import sys
import pandas as pd

from functools import partial

# for the video analysis
# from ring_analysis import *
# for plotting

# use interactive plotting
# matplotlib.use('Qt5Agg')
# set ticks to the inside of the spines
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

blue, green, yellow, orange, red, purple = [
    (0.30, 0.45, 0.69), (0.33, 0.66, 0.41), (0.83, 0.74, 0.37),
    (0.78, 0.50, 0.16), (0.77, 0.31, 0.32), (0.44, 0.22, 0.78)]


class Kalman_Filter():
    '''
    2D Kalman filter, assuming constant acceleration.

    Use a 2D Kalman filter to return the estimated position of points given 
    linear prediction of position assuming (1) fixed jerk, (2) gaussian
    jerk noise, and (3) gaussian measurement noise.

    Parameters
    ----------
    num_objects : int
        Number of objects to model as well to expect from detector.
    sampling_interval : float
        Sampling interval in seconds. Should equal (frame rate) ** -1.
    jerk : float
        Jerk is modeled as normally distributed. This is the mean.
    jerk_std : float
        Jerk distribution standard deviation.
    measurement_noise_x : float
        Variance of the x component of measurement noise.
    measurement_noise_y : float
        Variance of the y component of measurement noise.

    '''
    def __init__(self, num_objects, num_frames=None, sampling_interval=30**-1,
                 jerk=0, jerk_std=125,
                 measurement_noise_x=5, measurement_noise_y=5,
                 width=None, height=None):
        self.width = width
        self.height = height
        self.num_objects = num_objects
        self.num_frames = num_frames
        self.sampling_interval = sampling_interval
        self.dt = self.sampling_interval
        self.jerk = jerk
        self.jerk_std = jerk_std
        self.measurement_noise_x = measurement_noise_x
        self.measurement_noise_y = measurement_noise_y
        self.tkn_x, self.tkn_y = self.measurement_noise_x, self.measurement_noise_y
        # process error covariance matrix
        self.Ez = np.array(
            [[self.tkn_x, 0         ],
             [0,          self.tkn_y]])
        # measurement error covariance matrix (constant jerk)
        self.Ex = np.array(
            [[self.dt**6/36, 0,             self.dt**5/12, 0,             self.dt**4/6, 0           ],
             [0,             self.dt**6/36, 0,             self.dt**5/12, 0,            self.dt**4/6],
             [self.dt**5/12, 0,             self.dt**4/4,  0,             self.dt**3/2, 0           ],
             [0,             self.dt**5/12, 0,             self.dt**4/4,  0,            self.dt**3/2],
             [self.dt**4/6,  0,             self.dt**3/2,  0,             self.dt**2,   0           ],
             [0,             self.dt**4/6,  0,             self.dt**3/2,  0,            self.dt**2  ]])
        self.Ex *= self.jerk_std**2
        # set initial position variance
        self.P = np.copy(self.Ex)
        ## define update equations in 2D as matrices - a physics based model for predicting
        # object motion
        ## we expect objects to be at:
        # [state update matrix (position + velocity)] + [input control (acceleration)]
        self.state_update_matrix = np.array(
            [[1, 0, self.dt, 0,       self.dt**2/2, 0           ],
             [0, 1, 0,       self.dt, 0,            self.dt**2/2],
             [0, 0, 1,       0,       self.dt,      0           ],
             [0, 0, 0,       1,       0,            self.dt     ],
             [0, 0, 0,       0,       1,            0           ],
             [0, 0, 0,       0,       0,            1           ]])
        self.control_matrix = np.array(
            [self.dt**3/6, self.dt**3/6, self.dt**2/2, self.dt**2/2, self.dt, self.dt])
        # measurement function to predict next measurement
        self.measurement_function = np.array(
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0]])
        self.A = self.state_update_matrix
        self.B = self.control_matrix
        self.C = self.measurement_function
        ## initialize result variables
        self.Q_local_measurement = []  # point detections
        ## initialize estimateion variables for two dimensions
        self.max_tracks = self.num_objects
        dimension = self.state_update_matrix.shape[0]
        self.Q_estimate = np.empty((dimension, self.max_tracks))
        self.Q_estimate.fill(np.nan)
        if self.num_frames is not None:
            self.Q_loc_estimateX = np.empty((self.num_frames, self.max_tracks))
            self.Q_loc_estimateX.fill(np.nan)
            self.Q_loc_estimateY = np.empty((self.num_frames, self.max_tracks))
            self.Q_loc_estimateY.fill(np.nan)
        else:
            self.Q_loc_estimateX = []
            self.Q_loc_estimateY = []
        self.num_tracks = self.num_objects
        self.num_detections = self.num_objects
        self.frame_num = 0

    def get_prediction(self):
        '''
        Get next predicted coordinates using current state and measurement information.

        Returns
        -------
        estimated points : ndarray
            approximated positions with shape.
        '''
        ## kalman filter
        # predict next state with last state and predicted motion
        self.Q_estimate = self.A @ self.Q_estimate + (self.B * self.jerk)[:, None]
        # predict next covariance
        self.P = self.A @ self.P @ self.A.T + self.Ex
        # Kalman Gain
        try:
            self.K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.Ez)
            ## now assign the detections to estimated track positions
            # make the distance (cost) matrix between all pairs; rows = tracks and
            # cols = detections
            self.estimate_points = self.Q_estimate[:2, :self.num_tracks]
            # np.clip(self.estimate_points[0], -self.height/2, self.height/2, out=self.estimate_points[0])
            # np.clip(self.estimate_points[1], -self.width/2, self.width/2, out=self.estimate_points[1])
            return self.estimate_points.T  # shape should be (num_objects, 2)
        except:
            return np.array([np.nan, np.nan])
    

    def add_starting_points(self, points):
        assert points.shape == (self.num_objects, 2), print("input array should have "
                                                           "shape (num_objects X 2)")
        self.Q_estimate.fill(0)
        self.Q_estimate[:2] = points.T
        if self.num_frames is not None:
            self.Q_loc_estimateX[self.frame_num] = self.Q_estimate[0]
            self.Q_loc_estimateY[self.frame_num] = self.Q_estimate[1]
        else:
            self.Q_loc_estimateX.append(self.Q_estimate[0])
            self.Q_loc_estimateY.append(self.Q_estimate[1])
        self.frame_num += 1

    def add_measurement(self, points):
        ## detections matrix
        assert points.shape == (self.num_objects, 2), print("input array should have "
                                                           "shape (num_objects X 2)")
        self.Q_loc_meas = points
        # find nans, exclude from the distance matrix
        # no_nans_meas = np.isnan(self.Q_loc_meas[:, :self.num_tracks]) == False
        # no_nans_meas = no_nans_meas.max(1)
        # assigned_measurements = np.empty((self.num_tracks, 2))
        # assigned_measurements.fill(np.nan)
        # self.est_dist = scipy.spatial.distance_matrix(
        #     self.estimate_points.T,
        #     self.Q_loc_meas[:self.num_tracks][no_nans_meas])
        # use hungarian algorithm to find best pairings between estimations and measurements
        # if not np.any(np.isnan(self.est_dist)):
        # try:
        #     asgn = scipy.optimize.linear_sum_assignment(self.est_dist)
        # except:
        #     print(self.est_dist)
        # for num, val in zip(asgn[0], asgn[1]):
        #     assigned_measurements[num] = self.Q_loc_meas[no_nans_meas][val]
        # remove problematic cases
        # close_enough = self.est_dist[asgn] < 25
        # no_nans = np.logical_not(np.isnan(assigned_measurements)).max(1)
        # good_cases = np.logical_and(close_enough, no_nans)
        # if self.width is not None:
        #     in_bounds_x = np.logical_and(
        #         assigned_measurements.T[1] > 0, 
        #         assigned_measurements.T[1] < self.width)
        # if self.height is not None:
        #     in_bounds_y = np.logical_and(
        #         assigned_measurements.T[0] > 0, 
        #         assigned_measurements.T[0] < self.height)
        # good_cases = no_nans
        # good_cases = no_nans * in_bounds_x * in_bounds_y
        # apply assignemts to the update
        # for num, (good, val) in enumerate(zip(good_cases, assigned_measurements)):
        #     if good:
        #         self.Q_estimate[:, num] = self.Q_estimate[:, num] + self.K @ (
        #             val.T - self.C @ self.Q_estimate[:, num])
        #         self.track_strikes[num] = 0
        #     else:
        #         self.track_strikes[num] += 1
        #         self.Q_estimate[2:, num] = 0
        self.Q_estimate = self.Q_estimate + self.K @ (self.Q_loc_meas - self.C @ self.Q_estimate)
        # update covariance estimation
        self.P = (np.eye((self.K @ self.C).shape[0]) - self.K @ self.C) @ self.P
        ## store data
        if self.num_frames is not None:
            self.Q_loc_estimateX[self.frame_num] = self.Q_estimate[0]
            self.Q_loc_estimateY[self.frame_num] = self.Q_estimate[1]
        else:
            self.Q_loc_estimateX.append(self.Q_estimate[0])
            self.Q_loc_estimateY.append(self.Q_estimate[1])
        self.frame_num += 1

    def update_vals(self, **kwargs):
        """Allow for replacing parameters like jerk_std and noise estimates."""
        for key, val in kwargs.items():
            self.__setattr__(key, val)


class KalmanAngle(Kalman_Filter):
    """A special instance of the Kalman Filter for single object, 1D data."""
    def __init__(self, **kwargs):
        super().__init__(num_objects=1, measurement_noise_y=0, width=0, **kwargs) 
        self.last_point = 0
        self.revolutions = 0
        self.record = []
        # print key parameters
        # print(f"jerk std={self.jerk_std}, noise std={self.measurement_noise_x}")

    def store(self, point):
        """Converts single point to appropriate shape for the 2D filter.
        
        Note: keep the point and last point variables in wrapped format and 
        unwrap just for when adding the measurement.
        """
        point = np.copy(point)
        if point is np.nan:
            point = self.last_point
        # unwrap
        if (self.last_point < -np.pi/2) and (point > np.pi/2):
            self.revolutions -= 1
        elif (self.last_point > np.pi/2) and (point < -np.pi/2):
            self.revolutions += 1
        self.last_point = np.copy(point)
        point += 2*np.pi*self.revolutions
        self.record += [point]
        # add 0 for second dimension
        point = np.array([point, 0])[np.newaxis]
        if self.frame_num == 0:
            self.add_starting_points(point)
        else:
            self.add_measurement(point)

    def predict(self):
        output = self.get_prediction()[0, 0]
        output %= 2*np.pi
        if output > np.pi:
            output -= 2*np.pi
        elif output < -np.pi:
            output += 2*np.pi
        return output

class TrackingVideo():
    def __init__(self, filename):
        """Load the video file and get metadata.


        Parameters
        ----------
        filename : str
            The path to the video file.
        """
        self.filename = filename
        self.subject = os.path.basename(filename).split(".")[0]
        # load video from matlab video
        if filename.endswith(".mat"):
            # self.video = io.vreader(self.filename)
            self.video = scipy.io.loadmat(self.filename)
            self.times = self.video['t_v'][:, 0]
            self.video = self.video['vidData'][:, :, 0] # height x width x channel x num_frames
            self.video = self.video.transpose(2, 0, 1)
        # or load a frame generator 
        else:
            # self.video = io.vread(self.filename, as_grey=True)[..., 0]
            # self.times = np.arange(len(self.video))
            input_dict = {'-hwaccel': 'cuda', '-hwaccel_output_format': 'cuda'}
            output_dict = {'-c:v': 'h264_nvenc'}
            self.video = io.FFmpegReader(self.filename, inputdict=input_dict, outputdict=output_dict)
            self.video.shape = self.video.getShape()
            self.times = np.arange(self.video.shape[0]) / self.video.inputfps
            # todo: replace the array method for an interative method of getting heading data
        # get video metadata
        self.num_frames, self.height, self.width = self.video.shape[:3]
        # get center and radii from the tracked file
        # tracked_folder = os.path.join(os.path.dirname(self.filename), "tracked_data")
        # tracked_fns = os.listdir(tracked_folder)
        # tracked_fn = [os.path.join(tracked_folder, fn) for fn in tracked_fns if self.subject in fn][0]
        # self.circle_data = np.load(tracked_fn)
        # self.center = np.array([self.circle_data['x'][0], self.circle_data['y'][0]])
        # self.inner_radius, self.outer_radius = self.circle_data[['radius_small', 'radius_large']][0]
        # # get ring coordinates and angles
        # self.set_rings(self.center, self.inner_radius, self.outer_radius)

    def get_background(self):
        """Get average frame of the whole video."""
        self.background = self.video.mean(-1)

    def ring_gui(self):
        """Use a matplotlib GUI to set the ring radii and thickness."""
        # use the camera module to load this video as a dummy video
        # with the 
        import sys
        sys.path.append("..\\holocube")
        from camera import Camera
        cam = Camera(plot_stimulus=False, kalman=True, config_fn="..\\video_player.config", camera=self.filename, video_player_fn="..\\video_player_server.py", com_correction=True)
        # cam.arm()
        cam.displaying = True
        # cam.capture_dummy()
        cam.capture_start()
        cam.display_start()
        resp = input("adjust the ring parameters and press <enter> when you're happy: ")
        cam.display_stop()
        cam.capture_stop()
        # store results from the gui
        for var, var_storage in zip(['thresh', 'inner_r', 'outer_r'], ['threshold', 'inner_radius', 'outer_radius']): 
            self.__setattr__(var_storage, cam.__getattribute__(var))
        # save these new values into the corresponding h5 file
        # extension = "." + self.filename.split(".")[-1]
        # h5_fn = self.filename.replace(extension, ".h5")
        # breakpoint()
        # dataset = h5py.File(h5_fn, mode='r+')

    def set_rings(self, center, inner_radius=10, outer_radius=20, wing_radius=40, 
                  thickness=3):
        """Define two rings for heading detection.


        Parameters
        ----------
        (center_x, center_y) : tuple, len=2
            The 2D coordinate of the center of both rings.
        inner_radius : float, default=10
            The radius of the inner ring, which should intersect both sides 
            of the fly.
        outer_radius : float, default=20
            The radius of the outer ring, which should intersect only the abdomen.
        wing_radius : float, default=40
            The radius of the wing ring, which should intersect only the wings.            
        thickness : int, default=3
            The thickness of the rings.
        """
        if inner_radius > outer_radius:
            outer_r_new = inner_radius
            outer_radius = inner_radius
            inner_radius = outer_r_new
        self.inner_radius, self.outer_radius = inner_radius, outer_radius
        self.wing_radius = wing_radius
        self.center = center
        x, y = center
        # make a mask of the two rings
        xs, ys = np.arange(self.width), np.arange(self.height)
        xgrid, ygrid = np.meshgrid(xs, ys)
        xgrid, ygrid = xgrid.astype(float), ygrid.astype(float)
        xgrid -= x
        ygrid -= y
        dists = np.sqrt(xgrid ** 2 + ygrid ** 2)
        self.dists = dists
        angles = np.arctan2(ygrid, xgrid)
        self.angles = angles
        # get indices of the two ring masks
        # inner ring and angles:
        include_inner = (dists >= inner_radius - thickness/2) * (dists <= inner_radius + thickness/2)
        self.inner_inds = include_inner
        ys_inner, xs_inner = np.where(include_inner)
        self.inner_ring_coords = np.array([xs_inner, ys_inner]).T
        self.inner_ring_angles = angles[include_inner]
        # outer ring and angles:
        include_outer = (dists >= outer_radius - thickness/2) * (dists <= outer_radius + thickness/2)
        self.outer_inds = include_outer
        ys_outer, xs_outer = np.where(include_outer)
        self.outer_ring_coords = np.array([xs_outer, ys_outer]).T
        self.outer_ring_angles = angles[include_outer]
        # wing ring and angles:
        wing_radius = 2 * outer_radius
        include_wing = (dists >= wing_radius - thickness/2) * (dists <= wing_radius + thickness/2)
        self.wing_inds = include_wing
        ys_outer, xs_outer = np.where(include_wing)
        self.wing_ring_coords = np.array([xs_outer, ys_outer]).T
        self.wing_ring_angles = angles[include_wing]


    def get_heading(self, floor=0, ceiling=40, method='rings', wings=False, gui=False):
        """Threshold the video and get the heading for each frame.


        Parameters
        ----------
        floor : int, default=5
            The pixel value lower bound for the inclusive filter
        ceiling : int, default=np.inf
            The pixel value upper bound for the inclusive filter
        method : str, default='rings'
            The heading can be appprixmated several ways, but only one is
            currently available, 'rings'. TODO: 1) get the svd of the thresholded 
            coordinates; 2) fit an ellipse; 3) use deep lab cut or some other machine
            learning model. 
        wings : bool, default=False
            Whether to also calculate the left and right wingbeat amplitudes.
        """
        self.heading = []
        self.wing_vals = []
        if 'headings' not in dir(self):
            self.headings = {}
        print('processing individual frames:')
        for num, frame in enumerate(self.video):
            if method == 'combined':
                # get the thresholded frame accounting for inversions
                if floor == 0 and ceiling > 0:
                    thresh = ceiling
                    invert = False
                elif ceiling >= 255 and floor > 0:
                    thresh = floor
                    invert = True
                # get the ring coordinates
                outer_inds, outer_angs = np.where(self.outer_inds), self.outer_ring_angles
                inner_inds, inner_angs = np.where(self.inner_inds), self.inner_ring_angles
                # 1. get outer ring, which should include just the tail
                if frame.ndim > 2:
                    frame = frame.mean(-1).astype('uint8')
                # get thresholded frame                
                if invert:
                    frame_mask = frame < thresh
                else:
                    frame_mask = frame > thresh
                # account for shifts in the center of mass
                com = scipy.ndimage.measurements.center_of_mass(frame_mask)
                diff = np.array(com) - np.array([self.height/2, self.width/2])
                # get the outer ring values
                outer_ring = frame[outer_inds[0], outer_inds[1]]
                heading = np.nan
                # 2. find the tail and head orientation by thresholding the outer ring
                # values and calculate the tail heading as the circular mean
                if invert:
                    tail = outer_ring < thresh
                else:
                    tail = outer_ring > thresh
                tail_angs = outer_angs[tail]
                tail_dir = scipy.stats.circmean(tail_angs.flatten(), low=-np.pi, high=np.pi)
                # the head direction is the polar opposite of the tail
                head_dir = tail_dir + np.pi
                if head_dir > np.pi:
                    head_dir -= 2 * np.pi
                # 3. get bounds of head angles, ignoring angles within +/- 90 degrees of the tail
                lower_bounds, upper_bounds = [head_dir - np.pi / 2], [head_dir + np.pi / 2]
                # wrap bounds if they go outside of [-pi, pi]
                if lower_bounds[0] < -np.pi:
                    lb = np.copy(lower_bounds[0])
                    lower_bounds[0] = -np.pi
                    lower_bounds += [lb + 2 * np.pi]
                    upper_bounds += [np.pi]
                elif upper_bounds[0] > np.pi:
                    ub = np.copy(upper_bounds[0])
                    upper_bounds[0] = np.pi
                    upper_bounds += [ub - 2 * np.pi]
                    lower_bounds += [-np.pi]
                lower_bounds, upper_bounds = np.array(lower_bounds), np.array(upper_bounds)
                # 4. calculate the heading within the lower and upper bounds
                include = np.zeros(len(inner_angs), dtype=bool)
                for lower, upper in zip(lower_bounds, upper_bounds):
                    include += (inner_angs > lower) * (inner_angs < upper)
                if np.any(include):
                    inner_vals = frame[inner_inds[0], inner_inds[1]][include]
                    if invert:
                        head = inner_vals < thresh
                    else:
                        head = inner_vals > thresh
                    heading = scipy.stats.circmean(
                        inner_angs[include][head],
                        low=-np.pi, high=np.pi)
                # convert from heading angle to head position
                heading_pos = self.inner_radius * np.array([np.sin(heading), np.cos(heading)])
                # calculate direction vector between the center of the fly and the head
                direction = heading_pos - diff
                heading = np.arctan2(direction[0], direction[1])
                # store the shift in the center of mass for plotting
                self.com_shift = np.round(-diff).astype(int)
                # center and wrap the heading
                heading -= np.pi/2
                if heading < -np.pi:
                    heading += 2 * np.pi
                # store
                self.heading += [heading]
            elif method == 'rings':
                if frame.ndim > 2:
                    frame = frame.mean(-1).astype('uint8')
                # 1. get angles of the outside ring corresponding to the fly's tail end
                xs, ys = self.outer_ring_coords.T
                outer_vals = frame[ys, xs]
                tail = (outer_vals > floor) * (outer_vals <= ceiling)
                tail_angs = self.outer_ring_angles[tail]
                # 1.5. check that the distribution
                # 2. get circular mean of tail
                tail_dir = scipy.stats.circmean(tail_angs, low=-np.pi, high=np.pi)
                head_dir = tail_dir + np.pi
                if head_dir > np.pi:
                    head_dir -= 2 * np.pi
                # 3. get bounds of head angles, ignoring angles within +/- 60 degrees of the tail
                lower_bounds, upper_bounds = [head_dir - np.pi/2], [head_dir + np.pi/2]
                # wrap bounds if they go outside of [-pi, pi]
                if lower_bounds[0] < -np.pi:
                    lb = np.copy(lower_bounds[0])
                    lower_bounds[0] = -np.pi
                    lower_bounds += [lb + 2 * np.pi]
                    upper_bounds += [np.pi]
                elif upper_bounds[0] > np.pi:
                    ub = np.copy(upper_bounds[0])
                    upper_bounds[0] = np.pi
                    upper_bounds += [ub - 2 * np.pi]
                    lower_bounds += [-np.pi]
                # 4. get angles of the inside ring corresponding to both head and tail
                head_angs = []
                for lb, ub in zip(lower_bounds, upper_bounds):
                    include = (self.inner_ring_angles > lb) * (self.inner_ring_angles < ub)
                    if np.any(include):
                        xs, ys = self.inner_ring_coords[include].T
                        inner_vals = frame[self.inner_ring_coords[include][:, 1],
                                           self.inner_ring_coords[include][:, 0]]
                        head = (inner_vals > floor) * (inner_vals <= ceiling)
                        head_angs += [self.inner_ring_angles[include][head]]
                if len(head_angs) > 0:
                    head_angs = np.concatenate(head_angs)
                # 5. grab the head angs within those bounds
                head_ang = scipy.stats.circmean(head_angs, low=-np.pi, high=np.pi)
                self.heading += [head_ang]
                # extra: measure the wing 
                if wings:
                    # get the wing ring values
                    wing_vals = frame[self.wing_ring_coords[:, 1], self.wing_ring_coords[:, 0]]
                    self.wing_vals += [wing_vals]
                    # test: use the heading angle to reduce the search for the wings
                    # plt.imshow(frame)
                    # plt.scatter(self.wing_ring_coords[:, 0], self.wing_ring_coords[:, 1], c=wing_vals)
                    # plt.show()
                    # plot a vector using the heading angle
                    # plt.figure()
            elif method == 'svd':
                # 1. threshold the video
                body = (frame > floor) * (frame <= ceiling)
                breakpoint()
                # 2. get 2D coordinates representing the fly
                # ys, xs =
                # 3. get the orientation of the principle component
            print_progress(num, self.num_frames)
        if isinstance(self.video, io.ffmpeg.FFmpegReader):
            self.video = io.FFmpegReader(self.filename)
        # convert to ndarray
        self.heading = np.array(self.heading)
        if np.any(np.isnan(self.heading)):
            # if np.isnan(self.heading).mean() > .1:
            #     breakpoint()
            # replace nans with linear interpolation
            inds = np.arange(len(self.heading))
            no_nans = np.isnan(self.heading) == False
            f = scipy.interpolate.interp1d(inds[no_nans], self.heading[no_nans], fill_value='extrapolate')
            self.heading[no_nans == False] = f(inds[no_nans == False])
        # check for absurdly fast movements
        self.headings[method] = self.heading

    def video_preview(self, vid_fn=None, relative=False, marker_size=3):
        """Generate video with headings superimposed.


        Parameters
        ----------
        relative : bool, default=False
            Whether to generate the video after removing the fly's motion.
        marker_size : int, default=3
            The side length in pixels of the square marker used to indicate 
            the head.
        """
        if vid_fn is None:
            vid_fn = ".".join(self.filename.split(".")[:-1])
            vid_fn += "_heading.mp4"
        # get new video parameters
        vid_radius = int(3 * self.outer_radius)
        vid_center = np.array([vid_radius, vid_radius])
        # get the indices of the center pixels
        ylow = max(round(self.center[1]) - vid_radius, 0) 
        yhigh = min(round(self.center[1]) + vid_radius, self.height)
        xlow = max(round(self.center[0]) - vid_radius, 0)
        xhigh = min(round(self.center[0]) + vid_radius, self.width)
        # get the sizes for reindexing
        height, width = yhigh - ylow, xhigh - xlow
        new_height, new_width = vid_radius * 2, vid_radius * 2
        ystart = round((new_height - height)/2)
        ystop = ystart + height
        xstart = round((new_width - width)/2)
        xstop = xstart + width
        # crop the video first, to save time
        cropped_video = np.copy(self.video[:, ylow : yhigh, xlow : xhigh])[..., np.newaxis]
        # make an empty frame for temporary storage
        # frame_centered = np.zeros((2 * vid_radius, 2 * vid_radius, 3), dtype=float)
        # open a video and start storing frames
        # new_vid = io.FFmpegWriter(vid_fn)
        # todo: store into a numpy array and use vwrite instead
        new_vid = np.zeros((self.num_frames, 2 * vid_radius, 2 * vid_radius, 3), dtype='uint8')
        # for each frame:
        # for frame, orientation in zip(cropped_video, self.heading):
        for num, (frame, orientation, frame_centered) in enumerate(zip(
                cropped_video, self.heading, new_vid)):
            if not np.isnan(orientation):
                # get the head position
                d_vector = np.array([np.cos(orientation), np.sin(orientation)])
                pos = np.round(vid_center + self.inner_radius * d_vector).astype(int)
                # make a version of the frame with a red square centered at the mean orientation
                frame_centered[ystart:ystop, xstart:xstop] = frame
                # if specified, rotate the frame to get relative motion
                if relative:
                    frame_centered = scipy.ndimage.rotate(
                        frame_centered, (orientation + np.pi/2) * 180 / np.pi,
                        reshape=False)
                # otherwise, draw a line indicating the heading
                else:
                    try:
                        rr, cc, val = skimage.draw.line_aa(vid_radius, vid_radius, pos[0], pos[1])
                    except:
                        breakpoint()
                    # scale down the older values
                    old_vals = frame_centered[cc, rr].astype(float)
                    old_vals *= (1 - val)[..., np.newaxis]
                    frame_centered[cc, rr] = old_vals.astype('uint8')
                    frame_centered[cc, rr, 0] = val * 255
                    # frame_centered[cc, rr, 1:] = 10
                    frame_centered[pos[1] - marker_size : pos[1] + marker_size,
                                   pos[0] - marker_size : pos[0] + marker_size] = [155, 0, 0]
                # store the new frame
                # new_vid.writeFrame(frame_centered)
                # clear the temporary frame
                # frame_centered.fill(0)
            print_progress(num, self.num_frames)
        # new_vid.close()
        # use vwrite to store the array as a video
        io.vwrite(vid_fn, new_vid)
        
        print(f"Heading preview saved in {vid_fn}.")

    def extract_ring(self, ring='inner', bins=np.linspace(-np.pi, np.pi, 361)):
        """Bin frame pixels using one of the defined rings.

        Parameters
        ----------
        ring : str, default='inner'
            The ring variable to use for extracting the pixel values regularly.
        bins : np.linspace
            The list of values for partitioning the pixel orientations.
        """
        # grab the pertinent ring parameters
        angles, coords = self.__getattribute__(f"{ring}_ring_angles"), self.__getattribute__(f"{ring}_ring_coords")
        num_bins = len(bins) - 1
        # make empty array for storing the final graph
        vals = np.zeros((self.num_frames, num_bins), dtype='uint8')
        # group the lists of angles by the defined bins
        bin_groups = np.digitize(angles, bins)
        # get the video pixel values within these bins
        if isinstance(self.video, np.ndarray):
            in_ring = self.video[:, coords[:, 1], coords[:, 0]]
        else:
            in_ring = []
            self.video = io.FFmpegReader(self.filename)
            for frame in self.video: in_ring += [frame[coords[:, 1], coords[:, 0]]]
            in_ring = np.array(in_ring)
        # sort the pixel values using the bin group labels
        order = np.argsort(bin_groups)
        in_ring = in_ring[:, order]
        bin_groups = bin_groups[order]
        changes = np.diff(bin_groups) > 0
        changes = np.where(changes)[0]
        # get binned pixel values
        bin_vals = np.split(in_ring, changes, axis=1)
        bin_vals = np.array([bin_val.mean(1) for bin_val in bin_vals])
        # return the binned values
        return bin_vals

    def graph_preview(self, bins=np.linspace(-np.pi, np.pi, 361), wings=False):
        """A Space X Time graph of inner, outer, and inner-outer rings


        Parameters
        ----------
        bins : array-like, default=np.linspace(-np.pi, np.pi, 361)
            Define the bounds of bins used for flattening the ring values.
        wings : bool, default=False
            Whether to also plot the flattened wing data.
        """
        num_bins = len(bins) - 1
        # store one space X time graph per ring
        graphs = []
        # for each ring:
        rings = ['inner', 'outer']
        if wings:
            rings += ['wing']
        # bin pixel values using the ring extraction function
        for ring in rings:
            graphs += [self.extract_ring(ring, bins=bins)]
        # ring_angles = [self.small_ring_angles, self.large_ring_angles]
        # ring_coords = [self.small_ring_coords, self.large_ring_coords]
        # if wings:
        #     ring_angles += [self.wing_ring_angles]
        #     ring_coords += [self.wing_ring_coords]
        # for angles, coords in zip(ring_angles, ring_coords):
        #     # make empty array for storing the final graph
        #     vals = np.zeros((self.num_frames, num_bins), dtype='uint8')
        #     # group the lists of angles by the defined bins
        #     bin_groups = np.digitize(angles, bins)
        #     # get the video pixel values within these bins
        #     in_ring = self.video[:, coords[:, 1], coords[:, 0]]
        #     # sort the pixel values using the bin group labels
        #     order = np.argsort(bin_groups)
        #     in_ring = in_ring[:, order]
        #     bin_groups = bin_groups[order]
        #     changes = np.diff(bin_groups) > 0
        #     changes = np.where(changes)[0]
        #     # get binned pixel values
        #     bin_vals = np.split(in_ring, changes, axis=1)
        #     bin_vals = np.array([bin_val.mean(1) for bin_val in bin_vals])
        #     # store
        #     graphs += [bin_vals]
        # now graph:
        fig, axes = plt.subplots(ncols=len(graphs), sharey=True, sharex=True)
        # time_bins = np.append([0], self.times)
        time_bins = self.times
        for num, (ax, graph, ring) in enumerate(zip(axes, graphs, rings)):
            if graph.ndim > 2 and graph.shape[-1] > 1:
                ax.pcolormesh(bins, time_bins, graph[..., 0].T)
            else:
                ax.pcolormesh(bins, time_bins, graph.T)
            # line plot excluding steps > np.pi
            for (key, val), color in zip(self.headings.items(), [red, green, blue]):
                headings = np.copy(val)
                speed = np.append([0], np.diff(headings))
                headings[abs(speed) > np.pi] = np.nan
                ax.plot(headings, self.times, color=color, label=key)
                ax.set_title(f"{ring.capitalize()} Ring")
        # 1. the outer ring
        # axes[0].pcolormesh(bins, time_bins, graphs[0].T)
        # axes[0].scatter(self.headings, self.times, color=red, marker='.')
        # line plot excluding steps > np.pi
        # for (key, val), color in zip(self.headings.items(), [red, green, blue]):
        #     headings = np.copy(val)
        #     speed = np.append([0], np.diff(headings))
        #     headings[abs(speed) > np.pi] = np.nan
        #     axes[0].plot(headings, self.times, color=color, label=key)
        # axes[0].invert_yaxis()
        # axes[0].set_title("Inner Ring")
        # 2. the inner ring
        # axes[1].pcolormesh(bins, time_bins, graphs[1].T)
        # axes[1].set_title("Outer Ring")
        # if wings:
            # 3. the wing ring
            # axes[2].pcolormesh(bins, time_bins, graphs[2].T)
            # axes[2].set_title("Wing Ring")
        axes[-1].invert_yaxis()
        axes[-1].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                        ["-$\pi$", "-$\pi$/2", "0", "$\pi$/2", "$\pi$"])
        # format
        plt.tight_layout()
        plt.show()

class OfflineTracker():
    def __init__(self, folder="./", vid_extension='.mp4'):
        """Track fly headings for each video and store.

        Parameters
        ----------
        folder : path, default='./'
            The path of the folder to process.
        vid_extension : str, default='.mp4'
            The extensions of the videos we intend to process.
        """
        self.folder = folder
        self.vid_extension = vid_extension
        # get the full path of each element in folder
        self.fns = os.listdir(self.folder)
        self.fns = [os.path.join(self.folder, fn) for fn in self.fns]
        # get all of the relevant files
        self.vid_fns = [fn for fn in self.fns if fn.endswith(self.vid_extension)]
        self.h5_fns = [fn for fn in self.fns if fn.endswith(".h5")]
        # find all videos that have an associated h5 file
        new_vid_fns = []
        new_h5_fns = []
        for vid_fn in self.vid_fns:
            h5_fn = vid_fn.replace(self.vid_extension, '.h5')
            if h5_fn in self.h5_fns:
                new_vid_fns += [vid_fn]
                new_h5_fns += [h5_fn]
        self.vid_fns = new_vid_fns
        self.h5_fns = new_h5_fns

    def process_vids(self, start_over=False, method='combined', wings=False, gui=False, display=False):
        """For each video and h5 file, track and store the heading data.

        Parameters
        ----------
        start_over : bool, default=False
            Whether to check if the offline dataset has already been saved.
        method : str, options=['combined', 'rings', 'svd', 'fourier_mellin']
            The method of measuring the
        wings : bool, default=False
            Whether to measure the left and right wingbeat amplitudes.
        gui : bool, default=False
            Whether to use a GUI to manually define the ring and threshold parameters.
        display : bool, default=False
            Whether to display the results of each video.
        """
        for vid_fn, h5_fn in zip(self.vid_fns, self.h5_fns):
            # load the dataset, which specifies the inner and outer radii for
            # tracking
            data = TrackingTrial(h5_fn)
            # only continue if 1) this hasn't been run before or start_over
            # is set to True, 2) 'end_test' times were specified, and 3) the
            # dataset loaded successfully
            already_processed = 'camera_heading_offline' in dir(data)
            stop_specified = 'stop_test' in dir(data)
            success = not already_processed and stop_specified and data.load_success
            if success or start_over:
                # load the video and process the fly's heading
                track = TrackingVideo(vid_fn)
                if method == 'combined':
                    # reset the rings and store in the database
                    if gui:
                        track.ring_gui()
                        inner_r, outer_r, threshold = track.inner_radius, track.outer_radius, track.threshold
                        for val, lbl in zip([inner_r, outer_r, threshold], ['inner_r', 'outer_r', 'thresh']):
                            data.add_attr(lbl, val)
        for vid_fn, h5_fn in zip(self.vid_fns, self.h5_fns):
            # load the dataset, which specifies the inner and outer radii for
            # tracking
            data = TrackingTrial(h5_fn)
            # continue if 1) this hasn't been run before or start_over
            # is set to True, 2) 'end_test' times were specified, and 3) the
            # dataset loaded successfully
            already_processed = 'camera_heading_offline' in dir(data)
            stop_specified = 'stop_test' in dir(data)
            success = (not already_processed or start_over) and stop_specified and data.load_success
            if success or start_over:
                # load the video and process the fly's heading
                track = TrackingVideo(vid_fn)
                if method == 'combined':
                    # reset the rings 
                    inner_r, outer_r, threshold = data.inner_r, data.outer_r, data.thresh
                    # set the ring parameters from our dataset
                    center = (track.width/2, track.height/2)
                    track.set_rings(center, inner_radius=inner_r, outer_radius=outer_r, thickness=5)
                    track.get_heading(floor=0, ceiling=threshold, wings=wings, method='combined')
                    # why is the offline tracking so much longer than the online tracking?
                    # note: I figured out why. the framerate of online tracking is determined by the 
                    # framerate of the stimulus. this must have been limited to 60 Hz, for some reason, and
                    # the camera must have been set to 120 Hz, so there was a 2-fold difference in the time 
                    # course of the responses. Also, NaNs resulted in a lot of missing data due to the unwrap 
                    # function, not actually missing data. The online tracking can be re-calculated almost 
                    # perfectly by placing both on a regular timeline using the start_test and start_exp 
                    # attributes. So, we can pretty safely split up the offline heading data via the 
                    # following algorithm:
                    # 1) apply a timestamp to each frame of the offline heading
                    # 2) figure out the frame ratio between the duration of offline (k) and the online heading (j; k/j should be an integer)
                    # 3) based on the length of the online heading tests, calculate the length of the offline tests, k
                    # 4) generate a new dataset by grabbing the k values before each keyframe
                    # plot the offline tracking with timing based on the start and stop of the experiment
                    # heading_offline
                    # time_offline = np.linspace(data.start_exp, data.stop_exp, len(heading_offline))
                    # note: offline and online headings are off by about pi
                    # plt.plot(time_offline, np.unwrap((heading_offline-np.pi)%2*np.pi - np.pi))
                    # for kf, heading in zip(data.stop_test, data.camera_heading): plt.axvline(kf); ts = np.arange(len(heading)) / 120.; plt.plot(kf-2*ts[::-1], np.unwrap(heading))
                    if display:
                        track.graph_preview()
                    # the method below assumes that the recorded framerates for the offline and online
                    # heading are correct, but it looks like sometimes the holocube framerate says 120 
                    # but was actually 60 Hz
                    # get times associated with each frame
                    # time = track.times / data.framerate
                    # the total duration is the same between the two
                    # time_offline = np.linspace(0, data.time.max(), len(track.times))
                    heading_offline = track.headings['combined']
                    # time_online = data.query('time', sort_by='time')
                    # heading_online = data.query('camera_heading', sort_by='time')
                    heading_online = data.camera_heading
                    num_tests, num_frames = heading_online.shape
                    duration = data.stop_exp - data.start_exp
                    fps_offline = heading_offline.size / duration
                    fps_online = heading_online.size / duration
                    frame_ratio = fps_offline / fps_online
                    # frame_ratio = np.round(data.framerate / data.holocube_framerate)
                    num_frames_offline = int(round(frame_ratio * num_frames))
                    heading_offline_arr = np.zeros((num_tests, num_frames_offline), dtype=float)
                    total_frames = num_tests * frame_ratio * num_frames
                    extra_frames = heading_offline.shape[0] - total_frames
                    frame_padding = int(round(extra_frames / num_tests-1))
                    start, stop = data.start_exp, data.stop_exp
                    stops = data.stop_test
                    times_offline = np.linspace(start, stop, len(heading_offline) + 1)[:-1]
                    for num, (storage, stop) in enumerate(
                            zip(heading_offline_arr, stops)):
                        # 1. find the nearest frame to the stop point
                        frame_num = np.argmin(abs(times_offline - stop))
                        # 2. get the clip before this stop point
                        # account for slight differences in the length of each array
                        max_x = min(frame_num, storage.shape[0])
                        storage[- max_x:] = heading_offline[frame_num - max_x:frame_num]
                    heading_offline_arr -= np.pi
                    heading_offline_arr[heading_offline_arr < -np.pi] += 2* np.pi
                    heading_offline_arr[heading_offline_arr > np.pi] -= 2* np.pi
                elif method == 'svd':
                    # get the principle component of the above-threshold pixel coordinates
                    breakpoint()
                data.add_dataset('camera_heading_offline', heading_offline_arr)
                # add a time
                time_online = data.query('time', sort_by='test_ind')
                time_offline = np.linspace(
                    0, time_online.max(), heading_offline_arr.size).reshape(
                    heading_offline_arr.shape)
                data.add_dataset('time_offline', time_offline)
                print(f"updated {h5_fn}")
                # check if the online and offline headings are close
                # fig, ax = plt.subplots()
                # times_online = np.linspace(0, data.stop_exp - data.start_exp, heading_online.size)
                # times_offline = np.linspace(0, data.stop_exp - data.start_exp, heading_offline_arr.size)
                # ax.plot(times_online, heading_online.flatten(), color='k', zorder=2)
                # ax.plot(times_offline, heading_offline_arr.flatten(), linestyle="", marker='.', color=red, zorder=3)
                # plt.show()
                # breakpoint()

    def offline_comparison(self, smooth_offline=False):
        """Compare the offline from the online heading measurements.

         Measure the spatial and temporal performance of the online measurement
         in contrast to the more detailed offline measurement.
        """
        fig_summary, axes_summary = plt.subplots(
            nrows=3, ncols=2, gridspec_kw={'width_ratios': [5, 1]})
        # 1. demo individual traces and plot superimpose the bar trajectory
        demo_exp_num = 0
        demo_trial_num = 0
        demo_start_time = 114
        demo_stop_time = 132
        summary_lags = []
        summary_corrs = []
        summary_corr_TS = []
        for num, (vid_fn, h5_fn) in enumerate(zip(self.vid_fns, self.h5_fns)):
            # load the dataset, which specifies the inner and outer radii for
            # tracking
            data = TrackingTrial(h5_fn)
            if 'camera_heading_offline' in dir(data) and 'time' in dir(data):
                # make a figure with two equal rows with one wide and one narrow column
                # fig, axes = plt.subplots(nrows=3, ncols=2, gridspec_kw={'width_ratios':[5, 1]})
                # top row: plot the two time series in the top row
                # top_row, middle_row, bottom_row = axes
                # ts_ax, corr_ax = top_row
                sum_ts_ax, sum_corr_ax = axes_summary[0]
                time_online = data.query('time', sort_by='test_ind')
                # plot the offline
                if smooth_offline:
                    data.butterworth_filter('camera_heading_offline', 0, 10, sample_rate=data.framerate)
                    heading_offline = data.query('camera_heading_offline_smoothed',
                                                sort_by='test_ind')
                    data.butterworth_filter('camera_heading', 0, 10,
                                            sample_rate=data.holocube_framerate)
                    heading_online = data.query('camera_heading_smoothed',
                                                   sort_by='test_ind')

                else:
                    heading_offline = data.query('camera_heading_offline',
                                                    sort_by='test_ind')
                heading_online = data.query('camera_heading',
                                               sort_by='test_ind')
                time_offline = data.query('time_offline',
                                             sort_by='test_ind')
                # diff = heading_offline.flatten()[0] - heading_online.flatten()[0]
                # heading_online = np.unwrap(heading_online.flatten())
                # offset = heading_offline[0] - heading_online[0]
                # offset = diff - offset
                # ts_ax.plot(time_online.flatten(), heading_online.flatten(), color='k', alpha=1, lw=.5)
                # ts_ax.plot(time_offline.flatten(), heading_offline.flatten(), color='lightgray', zorder=1)
                # format
                # ts_ax.set_xticks([])
                # ts_ax.set_yticks([-np.pi, 0, np.pi], ["-$\pi$", "0", "$\pi$"])
                # sbn.despine(ax=ts_ax, bottom=True)
                # right axis: plot the cross-correlation of offline and online heading measurements
                # calculate the cross-correlation of each test, with online signal resampled to match
                # the length of the offline signal
                corrs = []
                peak_lags = []
                peak_corrs = []
                num_frames = heading_offline.shape[-1]
                lags = scipy.signal.correlation_lags(num_frames, num_frames,
                                               mode='same').astype(float)
                lags /= data.framerate
                for test_online, test_offline in zip(heading_online, heading_offline):
                    # interpolate points in online to match offline heading
                    online_resampled = np.repeat(test_online, 4)
                    _, corr = normal_correlate(
                        np.unwrap(test_offline), np.unwrap(online_resampled),
                        mode='same', circular=False)
                    corrs += [corr]
                    no_nans = np.isnan(test_offline)
                    no_nans += np.isnan(online_resampled)
                    no_nans = no_nans == False
                    # corr_ax.plot(lags, corr, lw=.5, color='k', alpha=.25)
                    # get peak maximum
                    pos_lags = lags >= 0
                    max_ind = np.argmax(corr[pos_lags])
                    peak_lags += [lags[pos_lags][max_ind]]
                    peak_corrs += [corr[pos_lags][max_ind]]
                # highlight the range of peaks
                # corr_ax.scatter(peak_lags, peak_corrs, color=red, alpha=.25,
                #                 edgecolors='none')
                # corr_ax.set_xlim(-.5, 1)
                # plot the mean cross-correlation
                mean_corr = np.nanmean(np.array(corrs), axis=0)
                summary_corr_TS += [mean_corr]
                max_ind = np.argmax(mean_corr[pos_lags])
                summary_corrs += [mean_corr[pos_lags][max_ind]]
                summary_lags += [lags[pos_lags][max_ind]]
                # corr_ax.plot(lags, mean_corr, lw=1, alpha=1, color='k')
                # sum_corr_ax.plot(lags, mean_corr, lw=.5, alpha=.25, color='k')
                # add text showing the median +/- IQR of the pearson correlations
                low, mid, high = np.percentile(peak_corrs, [25, 50, 75])
                IQR = high - low
                # corr_ax.text(.5, 1., f"{mid:.2}+/-{IQR:.5}")
                if num == 0:
                    # plot the offline and online headings
                    sum_ts_ax.plot(time_online.flatten(), heading_online.flatten(), color='k', alpha=1, lw=.5)
                    sum_ts_ax.plot(time_offline.flatten(), heading_offline.flatten(), color='lightgray', zorder=1)
                    # zoom into demo region
                    sum_ts_ax.set_xlim(demo_start_time, demo_stop_time)
                    sum_ts_ax.set_ylim(-np.pi/2, np.pi/2)
                    # format
                    sum_ts_ax.set_xticks([])
                    sum_ts_ax.set_yticks([-np.pi/2, 0, np.pi/2], ["-$\dfrac{\pi}{2}$", "0", "$\dfrac{\pi}{2}$"])
                    sum_ts_ax.set_ylabel("Heading")
                    sbn.despine(ax=sum_ts_ax, bottom=True, trim=True)
                    # plot the velocity time series below position
                    sum_velo_ax, sum_velo_hist_ax = axes_summary[1]
                    velo_offline = np.diff(np.unwrap(heading_offline).flatten())
                    velo_online = np.diff(np.unwrap(heading_online).flatten())
                    sum_velo_ax.plot(
                        time_online.flatten()[1:], velo_online, color='k',
                        alpha=1, lw=.5)
                    sum_velo_ax.plot(
                        time_offline.flatten()[1:], velo_offline,
                        color='lightgray', zorder=1)
                    # zoom into demo region
                    sum_velo_ax.set_xlim(demo_start_time, demo_stop_time)
                # plot the cross-correlation
                sum_corr_ax.plot(lags, mean_corr, lw=.5, alpha=.25, color='k')

        # format
        sum_corr_ax.set_ylim(0, 1.1)
        sum_corr_ax.set_yticks([0, .5, 1])
        sum_corr_ax.set_ylabel("Correlation")
        sum_corr_ax.set_xlim(-.1, .6)
        sum_corr_ax.set_xticks([0, .5])
        sbn.despine(ax=sum_corr_ax, bottom=False, trim=True)

                # right axis: plot the distribution of velocities
                # bottom row: plot the residual time series in the middle row

                # plot two insets: zoom into the a) highest and b) lowest speed
                # in the right margin, plot the histogram of residuals
        # scatterplot the peak correlations and lags
        sum_corr_ax.scatter(summary_lags, summary_corrs, color=red, alpha=.25,
                            edgecolors='none')
        sum_corr_ax.plot(lags, np.nanmean(summary_corr_TS, axis=0), color='k', alpha=1)
        plt.tight_layout()
        plt.show()
        breakpoint()

        # 2. plot individual residual traces


class TrackingExperiment():
    def __init__(self, dirname, remove_incompletes=True, **trial_kwargs):
        """Load all H5 TrackingTrial files from the same experiment.

        Parameters
        ----------
        dirname : path
            The directory to load.
        remove_incompletes : bool, default=True
            Whether to keep trials with missing tests.
        """
        self.dirname = dirname
        self.files = os.listdir(dirname)
        # get the path of the h5 files
        self.h5_files = [os.path.join(dirname, fn) for fn in self.files if fn.endswith(".h5")]
        # load a TrackingTrial for each file
        
        self.trials = []
        for fn in self.h5_files:
            trial = TrackingTrial(fn, **trial_kwargs)
            if trial.load_success:
                self.trials += [trial]
        if remove_incompletes:
            self.remove_incompletes()

    def query(self, same_size=True, **kwargs):
        """Get specified data from each trial."""
        ret = []
        if 'subset' not in kwargs.keys():
            kwargs['subset'] = {}
        for trial in self.trials:
            # if 'dirname' in kwargs.values:
            #     breakpoint()
            # res = trial.query(output=kwargs['output'], subset=kwargs['subset'])
            try:
                res = trial.query(**kwargs)
            except:
                breakpoint()
            ret += [res]
        # check if all the results have the same shape
        try:
            first_shape = ret[0].shape
        except:
            breakpoint()
        same_shape = [trial_data.shape == first_shape for trial_data in ret]
        # return as an array if they are all the same shape
        if np.all(same_shape):
            ret = np.array(ret)
        elif same_size:
            shapes = [arr.shape for arr in ret]
            sizes = np.array([arr.size for arr in ret])
            max_shape = shapes[np.argmax(sizes)]
            new_ret = []
            empty = np.zeros(max_shape)
            for arr in ret:
                empty.fill(np.nan)
                if arr.ndim > 1:
                    for vals, storage in zip(arr, empty):
                        storage[:len(vals)] = vals
                else:
                    empty[:, :len(arr)] = arr
                new_ret += [np.copy(empty)]
            ret = np.array(new_ret)
            # find the maximum shape and pad the others with NaNs to match
        return ret

    def query_bouts(self, **kwargs):
        """Get the bouts from each trial."""
        ret = []
        for trial in self.trials:
            ret += [trial.query_bouts(**kwargs)]
        return ret

    def query_saccades(self, **kwargs):
        """Get saccade data from each trial."""
        ret = []
        for trial in self.trials:
            ret += [trial.query_saccades(**kwargs)]
        return ret

    def add_dataset(self, name, vals):
        """Add this dataset to each trial.
        
        Add a dataset of the provided values under the specified name.

        Parameters
        ----------
        name : str
            The name of the dataset
        arr : np.ndarray, shape=(k) or (N, k)
            The list or array of arrays to store.
        """
        for trial, arr in zip(self.trials, vals):
            trial.add_dataset(name, arr)

    def center_initial(self, **kwargs):
        """Center the initial position for all tests of all trials."""
        for trial in self.trials:
            trial.center_initial(**kwargs)

    def unwrap(self, **kwargs):
        """Center the initial position for all tests of all trials."""
        for trial in self.trials:
            trial.unwrap(**kwargs)

    def remove_incompletes(self):
        """Use only trials with the maximum number of tests."""

        new_trials = []
        # find max number of tests
        max_tests = max([trial.num_tests for trial in self.trials])
        for trial in self.trials:
            if trial.num_tests == max_tests:
                new_trials += [trial]
        self.trials = new_trials

    def remove_too_fast(self, variable='camera_heading', speed_limit=np.pi/4, tolerance=.01):
        """Use only trials with speeds less than the limit.
        
        This is good for getting rid of trials with errors resulting in head-tail inversions.

        Parameters
        ----------
        variable : str, default='camera_heading'
            The variable used to apply this speed limit.
        speed_limit : float, default=pi/4
            The maximum allowable speed.
        tolerance : float in [0, 1], default=.01
            Keep trials with this proportion of frames that are too fast.
        """
        new_trials = []
        for trial in self.trials:
            vals = trial.query(variable, sort_by='test_ind')
            speeds = np.diff(vals, axis=-1)
            too_fast = abs(speeds) > speed_limit
            if not np.any(too_fast.mean(-1) > tolerance):
                new_trials += [trial]
        self.trials = new_trials

    def remove_nonrandoms(self, variable, invert=False):
        """Use only trials where variable is in increasing order.

        Parameters
        ----------
        variable: str
            The name of the variable that should be in ascending order.
        invert : bool, default=False
            Whether to instead remove only randomized trials.
        """
        new_trials = []
        for trial in self.trials:
            vals = getattr(trial, variable)
            if np.all(np.diff(vals) >= 0):
                new_trials += [trial]
        self.trials = new_trials

    def remove_subjects_w_nans(self, variable, thresh=0):
        """Use only trials without nans.

        Parameters
        ----------
        variable: str
            The name of the variable that should have no nans.
        thresh: float, default=0
            The maximum proportion of nans allowed. Must be between 0 and 1.
        """
        new_trials = []
        for trial in self.trials:
            vals = getattr(trial, variable)
            if np.mean(np.isnan(vals)) <= thresh:
                new_trials += [trial]
        self.trials = new_trials

    def remove_still_subjects(self, variable, thresh=10):
        """Use only trials where variable is .

        Parameters
        ----------
        variable: str
            The name of the variable that should have no nans.
        thresh: float, default=10
            The minimum tolerable number of unique values.
        """
        new_trials = []
        for trial in self.trials:
            vals = getattr(trial, variable)
            if len(np.unique(vals)) > thresh:
                new_trials += [trial]
        self.trials = new_trials

    def get_saccade_stats(self, **kwargs):
        """Process individual saccades and measure saccade data per trial."""
        for trial in self.trials:
            trial.get_saccade_stats(**kwargs)

    def remove_saccades(self, **saccade_kwargs):
        """Generate a new instance of the """
        for trial in self.trials:
            trial.remove_saccades(**saccade_kwargs)


    def butterworth_filter(self, low=1, high=6, key='camera_heading',
                           sample_rate=60.):
        """Apply a Butterworth filter to the dataset of each trial.

        Parameters
        ----------
        low : float, default=1
            The lower bound for filtering the specified dataset
        high: float, default=6
            The upper bound for filtering the specified dataset.
        key : str, default='camera_heading'
            The variable to filter.
        sample_rate : float, default=60
            The sample rate used for calulating frequencies.
        """
        for trial in self.trials:
            trial.butterworth_filter(key, low, high, sample_rate)
            trial.__setattr__(key, trial.__getattribute__(key+"_smoothed"))

    def plot_saccades(self, col_var, row_var, time_var='time', start=0, 
                      stop=.5, row_cmap=None, col_cmap=None, 
                      fig=None, right_margin=True, bottom_margin=True,
                      xlim=None, ylim=None, xticks=None, yticks=None, 
                      positive_amplitude=False, scale=1.5, reversal_split=False,
                      saccade_var='arr_relative', min_speed=350, max_speed=np.inf,
                      mean_bins=25,
                      **query_kwargs):
        """Plot saccade data in one big grid as in the plot summary below.
        
        The color for each subplot is determined by the average of the column and 
        row colors: color = sqrt(mean([col_color^2, row_color^2])). This is the 
        proper way to average two colors and should generate a unique color for each 
        plot. 

        Parameters
        ----------
        col_var, row_var : str
            The variable names to paramaterize along columns and rows of the grid.
        time_var : str, default='time'
            The time variable to use. So far there are two options: 1) time from the start 
            of the saccade or 2) time from the peak velocity.
        start, stop : float, default=0., np.inf.
            The start and stop times to include in the saccade.
        row_cmap, col_cmap : func or array-like, default=None
            The colormap to use for colorizing along the rows or columns. If both are
            supplied, the product of the two colors is used for each subplot. If a 
            function is supplied, it will be applied to the corresponding col_var or row_var.
            If a list is supplied, it must have as many elements as the corresponding column or
            row.
        right_margin, bottom_margin : bool, default=True
            Whether to plot the means from each subplot in the margin to the right
            or bottom of the axes. A colormap (row_cmap or col_cmap) must be provided
            in order to distinguish individual traces.
        xlim, ylim : tuple=(min, max), default=None
            The tuple of the minimum and maximum values for that dimension.
        xticks, yticks : list, default=None
            Specify the list of ticks on the x- or y-axis.
        positive_amplitude : bool, default=False
            Whether to normalize the traces so that they all end positive.
        reversal_split : bool, default=False
            Whether to separately plot saccades that are reversing direction from those going in
            the same direction.
        saccade_var : str, default='arr_relative'
            The saccade variable to plot. The default is the main heading relative to the start
            of the saccade. Choose one from the following: arr_relative, velocity, acceleration.
        summary_func : callable, default=np.nanmean
            The function to plot summarizing the data in each subplot.
        mean_bins : int, default=25
            The number of bins to use for bin averaging the saccade time series.
        min_speed, max_speed : float, default=350, np.inf
            The minimum and maximum peak speed to include in the saccades here.
        **query_kwargs
            These get passed to the query 
        """
        if 'subset' not in query_kwargs:
            query_kwargs['subset'] = {}
        if 'sort_by' not in query_kwargs:
            query_kwargs['sort_by'] = 'test_ind'
        subset = copy.copy(query_kwargs['subset'])
        # get the values used for coloring each subplot
        if row_var is not None:
            row_vals = self.query(output=row_var, sort_by=row_var, subset=subset)
        else:
            row_vals = [None]
        if col_var is not None:
            col_vals = self.query(output=col_var, sort_by=col_var, subset=subset)
        else:
            col_vals = [None]
        assert len(col_vals) > 0 or len(row_vals) > 0, "The subset is empty!"
        # todo: what's up with the number of axes?
        row_vals = np.unique(row_vals)
        col_vals = np.unique(col_vals)
        num_rows, num_cols = len(row_vals), len(col_vals)
        # get the colors from the specified colormaps
        colors = {}
        for cmap, vals, key in zip(
            [row_cmap, col_cmap], 
            [row_vals, col_vals],
            ['rows', 'columns']):
            if len(vals) > 0:
                if isinstance(cmap, str):
                    if isinstance(vals[0], (str, bytes)):
                        # if values are strings, sort them in alphabetical order and use their index for the colors
                        vals = np.argsort(vals)
                    norm = matplotlib.colors.Normalize(vals.min(), vals.max())
                    cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
                    colors[key] = cmap.to_rgba(vals)[:, :-1]
                elif callable(cmap):
                    colors[key] = cmap(vals)
                elif isinstance(row_cmap, (list, np.ndarray, tuple)):
                    assert len(cmap) == len(vals), (
                        f"Colormap list has {len(cmap)} elements but {len(vals)} {key}.")
                    colors[key] = row_cmap
                else:
                    colors[key] = []
            else:
                colors[key] = []
        # combine the color lists to make an array specifying the color of each subplot
        if len(colors['rows']) == num_rows and len(colors['columns']) == num_cols:
            # get the mean comination of row and column colors
            # try:
            if num_rows > 0 and num_cols > 0:
                color_mean = .5*(
                    colors['rows'][:, np.newaxis]**2 +
                    colors['columns'][np.newaxis, :]**2)
            else:
                max_ind = np.argmax([len(colors[key]) for key in colors.keys()])
                color_mean = tuple(colors.values())[max_ind]
                if num_cols == 0:
                    num_cols = 1
                if num_rows == 0:
                    num_rows = 1
            color_arr = np.sqrt(color_mean)
        elif len(colors['columns']) == num_cols:
            color_arr = np.repeat(colors['columns'][np.newaxis], num_rows, axis=0)
        elif len(colors['rows']) == num_rows:
            color_arr = np.repeat(colors['rows'][:, np.newaxis], num_cols, axis=1)
        else:
            color_arr = np.zeros((num_rows, num_cols, 3), dtype='uint8')
        # todo: add extra rows if split_reversal
        if reversal_split:
            num_rows *= 2
        # test: check that the colors are aligned properly. we want this array shape to be num_rows X num_cols
        # add a row or column if plotting in the margins
        if bottom_margin:
            num_rows += 1
        if right_margin:
            num_cols += 1
        # make the figure and axes with a grid defined by num_rows and num_cols
        figsize = (scale * num_cols + 1, scale*num_rows + 1)
        self.display = SummaryDisplay(
            num_rows=num_rows, num_cols=num_cols, right_margin=right_margin, 
            bottom_margin=bottom_margin, figsize=figsize)
        trace_axes = self.display.trace_axes
        # plot the data
        num_frames = self.trials[0].num_frames
        means = np.zeros((trace_axes.shape[0], trace_axes.shape[1], int(num_frames)), dtype=float)
        max_count = 0
        # split trace axes if reversal split
        if reversal_split:
            new_num_rows = round(num_rows / 2)
            trace_axes = trace_axes.reshape((new_num_rows, 2, trace_axes.shape[1]))
        for row_num, (row, row_val, row_colors, row_means) in enumerate(zip(
            trace_axes, row_vals, color_arr, means)):
            if right_margin:
                if reversal_split:
                    row_summ_axes = self.display.right_col[2 * row_num: 2 * row_num + 2]
                else:
                    row_summ_axes = [self.display.right_col[row_num]]
            else:
                row_summ_axes = None
            # update the subset dictionary
            if row_var is not None and row_val is not None:
                subset[row_var] = row_val
            if reversal_split:
                row = row.T
            for col_num, (col, col_val, color, ax_mean) in enumerate(zip(
                row, col_vals, row_colors, row_means)):
                if reversal_split:
                    same_ax, diff_ax = col
                else:
                    ax = col
                if bottom_margin:
                    col_summ_ax = self.display.bottom_row[col_num]
                else:
                    col_summ_ax = None
                if col_var is not None and col_val is not None:
                    subset[col_var] = col_val
                # todo: plot the indvidual saccades and a separate mean line for left- and rightward saccades
                # update the subset dictionary
                saccade_arr = []
                time_arr = []
                same_direction = []
                sample_size = len(self.trials)
                start_amps, start_times = [], []
                stop_amps, stop_times = [], []
                for trial in self.trials:
                    bouts = trial.query_bouts(sort_by=query_kwargs['sort_by'], subset=subset)
                    resps = trial.query(sort_by=query_kwargs['sort_by'], subset=subset)
                    for bout in bouts:
                        for saccade in bout.saccades:
                            peak_speed = abs(saccade.peak_velocity) * 180 / np.pi
                            if (peak_speed >= min_speed) * (peak_speed <= max_speed):
                                # get the corresponding time values
                                time = saccade.__getattribute__(time_var)
                                
                                heading = np.copy(saccade.__getattribute__(saccade_var))
                                # get the range of times to plot
                                include = (time >= start) * (time < stop)
                                # center the yvalues based on the y-intercept
                                zero_ind = np.argmin(abs(time))
                                # get the saccade start and stop indices
                                start_ind, stop_ind = saccade.start, saccade.stop
                                heading -= heading[zero_ind]
                                inds = np.where(include)[0]
                                pre_inds = inds[inds < start_ind]
                                post_inds = inds[inds > start_ind]
                                # if saccade.amplitude < 0 and positive_amplitude:
                                if np.nanmean(heading[post_inds]) < 0 and positive_amplitude:
                                    heading *= -1
                                # use this to average over the same time span for all saccades, allowing non-saccade values in the average
                                # saccade_arr += [heading[include]]
                                # time_arr += [time[include]]
                                # but to get the average including only the saccade data:
                                saccade_arr += [heading[include]]
                                time_arr += [time[include]]
                                # saccade_arr += [heading[saccade.start: saccade.stop]]
                                # time_arr += [time[saccade.start: saccade.stop]]
                                if reversal_split:
                                    # todo: this is half wrong when positive_amplitude is True
                                    # get the mean heading before 0
                                    pre_heading = np.nanmean(heading[pre_inds])
                                    post_heading = np.nanmean(heading[post_inds])
                                    same_dir = (pre_heading > 0) != (post_heading > 0)
                                    same_direction += [same_dir]
                                    # if pre_heading < 0:
                                    if same_dir:
                                        ax = same_ax
                                    else:
                                        ax = diff_ax
                                else:
                                    same_direction += [True]
                                    ax = col
                                ax.plot(heading, time, color='gray', lw=.25, alpha=.25, zorder=1)
                                ax.plot(heading[saccade.start:saccade.stop], time[saccade.start:saccade.stop], color='k', lw=.25, alpha=.5, zorder=2)
                                # plot the stop coordinate
                                stop_ind = saccade.stop
                                # ax.scatter(heading[stop_ind], time[stop_ind], marker='.', color='k', edgecolor='none', alpha=.5, zorder=2)
                                # store for the scatter plot and bar plots in bottom margins
                                stop_times += [time[stop_ind]]                                
                                start_times += [time[start_ind]]                                
                                stop_amps += [heading[stop_ind]]
                                start_amps += [heading[start_ind]]
                                # if time_var == 'relative_time':
                                #     ax.scatter(heading[start_ind], time[start_ind], marker='.', color='k', edgecolor='none', alpha=.5, zorder=2)
                                # if positive_amplitude:
                                    # plot a single mean
                                # else:
                                    # todo: plot the mean in the trace subplot and the mean +/- CI in the right margin
                                    # for each direction
                # make a scatterplot of the start and stop times
                if bottom_margin:
                    xvals = np.array(stop_amps)
                    vals, _, _ = col_summ_ax.hist(xvals, bins=np.linspace(xlim[0], xlim[1], 100), color=color, histtype='stepfilled', alpha=.75)
                    max_count = max(max_count, max(vals))
                    if time_var == 'relative_time':
                        xvals = np.array(start_amps)
                        vals, _, _ = col_summ_ax.hist(xvals, bins=np.linspace(xlim[0], xlim[1], 100), histtype='stepfilled', color=color, alpha=.5)
                        max_count = max(max_count, max(vals))
                # todo: plot different rows if reversal_split is specified
                # try:
                if not isinstance(col, (list, np.ndarray)):
                    col = [col]
                if row_summ_axes is None:
                    row_summ_axes = [None]
                for ax, row_summ_ax, same_dir in zip(col, row_summ_axes, [True, False]):
                    # get the mean time and heading within 20 evenly distributed bins
                    time_bins_pos = []
                    saccade_bins_mean_pos = []
                    saccade_bins_sem_pos = []
                    time_bins_neg = []
                    saccade_bins_mean_neg = []
                    saccade_bins_sem_neg = []
                    time_edges = np.linspace(start, stop, mean_bins)
                    signs = [1, -1]
                    if positive_amplitude:
                        # reducing to just the positive sign, the nested zip loops will skip the negative calculations
                        signs = signs[:1]
                    for time_start, time_stop in zip(time_edges[:-1], time_edges[1:]):
                        for sign, time_bins, saccade_bins_mean, saccade_bins_sem in zip(
                            signs, [time_bins_pos, time_bins_neg], 
                            [saccade_bins_mean_pos, saccade_bins_mean_neg],
                            [saccade_bins_sem_pos, saccade_bins_sem_neg]):
                            # time_bin = []
                            saccade_bin = []
                            time_bins += [np.mean([time_start, time_stop])]
                            for time, heading, direction in zip(time_arr, saccade_arr, same_direction):
                                if direction == same_dir:
                                    # include only points within the time interval
                                    include = (time >= time_start) * (time < time_stop)
                                    # and only those within headings in the same sign
                                    if np.any(include):
                                        headings_mean = np.nanmean(heading[include])
                                        if headings_mean/abs(headings_mean) == sign or positive_amplitude:
                                            saccade_bin += [heading[include]]
                                    else:
                                        saccade_bin += [np.array([np.nan])]
                            if len(saccade_bin) > 0:
                                saccade_bin = np.concatenate(saccade_bin)
                                # saccade_bins_mean += [np.nanmean(saccade_bin)]
                                saccade_bins_mean += [np.nanmedian(saccade_bin)]
                                saccade_bins_sem += [np.nanstd(saccade_bin)/np.sqrt(len(saccade_bin))]
                            else:
                                saccade_bins_mean += [np.nan]
                                saccade_bins_sem += [np.nan]
                    # make into numpy arrays
                    if not positive_amplitude:
                        saccade_bins_mean_neg, saccade_bins_sem_neg = np.array(saccade_bins_mean_neg), np.array(saccade_bins_sem_neg)
                    time_bins = np.array(time_bins)
                    saccade_bins_mean_pos, saccade_bins_sem_pos = np.array(saccade_bins_mean_pos), np.array(saccade_bins_sem_pos)
                    for saccade_bins_mean, sign in zip([saccade_bins_mean_pos, saccade_bins_mean_neg], signs):
                        ax.plot(saccade_bins_mean, time_bins, color='w', zorder=3, lw=2)
                        ax.plot(saccade_bins_mean, time_bins, color=color, zorder=4, linestyle=['-', ':'][int(sign > 0)])
                    # plot mean +/- SEM
                    if right_margin:
                        for sign, saccade_bins_mean, saccade_bins_sem in zip(
                            signs,
                            [saccade_bins_mean_pos, saccade_bins_mean_neg],
                            [saccade_bins_sem_pos, saccade_bins_sem_neg]):
                            row_summ_ax.plot(saccade_bins_mean, time_bins, color=color, alpha=.5, zorder=3)
                            # plot the error
                            # row_summ_ax.errorbar(
                            #     saccade_bins_mean, time_bins, xerr=saccade_bins_sem, 
                            #     color=color, alpha=.5, zorder=2)
                            # plot the mean stop amplitude and time
                            # mean_stop_amp = np.mean(stop_amps)
                            # mean_stop_time = np.mean(stop_times)
                            # todo: replace with arrows
                            # row_summ_ax.scatter(mean_stop_amp, mean_stop_time, color=color, zorder=3, marker='o', edgecolor='none', alpha=.5)
                            # scale = 1
                            # row_summ_ax.scatter(mean_stop_amp, mean_stop_time, color='w', zorder=5, marker='X', edgecolor='none', alpha=1, lw=5)
                            # row_summ_ax.scatter(mean_stop_amp, mean_stop_time, color=color, zorder=6, marker='X', edgecolor='none', alpha=.75)
                            # row_summ_ax.scatter(mean_stop_amp, mean_stop_time, color=color, zorder=4, marker='X', alpha=1)
                            # if time_var == 'relative_time':
                            #     mean_start_amp = np.mean(start_amps)
                            #     mean_start_time = np.mean(start_times)
                            #     # row_summ_ax.scatter(mean_start_amp, mean_start_time, color='w', zorder=5, marker='X', edgecolor='none', alpha=1, lw=5)
                            #     row_summ_ax.scatter(mean_start_amp, mean_start_time, color=color, zorder=4, marker='X', alpha=1)

                    # if bottom_margin:
                    #     for sign, saccade_bins_mean, saccade_bins_sem in zip(
                    #         signs,
                    #         [saccade_bins_mean_pos, saccade_bins_mean_neg],
                    #         [saccade_bins_sem_pos, saccade_bins_sem_neg]):
                            # col_summ_ax.plot(saccade_bins_mean, time_bins, color=color)
                            # col_summ_ax.errorbar(
                            #     saccade_bins_mean, time_bins, xerr=saccade_bins_sem, 
                            #     color=color, alpha=.5)
                            # col_summ_ax.fill_betweenx(
                            #     time_bins, saccade_bins_mean - saccade_bins_sem, 
                            #     saccade_bins_mean + saccade_bins_sem, color=color, alpha=.3,
                            #     linewidth=0.0)
                # except:
                #     breakpoint()
        # breakpoint()
        # add the xticks, yticks, and axis labels
        self.display.format(xlim=xlim, ylim=ylim, 
                            xlabel='heading', ylabel=time_var+' (s)', 
                            xticks=xticks, yticks=yticks,
                            special_bottom_left=bottom_margin)
        # format the bottom and right margins to show the 
        if bottom_margin:
            bottom_row = self.display.bottom_row
            for num, (ax) in enumerate(bottom_row):
                ax.set_ylim(0, max_count)
                if num == 0:
                    ax.set_yticks([0, max_count])
                    ax.set_ylabel("count")
                else:
                    ax.set_yticks([])
                sbn.despine(ax=ax, bottom=False, left=num!=0)
        if right_margin and bottom_margin:
            self.display.corner_ax.axis('off')
        # add the row values
        # make specific labels if reversal_split
        if reversal_split:
            new_row_vals = []
            for val in row_vals:
                # new_row_vals += [val]
                # new_row_vals += [val]
                new_row_vals += [f"same\n{val}"]
                new_row_vals += [f"reversal\n{val}"]
            row_vals = new_row_vals
        self.display.label_margins(row_vals, row_var, col_vals, col_var)
        # add the sample size to the first subplot
        self.display.fig.suptitle(f"N={sample_size}")

    def plot_saccade_dynamics(self, col_var, row_var, row_cmap=None, col_cmap=None,
                              fig=None, right_margin=True, bottom_margin=True, scale=1.5, 
                              reversal_split=False, output='start', 
                              heading_var='camera_heading', reference_var=None, bins=21,
                              min_speed=350, max_speed=np.inf, **query_kwargs):
        """Plot saccade position (x) and amplitude (y) in a grid as in the plot summary below.
        
        These plots will allow us to measure the 

        Parameters
        ----------
        col_var, row_var : str
            The variable names to paramaterize along columns and rows of the grid.
        time_var : str, default='time'
            The time variable to use. So far there are two options: 1) time from the start 
            of the saccade or 2) time from the peak velocity.
        right_margin, bottom_margin : bool, default=True
            Whether to plot the means from each subplot in the margin to the right
            or bottom of the axes. A colormap (row_cmap or col_cmap) must be provided
            in order to distinguish individual traces.
        output : str, default='start'
            Whether to use the start or stop position of the saccade.
        min_speed, max_speed : float, default=350, np.inf
            The minimum and maximum peak speed to include in the saccades here.
        heading_var : str, default='camera_heading'
            The variable to use for the saccade heading.
        reference_var : str, default=None
            If provided, the saccades will be aligned to the value of this variable.
        bins : int, default=21
        **query_kwargs
            These get passed to the query 
        """
        # for now, no bottom or right margins
        bottom_margin, right_margin = False, False
        assert output in ['start', 'stop'], "output parameter must be either 'start' or 'stop'"
        if 'subset' not in query_kwargs:
            query_kwargs['subset'] = {}
        if 'sort_by' not in query_kwargs:
            query_kwargs['sort_by'] = 'test_ind'
        subset = copy.copy(query_kwargs['subset'])
        # get the values used for coloring each subplot
        if row_var is not None:
            row_vals = self.query(output=row_var, sort_by=row_var, subset=subset)
        else:
            row_vals = [None]
        if col_var is not None:
            col_vals = self.query(output=col_var, sort_by=col_var, subset=subset)
        else:
            col_vals = [None]
        assert len(col_vals) > 0 or len(row_vals) > 0, "The subset is empty!"
        # todo: what's up with the number of axes?
        row_vals = np.unique(row_vals)
        col_vals = np.unique(col_vals)
        num_rows, num_cols = len(row_vals), len(col_vals)
        # get the colors from the specified colormaps
        # get the colors from the specified colormaps
        colors = {}
        for cmap, vals, key in zip(
            [row_cmap, col_cmap], 
            [row_vals, col_vals],
            ['rows', 'columns']):
            if len(vals) > 0:
                if isinstance(cmap, str):
                    if isinstance(vals[0], (str, bytes)):
                        # if values are strings, sort them in alphabetical order and use their index for the colors
                        vals = np.argsort(vals)
                    norm = matplotlib.colors.Normalize(vals.min(), vals.max())
                    cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
                    colors[key] = cmap.to_rgba(vals)[:, :-1]
                elif callable(cmap):
                    colors[key] = cmap(vals)
                elif isinstance(row_cmap, (list, np.ndarray, tuple)):
                    assert len(cmap) == len(vals), (
                        f"Colormap list has {len(cmap)} elements but {len(vals)} {key}.")
                    colors[key] = row_cmap
                else:
                    colors[key] = []
            else:
                colors[key] = []
        # combine the color lists to make an array specifying the color of each subplot
        if len(colors['rows']) == num_rows and len(colors['columns']) == num_cols:
            # get the mean comination of row and column colors
            # try:
            if num_rows > 0 and num_cols > 0:
                color_mean = .5*(
                    colors['rows'][:, np.newaxis]**2 +
                    colors['columns'][np.newaxis, :]**2)
            else:
                max_ind = np.argmax([len(colors[key]) for key in colors.keys()])
                color_mean = tuple(colors.values())[max_ind]
                if num_cols == 0:
                    num_cols = 1
                if num_rows == 0:
                    num_rows = 1
            color_arr = np.sqrt(color_mean)
        elif len(colors['columns']) == num_cols:
            color_arr = np.repeat(colors['columns'][np.newaxis], num_rows, axis=0)
        elif len(colors['rows']) == num_rows:
            color_arr = np.repeat(colors['rows'][:, np.newaxis], num_cols, axis=1)
        else:
            color_arr = np.zeros((num_rows, num_cols, 3), dtype='uint8')
        # todo: add extra rows if split_reversal
        # test: check that the colors are aligned properly. we want this array shape to be num_rows X num_cols
        # add a row or column if plotting in the margins
        if bottom_margin:
            num_rows += 1
        if right_margin:
            num_cols += 1
        # make the figure and axes with a grid defined by num_rows and num_cols
        figsize = (scale * num_cols + 1, scale*num_rows + 1)
        self.display = SummaryDisplay(
            num_rows=num_rows, num_cols=num_cols, right_margin=right_margin, 
            bottom_margin=bottom_margin, figsize=figsize)
        trace_axes = self.display.trace_axes
        # plot the data
        num_frames = self.trials[0].num_frames
        means = np.zeros((trace_axes.shape[0], trace_axes.shape[1], int(num_frames)), dtype=float)
        max_count = 0
        # split trace axes if reversal split
        if reversal_split:
            new_num_rows = round(num_rows / 2)
            trace_axes = trace_axes.reshape((new_num_rows, 2, trace_axes.shape[1]))
        for row_num, (row, row_val, row_colors, row_means) in enumerate(zip(
            trace_axes, row_vals, color_arr, means)):
            if right_margin:
                if reversal_split:
                    row_summ_axes = self.display.right_col[2 * row_num: 2 * row_num + 2]
                else:
                    row_summ_axes = [self.display.right_col[row_num]]
            else:
                row_summ_axes = None
            # update the subset dictionary
            if row_var is not None and row_val is not None:
                subset[row_var] = row_val
            if reversal_split:
                row = row.T
            for col_num, (col, col_val, color, ax_mean) in enumerate(zip(
                row, col_vals, row_colors, row_means)):
                if reversal_split:
                    same_ax, diff_ax = col
                else:
                    ax = col
                if bottom_margin:
                    col_summ_ax = self.display.bottom_row[col_num]
                else:
                    col_summ_ax = None
                if col_var is not None and col_val is not None:
                    subset[col_var] = col_val
                # plot the indvidual saccades and a separate mean line for left- and rightward saccades
                # update the subset dictionary
                sample_size = len(self.trials)
                start_pos, stop_pos = [], []
                amps = []
                # collect the saccade starting positions and amplitudes for making a 2D plot 
                for trial in self.trials:
                    bouts = trial.query_bouts(sort_by=query_kwargs['sort_by'], subset=subset)
                    resps = trial.query('camera_heading', sort_by=query_kwargs['sort_by'], subset=subset)
                    bar_positions = trial.query('bar_orientation', sort_by=query_kwargs['sort_by'], subset=subset)
                    for bout, bar_position, resp in zip(bouts, bar_positions, resps):
                        bar_position = np.unwrap(bar_position)
                        bar_position += np.pi
                        bar_position %= 2*np.pi
                        bar_position -= np.pi
                        for saccade in bout.saccades:
                            peak_speed = abs(saccade.peak_velocity) * 180 / np.pi
                            if (peak_speed >= min_speed) * (peak_speed <= max_speed):
                                # store the start position, stop position, and signed amplitude
                                # todo: get the position of the bar and calculate heading angles relative to the bar
                                for storage, pos in zip([start_pos, stop_pos], [saccade.start, saccade.stop]):
                                    bar = bar_position[pos]
                                    head = resp[pos]
                                    storage += [bar - head]
                                amps += [saccade.amplitude]
                    resps = trial.query(heading_var, sort_by=query_kwargs['sort_by'], subset=subset)
                    # convert the bar_positions variable to a general reference angle input parameter
                    if reference_var in dir(trial):
                        reference_positions = trial.query(reference_var, sort_by=query_kwargs['sort_by'], subset=subset)
                        for bout, reference_position, resp in zip(bouts, reference_positions, resps):
                            reference_position = np.unwrap(reference_position)
                            reference_position += np.pi
                            reference_position %= 2*np.pi
                            reference_position -= np.pi
                            times, saccades = bout.query_saccades(sort_by=query_kwargs['sort_by'], subset=subset)
                            breakpoint()
                            for saccade in saccades:
                                peak_speed = abs(saccade.peak_velocity) * 180 / np.pi
                                if (peak_speed >= min_speed) * (peak_speed <= max_speed):
                                    # store the start position, stop position, and signed amplitude
                                    # todo: get the position of the bar and calculate heading angles relative to the bar
                                    for storage, pos in zip([start_pos, stop_pos], [saccade.start, saccade.stop]):
                                        bar = reference_position[pos]
                                        head = resp[pos]
                                        storage += [bar - head]
                                    amps += [saccade.amplitude]
                    # use no reference angle if reference_var is not in the trial data
                    else:
                        for bout, resp in zip(bouts, resps):
                            for saccade in bout.saccades:
                                peak_speed = abs(saccade.peak_velocity) * 180 / np.pi
                                if (peak_speed >= min_speed) * (peak_speed <= max_speed):
                                    # store the start position, stop position, and signed amplitude
                                    # get the position of the bar and calculate heading angles
                                    for storage, pos in zip([start_pos, stop_pos], [saccade.start, saccade.stop]):
                                        head = resp[pos]
                                        storage += [head]
                if not isinstance(col, (list, np.ndarray)):
                    col = [col]
                amplitude = np.array(amps)
                if output == 'start':
                    position = np.array(start_pos)
                else:
                    position = np.array(stop_pos)
                position += np.pi
                position %= 2*np.pi
                position -= np.pi
                for ax, row_summ_ax in zip(col, row_summ_axes):
                    data = {'amplitude':amplitude, 'position':position}
                    # ax = plt.gca()
                    # sbn.kdeplot(data=data, x='position', y='amplitude', ax=ax, levels=20, fill=True, cmap='Greys')
                    # ax.scatter(position, amplitude, marker='.', alpha=.1, color='k', edgecolor='none')
                    ax.hist2d(position, amplitude, bins=21, range=[[-np.pi, np.pi],[-np.pi, np.pi]], cmap='Greys')
                    # ax.set_xlabel(None)
                    # ax.set_ylabel(None)
                    # ax.set_xlabel('position')
                    # ax.set_ylabel('amplitude')
                    ax.set_aspect('equal')
                    # ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['-$\pi$', '-$\pi$/2', '0', '$\pi$/2', '$\pi$'])
                    # ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['-$\pi$', '-$\pi$/2', '0', '$\pi$/2', '$\pi$'])
                    # plot mean +/- SEM
        # add the xticks, yticks, and axis labels
        ticks = ([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['-$\pi$', '-$\pi$/2', '0', '$\pi$/2', '$\pi$'])
        self.display.format(xlim=(np.pi, np.pi), ylim=(-1.25*np.pi, 1.25*np.pi), 
                            xlabel='orientation', ylabel='amplitude', 
                            xticks=ticks, yticks=ticks)
        plt.show()
        # format the bottom and right margins to show the 
        if bottom_margin:
            bottom_row = self.display.bottom_row
            for num, (ax) in enumerate(bottom_row):
                ax.set_ylim(0, max_count)
                if num == 0:
                    ax.set_yticks([0, max_count])
                    ax.set_ylabel("count")
                else:
                    ax.set_yticks([])
                sbn.despine(ax=ax, bottom=False, left=num!=0)
            self.display.corner_ax.axis('off')
        # add the row values
        # make specific labels if reversal_split
        if reversal_split:
            new_row_vals = []
            for val in row_vals:
                # new_row_vals += [val]
                # new_row_vals += [val]
                new_row_vals += [f"same\n{val}"]
                new_row_vals += [f"reversal\n{val}"]
            row_vals = new_row_vals
        self.display.label_margins(row_vals, row_var, col_vals, col_var)
        # add the sample size to the first subplot
        self.display.fig.suptitle(f"N={sample_size}")

    def main_sequence_analysis(self, group_var='bg_gain', cmap='viridis', scale=1, subset={}):
        """Plot the relation between saccade peak velocity, duration, and magnitude. 
        
        Parameters
        ----------
        group_var : str, default='bg_gain'
            The variable to use for subsetting the saccade bouts and coloring individually.
        cmap : str or func
            The colormap applied to the group values.
        scale : float, default=1
            Scale parameter for determining the figure size. 1 results in a 3x5 figure.
        subset : dict, default={}
            Optionally filter data before plotting. See TrackingTrial.query for more info.
        """
        # allow for filtering of the data using subset
        group_vals = np.unique(self.query(output=group_var, sort_by=group_var, subset=subset)[0])
        # get the color for each group
        if isinstance(cmap, str):
            norm = matplotlib.colors.Normalize(group_vals.min(), group_vals.max())
            cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            colors = cmap.to_rgba(group_vals)[:, :-1]
        elif callable(cmap):
            colors = cmap(group_vals)
        elif isinstance(cmap, (list, np.ndarray, tuple)):
            assert len(cmap) == len(group_vals), (
                f"Colormap list has {len(cmap)} elements but there are {len(group_vals)} colors listed.")
            colors = cmap
        # make a plot with subplots for saccade peak velocity and duration
        # but also include marginal plots for boxplot comparisons
        fig, axes = plt.subplots(nrows=3, ncols=2, width_ratios=[1, 1], height_ratios=[1, 1, 1], figsize=(4*scale, 6*scale))
        # turn off the corner subplot
        axes[-1, -1].axis('off')
        bottom_ax = axes[-1, 0]
        right_col = axes[:-1, -1]
        scatter_axes = axes[:-1, 0]
        # for each group, plot the peak velocity and duration as a function of magnitude
        dur_meds, speed_meds, mag_meds = [], [], []
        for num, (group, color) in enumerate(zip(group_vals, colors)):
            subset[group_var] = group
            peak_velo = abs(np.array(self.query(output='saccade_peak_velocity', sort_by=group_var, subset=subset)))
            peak_velo *= 180. / np.pi
            duration = np.array(self.query(output='saccade_duration', sort_by=group_var, subset=subset))
            amplitude = abs(np.array(self.query(output='saccade_amplitude', sort_by=group_var, subset=subset)))
            sizes = np.unique([arr.size for arr in amplitude])
            same_sizes = len(sizes) == 1
            # todo: get the 95% CI of the mean duration, amplitude, and speed
            # 0. for each variable, 
            for var, storage in zip([peak_velo, duration, amplitude], [speed_meds, dur_meds, mag_meds]):
                # 1. get the mean per trial
                if same_sizes:
                    avg = np.nanmean(var, axis=1)
                else:
                    avg = np.array([np.nanmean(subvar) for subvar in var])
                # later, bootstrap entire sequences, one per subject
                storage += [avg]
            # dur_meds += [np.nanmedian(duration)]
            # mag_meds += [np.nanmedian(amplitude)]
            # speed_meds += [np.nanmedian(peak_velo)]
            for ys, ax, right_ax in zip([duration, peak_velo], scatter_axes, right_col):
                # scatterplot for the main sequence
                # check if the subsets are of different sizes
                if len(sizes) > 1:
                    for yvals, amp in zip(ys, amplitude):
                        ax.scatter(amp, yvals, color=color, marker='.', edgecolor='none', alpha=.5)
                else:
                    ax.scatter(amplitude, ys, color=color, marker='.', edgecolor='none', alpha=.5)
                # jitterplot of yvalues in the right axis
                # xjitter = np.random.normal(0, .1, size=len(ys))
                # xvals = num + xjitter
                # right_ax.scatter(xvals, ys, color=color, marker='.', edgecolor='none', alpha=.5)
                # todo: bootstrap 84% confidence intervals for comparing the group means
            # jitterplot of the group values on the margins
            # yjitter = np.random.normal(0, .1, size=len(ys))
            # yvals = num + yjitter
            # bottom_ax.scatter(amplitude, yvals, color=color, marker='.', edgecolor='none', alpha=.5)
        # plot the medians:
        dur_meds, speed_meds, mag_meds = np.array(dur_meds), np.array(speed_meds), np.array(mag_meds)
        # get the bootstrapped 95% CI for each group val by randomly sampling on a per-subject basis
        dur_lows, dur_mids, dur_highs = [], [], []
        speed_lows, speed_mids, speed_highs = [], [], []
        mag_lows, mag_mids, mag_highs = [], [], []
        num_groups, num_trials = dur_meds.shape
        for lows, mids, highs, vals in zip(
            [dur_lows, speed_lows, mag_lows], 
            [dur_mids, speed_mids, mag_mids],
            [dur_highs, speed_highs, mag_highs],
            [dur_meds, speed_meds, mag_meds]):
            # vals has shape len(group_vals) x len(self.trials)
            inds = np.random.randint(0, num_trials, size=(len(self.trials), 10000))
            # note: each vals has a mean value per subject
            pseudo_distro = vals[:, inds]
            pseudo_distro = np.nanmean(pseudo_distro, axis=1)
            low, mid, high = np.percentile(pseudo_distro, [8, 50, 92], axis=-1)
            mids += [mid]
            lows += [low]
            highs += [high]
        dur_lows, dur_mids, dur_highs = dur_lows[0], dur_mids[0], dur_highs[0]
        speed_lows, speed_mids, speed_highs = speed_lows[0], speed_mids[0], speed_highs[0]
        mag_lows, mag_mids, mag_highs = mag_lows[0], mag_mids[0], mag_highs[0]
        # dur_meds, speed_meds, mag_meds = dur_meds.mean(-1), speed_meds.mean(-1), mag_meds.mean(-1)
        for ax, lows, meds, highs, vals in zip(
            right_col[:2], [dur_lows, speed_lows], [dur_mids, speed_mids], [dur_highs, speed_highs], [dur_meds, speed_meds]):
            for val in vals.T:
                ax.plot(range(len(val)), val, color='gray', alpha=.25, zorder=2)
                ax.scatter(range(len(val)), val, c=colors, edgecolors='none', marker='o', zorder=3, alpha=.5)
            for num, (low, meds, high) in enumerate(zip(lows, meds, highs)):
                ax.plot([num, num], [low, high], color='k', zorder=4)
                ax.scatter(num, np.nanmean(meds, axis=-1), color='k', marker='o', edgecolors='w', linewidths=2, zorder=5)
        lows, meds, highs = mag_lows, mag_meds, mag_highs
        ax = bottom_ax
        for vals in meds.T:
            ax.plot(vals, range(len(vals)), color='gray', alpha=.25, zorder=2)
            ax.scatter(vals, range(len(vals)), c=colors, edgecolors='none', marker='o', zorder=3, alpha=.5)
        for num, (low, meds, high) in enumerate(zip(lows, meds, highs)):
            ax.plot([low, high], [num, num], color='k', zorder=4)
            ax.scatter(np.nanmean(meds, axis=-1), num, color='k', marker='o', edgecolors='w', linewidths=2, zorder=5)
        # formatting:
        # transform the x-axes to log scale
        for ax in [scatter_axes[0], scatter_axes[1], bottom_ax]:
            ax.set_xscale('log')
        # transform the y-axes to log scale
        for ax in [scatter_axes[0], scatter_axes[1], right_col[0], right_col[1]]:
        # for ax in [scatter_axes[0], right_col[0]]:
            ax.set_yscale('log')
        # remove the minor ticks
        for ax in [right_col[0], right_col[1], bottom_ax]:
            ax.minorticks_off()
        for ax in scatter_axes:
            ax.tick_params(axis='x', which='minor', bottom=False)
        # set limits on the scatterplots
        ylims = []
        scatter_axes[1].set_ylim(0)
        # xmin = scatter_axes[0].get_xlim()[0]
        for ax in scatter_axes:
            # ax.set_xlim(0)
            # ax.set_ylim(0)
            ylims += [ax.get_ylim()]
        #     xmin = min(xmin, ax.get_xlim()[0])
        # for ax in scatter_axes:
        #     ax.set_xlim(xmin, np.pi)
        # xmin = np.pi / 16
        xmin = np.pi/128
        xmax = 2*np.pi
        for ax in np.append(scatter_axes, [bottom_ax]):
            ax.set_xlim(xmin, xmax)
        # set limits on the marginal plots to match the scatterplots
        for ax, ylim in zip(right_col, ylims):
            ax.set_ylim(ylim[0], ylim[1])
        # remove the bottom spines of both scatter axes
        for ax, lbl in zip(scatter_axes, ["duration (s)", "peak speed ($\degree$/s)"]):
            ax.set_xticks([])
            # ax.set_yticks([])
            sbn.despine(ax=ax, bottom=True, trim=False)
            # label the y-axis
            ax.set_ylabel(lbl)
        # remove the bottom spines of the top right axis
        right_col[0].set_xticks([])
        right_col[0].set_yticks([])
        sbn.despine(ax=right_col[0], bottom=True, left=True, trim=True)
        # remove the left spines of the top right axis
        right_col[1].set_xticks(range(len(group_vals)), group_vals)
        right_col[1].set_yticks([])
        # label the x-axis
        right_col[1].set_xlabel(group_var.replace("_", " "))
        sbn.despine(ax=right_col[1], bottom=False, left=True, trim=True)
        # trim spines for the bottom ax
        bottom_ax.set_yticks(range(len(group_vals)), group_vals)
        # bottom_ax.set_xticks([np.pi/16, np.pi/8, np.pi/4, np.pi/2, np.pi, 2*np.pi], ['$\pi$/16', '$\pi$/8', '$\pi$/4', '$\pi$/2', '$\pi$', '2$\pi$'])
        bottom_ax.set_xticks([np.pi/64, np.pi/32, np.pi/16, np.pi/8, np.pi/4, np.pi/2, np.pi], ['$\pi$/64', '$\pi$/32', '$\pi$/16', '$\pi$/8', '$\pi$/4', '$\pi$/2', '$\pi$'])
        # bottom_ax.set_xlim(xmin, xmax)
        # label the bottom axis
        bottom_ax.set_xlabel("magnitude")
        bottom_ax.set_ylabel(group_var.replace("_", " "))
        sbn.despine(ax=bottom_ax, trim=True)
        plt.tight_layout()
        # plt.show()

    def plot_summary(self, xvar, yvar, col_var, row_var, 
                     row_cmap=None, col_cmap=None, fig=None,
                     right_margin=True, bottom_margin=True,
                     xlim=None, ylim=None, xticks=None, yticks=None, 
                     logx=False, logy=False, display=None,
                     summary_func=np.nanmean, xlabel=None, ylabel=None,
                     plot_type='line', bins=100, use_density=False,
                     confidence_interval=False, confidence=.84,
                     **query_kwargs):
        """Plot experimental data in one big grid.
        
        The color for each subplot is determined by the average of the column and 
        row colors: color = sqrt(mean([col_color^2, row_color^2])). This is the 
        proper way to average two colors and should generate a unique color for each 
        plot. 

        todo: add option to define colors in HSV space so that the user can
        align rows and columns with independent channels. For instance, rows can
        correspond to different hues while columns correspond to saturations. 

        Parameters
        ----------
        xvar, yvar : str
            The variable names to plot along the x- and y-axis of each subplot.
        col_var, row_var : str
            The variable names to paramaterize along columns and rows of the grid.
        row_cmap, col_cmap : func or array-like, default=None
            The colormap to use for colorizing along the rows or columns. If both are
            supplied, the product of the two colors is used for each subplot. If a 
            function is supplied, it will be applied to the corresponding col_var or row_var.
            If a list is supplied, it must have as many elements as the corresponding column or
            row.
        right_margin, bottom_margin : bool, default=True
            Whether to plot the means from each subplot in the margin to the right
            or bottom of the axes. A colormap (row_cmap or col_cmap) must be provided
            in order to distinguish individual traces.
        xlim, ylim : tuple=(min, max), default=None
            The tuple of the minimum and maximum values for that dimension.
        xticks, yticks : list, default=None
            Specify the list of ticks on the x- or y-axis.
        logx, logy : bool, default=False
            Whether to format the x- or y-axis on a log scale.
        display : SummaryDisplay, default=None
            Option to provide a SummaryDisplay with the same subplot arrangement
            to allow superimposing different datasets.
        xlabel, ylabel : str, default=None
            Option to provide custom x- and y-labels.
        summary_func : callable, default=np.nanmean
            The function for generating the summary statistic of interest in the right and bottom
            margins. It's applied to the sets of traces plotted in the trace plots.
        plot_type : str, default='line'
            Whether to plot the data as a line plot ('line'), a 2D histogram ('hist2d'),
            or as 2d simulated trajectory ('trajectory2d').
        bins : int or tuple, default=None
            The bins parameter to pass to the 2D histogram.
        use_density : bool, default=False
            Whether to plot the density instead of the count in the 2d histogram.
        **query_kwargs
            These get passed to the query 
        """
        if 'subset' not in query_kwargs:
            query_kwargs['subset'] = {}
        if 'sort_by' not in query_kwargs:
            query_kwargs['sort_by'] = 'test_ind'
        subset = copy.copy(query_kwargs['subset'])
        # get the values used for coloring each subplot
        row_vals = self.query(output=row_var, sort_by=row_var, subset=subset)
        col_vals = self.query(output=col_var, sort_by=col_var, subset=subset)
        assert len(col_vals) > 0 or len(row_vals) > 0, "The subset is empty!"
        # todo: what's up with the number of axes?
        row_vals = np.unique(row_vals)
        col_vals = np.unique(col_vals)
        num_rows, num_cols = len(row_vals), len(col_vals)
        # get the colors from the specified colormaps
        colors = {}
        for cmap, vals, key in zip(
            [row_cmap, col_cmap], 
            [row_vals, col_vals],
            ['rows', 'columns']):
            if len(vals) > 0:
                if isinstance(cmap, str):
                    if isinstance(vals[0], (str, bytes)):
                        # if values are strings, sort them in alphabetical order and use their index for the colors
                        vals = np.argsort(vals)
                    norm = matplotlib.colors.Normalize(vals.min(), vals.max())
                    cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
                    colors[key] = cmap.to_rgba(vals)[:, :-1]
                elif callable(cmap):
                    colors[key] = cmap(vals)
                elif isinstance(row_cmap, (list, np.ndarray, tuple)):
                    assert len(cmap) == len(vals), (
                        f"Colormap list has {len(cmap)} elements but {len(vals)} {key}.")
                    colors[key] = row_cmap
                else:
                    colors[key] = []
            else:
                colors[key] = []
        # combine the color lists to make an array specifying the color of each subplot
        if len(colors['rows']) == num_rows and len(colors['columns']) == num_cols:
            # get the mean combination of row and column colors
            color_mean = .5*(
                colors['rows'][:, np.newaxis]**2 +
                colors['columns'][np.newaxis, :]**2)
            color_arr = np.sqrt(color_mean)
            # elif colors['columns'] is not None:
            #     color_arr = np.repeat(colors['columns'][np.newaxis], num_rows, axis=0)
            # elif colors['rows'] is not None:
            #     color_arr = np.repeat(colors['rows'][:, np.newaxis], num_cols, axis=0)
        elif len(colors['columns']) == num_cols:
            color_arr = np.repeat(colors['columns'][np.newaxis], num_rows, axis=0)
        elif len(colors['rows']) == num_rows:
            color_arr = np.repeat(colors['rows'][:, np.newaxis], num_cols, axis=1)
        else:
            color_arr = np.zeros((num_rows, num_cols, 3), dtype='uint8')
        # test: check that the colors are aligned properly. we want this array shape to be num_rows X num_cols
        # add a row or column if plotting in the margins
        if bottom_margin:
            num_rows += 1
        if right_margin:
            num_cols += 1
        # make the figure and axes with a grid defined by num_rows and num_cols
        figsize = (1.5 * num_cols + 1, 1.5*num_rows + 1)
        format_axes = True
        if display is None:
            self.display = SummaryDisplay(
                num_rows=num_rows, num_cols=num_cols, right_margin=right_margin, 
                bottom_margin=bottom_margin, figsize=figsize)
        else:
            self.display = display
            format_axes = False
        trace_axes = self.display.trace_axes
        # plot the data
        num_frames = self.trials[0].num_frames
        repetitions = []
        max_radius = 0
        for row_num, (row, row_val, row_colors) in enumerate(zip(
            trace_axes, row_vals, color_arr)):
            if right_margin:
                row_summ_ax = self.display.right_col[row_num]
            else:
                row_summ_ax = None
            # update the subset dictionary
            subset[row_var] = row_val
            for col_num, (ax, col_val, color) in enumerate(zip(
                row, col_vals, row_colors)):
                while isinstance(ax, np.ndarray):
                    ax = ax[0]
                if bottom_margin:
                    col_summ_ax = self.display.bottom_row[col_num]
                else:
                    col_summ_ax = None
                subset[col_var] = col_val
                # todo: get the subset x- and y-values
                # update the subset dictionary
                xs = self.query(same_size=True, output=xvar, subset=subset, sort_by=query_kwargs['sort_by'])
                ys = self.query(same_size=True, output=yvar, subset=subset, sort_by=query_kwargs['sort_by'])
                # todo: pad arrays in included_xs to match the longest one and convert to an array
                # get sample size
                # sample_size = xs[0].shape[0]
                # breakpoint()
                # if callable(summary_func):
                #     # ax_mean = np.nanmean(xs, axis=0)
                #     ax_mean = summary_func(xs, axis=0)
                #     y = ys[0]
                #     # mean_vals = np.array(mean_vals)
                #     # ax_mean = mean_vals.mean(0)
                #     ax.plot(ax_mean, y, color='w', zorder=4, lw=4)
                #     ax.plot(ax_mean, y, color=color, zorder=5)
                if isinstance(xs, np.ndarray):
                    sample_size = xs.shape[0]
                    if xs.size > 0:
                        reps = xs.shape[1]
                        # reshape the data
                        ys = ys.reshape(-1, ys.shape[-1])
                        xs = xs.reshape(-1, xs.shape[-1])
                        # plot individual traces and the mean
                        # mean_vals = []
                        # for num, x in enumerate(xs):
                        #     if x.size > 0:
                        #         y = ys[num]
                        #         breakpoint()
                        #         ax.plot(x, y, color='gray', lw=.5, alpha=.5)
                        #         mean_vals += [x]
                        nans = np.isnan(xs)
                        not_all_nans = nans.mean(1) < 1
                        new_xs, new_ys = xs[not_all_nans], ys[not_all_nans]
                        if plot_type == 'hist2d':
                            # take histogram of the x and y data in order to get the maximum
                            # fig, ax = plt.subplots()
                            no_nans = np.isnan(new_xs) == False
                            hist, xedges, yedges = np.histogram2d(new_xs[no_nans], new_ys[no_nans], bins=bins, density=use_density)
                            # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(1,1,1), color])
                            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(1,1,1), (0, 0, 0)])
                            max_val = hist.max()
                            mesh = ax.pcolormesh(xedges, yedges, hist.T, cmap=cmap, vmin=0, vmax=hist.max())
                            # plt.colorbar(mesh, ax=ax)
                        elif plot_type == 'trajectory2d':
                            # todo: plot the 2d trajectory for each fly 
                            d_vectors = np.array([np.cos(new_xs), np.sin(new_xs)]).transpose(1, 0, 2)
                            # replace d_vectors with 0 if nan
                            d_vectors[np.isnan(d_vectors)] = 0
                            trajectory = np.cumsum(d_vectors, axis=-1)
                            trajectory /= trajectory.shape[-1]
                            # find the last non-nan number in the trajectory
                            last_pos = trajectory[..., -1]
                            radii = np.linalg.norm(last_pos, axis=1)
                            # normalize the trajectory to the maximum radius
                            # trajectory /= radii[:, np.newaxis, np.newaxis]
                            ax.plot(trajectory[:, 0].T, trajectory[:, 1].T, lw=.5, color='k', alpha=.25)
                            # todo: plot the radius and corresponding circle for each fly
                            max_radius = max(np.nanmax(radii), max_radius)
                            # scatterplot of the last position
                            ax.scatter(last_pos[..., 0], last_pos[..., 1], color='k', marker='.', s=.5)
                            for radius, pos in zip(radii, last_pos):
                                circle = plt.Circle((0, 0), radius=radius, color=color, alpha=.25, fill=False, lw=.5)
                                ax.add_artist(circle)
                        else:
                            ax.plot(new_xs.T, new_ys.T, color='gray', lw=.5, alpha=.5)
                            # if np.squeeze(xs).ndim > 1:
                            #     # ax.plot(xs.T, ys.T, color='gray', lw=.5, alpha=.5)
                            #     reps = 0
                            #     for num, (x, y) in enumerate(zip(xs, ys)):
                            #         try:
                            #             ax.plot(x, y, color='gray', lw=.5, alpha=.5)
                            #             reps += 1
                            #         except:
                            #             pass
                            # else:
                        no_nans = np.isnan(ys) == False
                        all_no_nans = np.all(no_nans, axis=1)
                        y = ys[all_no_nans][0]
                        if callable(summary_func) and plot_type in ['hist2d', 'line']:
                            # ax_mean = np.nanmean(xs, axis=0)
                            ax_mean = summary_func(xs, axis=0)
                            # mean_vals = np.array(mean_vals)
                            # ax_mean = mean_vals.mean(0)
                            if plot_type == 'hist2d':
                                ax.scatter(ax_mean, y, color='w', zorder=4, marker='.', s=1)
                                ax.scatter(ax_mean, y, color=color, zorder=5, marker='.', s=.5)
                            else:
                                ax.plot(ax_mean, y, color='w', zorder=4, lw=1)
                                ax.plot(ax_mean, y, color=color, zorder=5)
                        # plot the mean in the two summary plots
                        if row_summ_ax is not None:
                            if plot_type in ['hist2d', 'line']:
                                if callable(summary_func):
                                    summary = summary_func(xs, axis=0)
                                    row_summ_ax.plot(summary, y, color=color, zorder=1)
                                    if confidence_interval:
                                        # todo: get the 84% C.I. for each time point using bootstrapping
                                        tot = xs.shape[0]
                                        # todo: use a for loop to avoid loading the full pseudo distribution
                                        rand_samples = np.random.randint(0, tot-1, (tot, 1000))
                                        lows, highs = [], []
                                        for frame_vals in xs.T:
                                            pseudo_distro = frame_vals[rand_samples]
                                            summary = summary_func(pseudo_distro, axis=0)
                                            lb, ub = (1 - confidence)/2, (1 - confidence)/2
                                            low, high = np.percentile(summary, (lb, ub))
                                            lows += [low]
                                            highs += [high]
                                        lows, highs = np.array(lows), np.array(highs)
                                        row_summ_ax.fill_betweenx(y, lows, highs, color=color, alpha=.3, zorder=2)
                            else:
                                y = new_ys[0]
                                # measure the radial distance travelled
                                dist = np.linalg.norm(trajectory, axis=1)
                                row_summ_ax.plot(dist.T, new_ys.T, color=color, lw=.5, alpha=.5)
                                # plot the mean distance travelled
                                mean_dist = np.nanmean(dist, axis=0)
                                row_summ_ax.plot(mean_dist, y, color='w', lw=3, zorder=3)
                                row_summ_ax.plot(mean_dist, y, color=color, lw=2, zorder=4)
                        if col_summ_ax is not None:
                            if plot_type in ['hist2d', 'line']:
                                if callable(summary_func):
                                    summary = summary_func(xs, axis=0)
                                    col_summ_ax.plot(summary, y, color=color)
                            else:
                                y = new_ys[0]
                                # measure the radial distance travelled
                                dist = np.linalg.norm(trajectory, axis=1)
                                col_summ_ax.plot(dist.T, new_ys.T, color=color, lw=.5, alpha=.5)
                                # plot the mean distance travelled
                                mean_dist = np.nanmean(dist, axis=0)
                                col_summ_ax.plot(mean_dist, y, color='w', lw=3, zorder=3)
                                col_summ_ax.plot(mean_dist, y, color=color, lw=2, zorder=4)
                        repetitions += [not_all_nans.sum()]
                elif len(xs) > 0:
                    # reshape the data
                    sample_size = len(xs)
                    # xvals, yvals = np.concatenate(xs), np.concatenate(ys)
                    # repetitions += [len(xvals)]
                    reps = 0
                    # ax.plot(xvals.T, yvals.T, color='gray', lw=.5, alpha=.5)
                    xvals = []
                    for x, y in zip(xs, ys):
                        if x.size > 0 and y.size > 0:
                            ax.plot(x, y, color='gray', lw=.5, alpha=.5)
                            reps += 1
                            xvals += [x]
                            good_y = y
                    y = good_y
                    xvals = np.array(xvals)
                    if callable(summary_func):
                        ax_mean = summary_func(xvals, axis=0)
                        # mean_vals = np.array(mean_vals)
                        # ax_mean = mean_vals.mean(0)
                        ax.plot(ax_mean, y, color='w', zorder=4, lw=4)
                        ax.plot(ax_mean, y, color=color, zorder=5)
                    # plot the mean in the two summary plots
                    if row_summ_ax is not None:
                        if callable(summary_func):
                            summary = summary_func(xvals, axis=0)
                            row_summ_ax.plot(summary, y, color=color)
                    if col_summ_ax is not None:
                        if callable(summary_func):
                            summary = summary_func(xvals, axis=0)
                            col_summ_ax.plot(summary, y, color=color)
                repetitions += [reps]
                # breakpoint()
                # add the xticks, yticks, and axis labels
        if format_axes:
            # if plotting trajectories, let's keep the an equal aspect ratio
            if plot_type == 'trajectory2d':
                for ax in trace_axes.flatten():
                    ax.set_aspect('equal')
                    ax.set_xlim(-max_radius, max_radius)
                    ax.set_ylim(-max_radius, max_radius)
            if xlabel is None:
                xlabel = xvar
            if ylabel is None:
                ylabel = yvar
            self.display.format(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel,
                                xticks=xticks, yticks=yticks, logx=logx, logy=logy)
            # add the row values
            self.display.label_margins(row_vals, row_var, col_vals, col_var)
            # add the sample size to the first subplot
            self.display.fig.suptitle(f"N={sample_size}, {min(repetitions)}–{max(repetitions)} traces per subject")
        # plt.show()

    def plot_histogram_summary(
            self, xvar, col_var, row_var, use_probability=False, bins=None,
            row_cmap=None, col_cmap=None, fig=None,
            right_margin=True, bottom_margin=True,
            xlim=None, xticks=None, logx=False, logy=False, display=None, ylim=None,
            summary_func=partial(scipy.stats.circmean, low=-np.pi, high=np.pi), xlabel=None, 
            **query_kwargs):
        """Plot histgrams of data in one big grid arranged by two variables.
        
        The color for each subplot is determined by the average of the column and 
        row colors: color = sqrt(mean([col_color^2, row_color^2])). This is the 
        proper way to average two colors and should generate a unique color for each 
        plot. 

        todo: add option to define colors in HSV space so that the user can
        align rows and columns with independent channels. For instance, rows can
        correspond to different hues while columns correspond to saturations. 

        Parameters
        ----------
        xvar : str
            The variable name to use for the histograms.
        col_var, row_var : str
            The variable names to paramaterize along columns and rows of the grid.
        row_cmap, col_cmap : func or array-like, default=None
            The colormap to use for colorizing along the rows or columns. If both are
            supplied, the product of the two colors is used for each subplot. If a 
            function is supplied, it will be applied to the corresponding col_var or row_var.
            If a list is supplied, it must have as many elements as the corresponding column or
            row.
        right_margin, bottom_margin : bool, default=True
            Whether to plot the means from each subplot in the margin to the right
            or bottom of the axes. A colormap (row_cmap or col_cmap) must be provided
            in order to distinguish individual traces.
        xlim, ylim : tuple=(min, max), default=None
            The tuple of the minimum and maximum values for that dimension.
        xticks, yticks : list, default=None
            Specify the list of ticks on the x- or y-axis.
        logx, logy : bool, default=False
            Whether to format the x- or y-axis on a log scale.
        display : SummaryDisplay, default=None
            Option to provide a SummaryDisplay with the same subplot arrangement
            to allow superimposing different datasets.
        xlabel, ylabel : str, default=None
            Option to provide custom x- and y-labels.
        summary_func : callable, default=np.nanmean
            The function for generating the summary statistic of interest in the right and bottom
            margins. It's applied to the sets of traces plotted in the trace plots.
        **query_kwargs
            These get passed to the query 
        """
        if 'subset' not in query_kwargs:
            query_kwargs['subset'] = {}
        if 'sort_by' not in query_kwargs:
            query_kwargs['sort_by'] = 'test_ind'
        subset = copy.copy(query_kwargs['subset'])
        # get the values used for coloring each subplot
        row_vals = self.query(output=row_var, sort_by=row_var, subset=query_kwargs['subset'])
        col_vals = self.query(output=col_var, sort_by=col_var, subset=query_kwargs['subset'])
        assert len(col_vals) > 0 or len(row_vals) > 0, "The subset is empty!"
        # todo: what's up with the number of axes?
        row_vals = np.unique(row_vals)
        col_vals = np.unique(col_vals)
        num_rows, num_cols = len(row_vals), len(col_vals)
        # get the colors from the specified colormaps
        colors = {}
        for cmap, vals, key in zip(
            [row_cmap, col_cmap], 
            [row_vals, col_vals],
            ['rows', 'columns']):
            if len(vals) > 0:
                if isinstance(cmap, str):
                    if isinstance(vals[0], (str, bytes)):
                        # if values are strings, sort them in alphabetical order and use their index for the colors
                        vals = np.argsort(vals)
                    # add comments below
                    # make a colormap
                    norm = matplotlib.colors.Normalize(vals.min(), vals.max())
                    cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
                    colors[key] = cmap.to_rgba(vals)[:, :-1]
                elif callable(cmap):
                    colors[key] = cmap(vals)
                elif isinstance(row_cmap, (list, np.ndarray, tuple)):
                    assert len(cmap) == len(vals), (
                        f"Colormap list has {len(cmap)} elements but {len(vals)} {key}.")
                    colors[key] = row_cmap
                else:
                    colors[key] = []
            else:
                colors[key] = []
        # combine the color lists to make an array specifying the color of each subplot
        if len(colors['rows']) == num_rows and len(colors['columns']) == num_cols:
            # get the mean combination of row and column colors
            color_mean = .5*(
                colors['rows'][:, np.newaxis]**2 +
                colors['columns'][np.newaxis, :]**2)
            color_arr = np.sqrt(color_mean)
            # elif colors['columns'] is not None:
            #     color_arr = np.repeat(colors['columns'][np.newaxis], num_rows, axis=0)
            # elif colors['rows'] is not None:
            #     color_arr = np.repeat(colors['rows'][:, np.newaxis], num_cols, axis=0)
        elif len(colors['columns']) == num_cols:
            color_arr = np.repeat(colors['columns'][np.newaxis], num_rows, axis=0)
        elif len(colors['rows']) == num_rows:
            color_arr = np.repeat(colors['rows'][:, np.newaxis], num_cols, axis=1)
        else:
            color_arr = np.zeros((num_rows, num_cols, 3), dtype='uint8')
        # test: check that the colors are aligned properly. we want this array shape to be num_rows X num_cols
        # add a row or column if plotting in the margins
        if bottom_margin:
            num_rows += 1
        if right_margin:
            num_cols += 1
        # make the figure and axes with a grid defined by num_rows and num_cols
        figsize = (1.5 * num_cols + 1, 1.5*num_rows + 1)
        format_axes = True
        if display is None:
            self.display = SummaryDisplay(
                num_rows=num_rows, num_cols=num_cols, right_margin=right_margin, 
                bottom_margin=bottom_margin, figsize=figsize)
        else:
            self.display = display
            format_axes = False
        trace_axes = self.display.trace_axes
        # plot the data
        num_frames = self.trials[0].num_frames
        repetitions = []
        if 'histograms' not in dir(self):
            self.histograms = {}
        self.histograms[xvar] = {}
        for row_num, (row, row_val, row_colors) in enumerate(zip(
            trace_axes, row_vals, color_arr)):
            row_hists = []
            if right_margin:
                row_summ_ax = self.display.right_col[row_num]
            else:
                row_summ_ax = None
            # update the subset dictionary
            subset[row_var] = row_val
            for col_num, (ax, col_val, color) in enumerate(zip(
                row, col_vals, row_colors)):
                while isinstance(ax, np.ndarray):
                    ax = ax[0]
                if bottom_margin:
                    col_summ_ax = self.display.bottom_row[col_num]
                else:
                    col_summ_ax = None
                subset[col_var] = col_val
                # todo: get the subset x- and y-values
                # update the subset dictionary
                xs = self.query(same_size=True, output=xvar, subset=subset, sort_by=query_kwargs['sort_by'])
                if isinstance(xs, np.ndarray):
                    sample_size = xs.shape[0]
                    if xs.size > 0:
                        reps = xs.shape[1]
                        # reshape the data
                        xs = xs.reshape(-1, xs.shape[-1])
                        nans = np.isnan(xs)
                        not_all_nans = nans.mean(1) < 1
                        counts, _, _ = ax.hist(xs[not_all_nans].flatten(), color=color, alpha=1, density=use_probability, bins=bins, histtype='stepfilled')
                        self.histograms[xvar][(row_val, col_val)] = counts
                        if callable(summary_func):
                            no_nans = np.isnan(xs) == False
                            ax_mean = summary_func(xs[no_nans])
                            # ax_mean = mean_vals.mean(0)
                            ax.axvline(ax_mean, color='w', zorder=4, lw=4)
                            ax.axvline(ax_mean, color='r', zorder=4)
                            # ax.plot(ax_mean, y, color=color, zorder=5)
                        # plot the mean in the two summary plots
                        for ax, vals in zip([row_summ_ax, col_summ_ax], [row_vals, col_vals]):
                            if ax is not None:
                                if callable(summary_func):
                                    no_nans = np.isnan(xs) == False
                                    summary = summary_func(xs[no_nans])
                                    alpha = 1./float(len(vals))
                                    ax.hist(xs.flatten(), color=color, zorder=1, alpha=1, bins=bins, histtype='step', density=use_probability)
                                    ax.axvline(summary, color='w', zorder=4, lw=4)
                                    ax.axvline(summary, color=color, zorder=4)
                        # if col_summ_ax is not None:
                        #     if callable(summary_func):
                        #         summary = summary_func(xs, axis=0)
                        #         col_summ_ax.plot(summary, y, color=color)
                        repetitions += [not_all_nans.sum()]
        if format_axes:
            if xlabel is None:
                xlabel = xvar
            if use_probability:
                # ylim = (0, 1)
                ylabel = "Probability"
            else:
                ylabel = "Count"
            self.display.format(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel,
                                xticks=xticks, logx=logx, logy=logy)
            # add the row values
            self.display.label_margins(row_vals, row_var, col_vals, col_var)
            # add the sample size to the first subplot
            self.display.fig.suptitle(f"N={min(repetitions)}–{max(repetitions)} traces per subplot")
        # plt.show()


class SummaryDisplay():
    def __init__(self, num_rows=1, num_cols=1, right_margin=True, bottom_margin=True,
                 **fig_kwargs):
        """Setup a figure with a grid of subplots to iteratively populate.
        
        Parameters
        ----------
        num_rows, num_cols : int, default=1
            The number of rows and columns to include.
        right_margin, bottom_margin : bool, default=True
            Whether to plot axes in the right or bottom margins.
        **fig_kwargs
            Formatting options for the matplotlib figure.
        """
        # make the figure and axes
        if num_rows == 0:
            num_rows = 1
        if num_cols == 0:
            num_cols = 1
        self.fig, self.axes = plt.subplots(num_rows, num_cols, layout='constrained',
                                           **fig_kwargs)
        if num_rows == 1 and num_cols == 1:
            self.axes = np.array([self.axes])[:, np.newaxis]
        if num_rows == 1:
            self.axes = self.axes[np.newaxis, :]
        if num_cols == 1:
            self.axes = self.axes[:, np.newaxis]
        # unless using the margins, all axes are for traces
        self.trace_axes = self.axes
        self.left = np.zeros(self.axes.shape, dtype=bool)
        self.bottom = np.copy(self.left)
        self.left[:, 0], self.bottom[-1, :] = True, True
        self.right_margin = right_margin
        # partition the right column from self.trace_axes
        if right_margin:
            if bottom_margin:
                self.right_col = self.axes[:-1, -1]
            else:
                self.right_col = self.axes[:, -1]
            self.trace_axes = self.trace_axes[:, :-1]
        # partition the bottom row from self.trace_axes
        self.bottom_margin = bottom_margin
        if bottom_margin:
            if right_margin:
                self.bottom_row = self.axes[-1, :-1]
            else:
                self.bottom_row = self.axes[-1, :]
            self.trace_axes = self.trace_axes[:-1, :]
        # if using both margins, avoid the corner margin
        if bottom_margin and right_margin:
            self.left[-1, -1] = False
            self.bottom[-1, -1] = False
            self.bottom[-2, -1] = True
            self.corner_ax = self.axes[-1, -1]
        # keep track of the left and bottom bounds for adding row and column labels
        self.left_bound = 0
        self.bottom_bound = 0

    def label_margins(self, row_vals=None, row_label=None, col_vals=None, col_label=None):
        """Add values and a label to indicate differences across rows.

        Parameters
        ----------
        row_vals : array-like (optional)
            The values to add to each row, left of the ylabel. 
        row_label : str (optional)
            The variable label applied to the row values. If the bottom 
            margin is used, the label will be centered between the trace axes. 
        col_vals : array-like (optional)
            The values to add to each column, below the xlabel. 
        col_label : str (optional)
            The variable label applied to the column values. If the right
            margin is used, the label will still be centered between the trace axes. 
        """
        # add the row values to the ylabels of the left column
        left_col = self.trace_axes[:, 0]
        for ax, val in zip(left_col, row_vals):
            lbl = ax.get_ylabel()
            if isinstance(val, (float, int)):
                ax.set_ylabel(f"{val:.1f}\n\n{lbl}")
            else:
                ax.set_ylabel(f"{val}\n\n{lbl}")
        # add the column values below the xlabels of the bottom row
        bottom_row = self.axes[-1]
        if self.right_margin:
            bottom_row = bottom_row[:-1]
        for ax, val in zip(bottom_row, col_vals):
            lbl = ax.get_xlabel()
            if isinstance(val, float):
                ax.set_xlabel(f"{lbl}\n\n{val:.1f}")
            else:
                ax.set_xlabel(f"{lbl}\n\n{val}")
        # adjust the subplots to fit the row or column label
        adjustment = .25
        fig_width, fig_height = self.fig.get_size_inches()
        if row_label is not None:
            # the adjustment should be a fixed amount. width=12 and lb=.02 => adjustment=.02*12=.24
            prop = adjustment / fig_width
            self.left_bound += prop
        if col_label is not None:
            prop = adjustment / fig_height
            self.bottom_bound += prop
        try:
            self.fig.tight_layout(
                rect=[self.left_bound, self.bottom_bound, 
                      1 - self.left_bound, 1 - self.bottom_bound])
        except:
            pass
        if row_label is not None:
            # calculate the y locations to add spines to
            yvals = []
            for ax in left_col: 
                yvals += [np.mean(np.asarray(ax.get_position())[:, 1])]
            yvals = np.array(yvals)
            # add row label to the center of the others
            label = row_label.replace("_", " ")
            txt_pos = (.02 * 12) / fig_width
            self.fig.text(txt_pos, yvals.mean(), label, va='center', ha='center', rotation='vertical')
            # plot a straight line from the bottom to the top yvals, just below the label
            if len(yvals) > 0:
                x = (.05 * 12) / fig_width
                line = matplotlib.lines.Line2D(
                    [x, x], [yvals.min(), yvals.max()], lw=1, color='k')
                self.fig.add_artist(line)
                # get tick size based on absolute length
                tick_length = (.005 * 12) / fig_width
                for yval in yvals:
                    tick = matplotlib.lines.Line2D(
                        [x, x + tick_length], [yval, yval], lw=1, color='k')
                    self.fig.add_artist(tick)
        if col_label is not None:
            # calculate the y locations to add spines to
            xvals = []
            for ax in bottom_row: xvals += [np.mean(np.asarray(ax.get_position())[:, 0])]
            xvals = np.array(xvals)
            # add row label to the center of the others
            label = col_label.replace("_", " ")
            txt_height = (.025 * 12) / fig_height
            self.fig.text(xvals.mean(), txt_height, label, va='center', ha='center', rotation='horizontal')
            # plot a straight line from the bottom to the top yvals, just below the label
            if len(xvals) > 0:
                y = (.055 * 12) / fig_height
                line = matplotlib.lines.Line2D(
                    [xvals.min(), xvals.max()], [y, y], lw=1, color='k')
                self.fig.add_artist(line)
                tick_length = (.005 * 10) / fig_height
                for xval in xvals:
                    tick = matplotlib.lines.Line2D(
                        [xval, xval], [y, y + tick_length], lw=1, color='k')
                    self.fig.add_artist(tick)

    def format(self, xlim=None, ylim=None, xticks=None, yticks=None, 
               xlabel=None, ylabel=None, special_bottom_left=False, 
               logx=False, logy=False, despine_right=True):
        """Format the subplots and figure.
        
        Parameters
        ----------
        xlim, ylim : tuple=(min, max), default=None
            The tuple of the minimum and maximum values for that dimension.
        xticks, yticks : tuple=(tickvalues, ticklabels), default=None
            The tuple specifying the tickvalues to plot on the corresponding x- or y-axis.
        xlabel, ylabel : str, default=None
            The label for the corresponding x- or y-axis.
        special_bottom_left : bool, default=False
            Whether to exclude the bottom left subplot.
        logx, logy : bool, default=False
            Whether to format the x or y-axis along a log scale.
        despine_right : bool, default=True
            Whether to remove the left spine from the right column subplots.
        """
        # account for the bottom right subplot when plotting in the right margin
        inds = np.arange(len(self.axes.flatten()))
        if self.right_margin:
            inds = inds[:-1]
        for ax in self.axes.flatten()[inds]:
            # plot the x=0 and y=0 lines 
            ax.axhline(0, linestyle='--', color='k', zorder=1, lw=.5)
            ax.axvline(0, linestyle='--', color='k', zorder=1, lw=.5)
        num_rows, num_cols = len(self.axes), len(self.axes[0])
        for row_num, (row, are_left, are_bottom) in enumerate(zip(self.axes, self.left, self.bottom)):
            for col_num, (ax, is_left, is_bottom) in enumerate(zip(row, are_left, are_bottom)):
                # set the plot limits if they were specified
                if xlim is not None:
                    ax.set_xlim(xlim[0], xlim[1])
                if ylim is not None:
                    if not (special_bottom_left and is_bottom and is_left):
                        ax.set_ylim(ylim[0], ylim[1])
                # change the x or y scale
                if logx:
                    ax.set_xscale('log')
                    if not is_bottom:
                        ax.tick_params(axis="x", which="minor", bottom=False)
                if logy:
                    ax.set_yscale('log')
                    if not is_left:
                        ax.tick_params(axis="y", which="minor", left=False)

                if (logx or logy) and not (is_left or is_bottom):
                    ax.minorticks_off()
                # if in the bottom row and xticks were provided, plot the xticks
                if is_bottom:
                    if xlabel is not None:
                        ax.set_xlabel(xlabel)
                    if xticks is not None:
                        ax.set_xticks(xticks[0], xticks[1])
                # otherwise, clean up the ticks
                else:
                    ax.set_xticks([])
                # if in the left row and yticks are provided, plot the yticks
                if is_left:
                    if not (special_bottom_left and is_bottom): 
                        if ylabel is not None:
                            ax.set_ylabel(ylabel)
                        if yticks is not None:
                            ax.set_yticks(yticks[0], yticks[1])
                # otherwise, clean up the ticks
                else:
                    ax.set_yticks([])
                # despine the axes
                # if not special_bottom_left:
                sbn.despine(ax=ax, left=is_left==False, bottom=is_bottom==False, trim=True)


class TrackingTrial():
    def __init__(self, filename, holocube_framerate=120):
        """Load an H5 TrackingTrial file from the magnocube library.

        Parameters
        ----------
        filename : path
            Path to the TrackingTrial H5 file.
        """
        self.filename = filename
        # load the h5 file
        self.file_opened = False
        self.load_success = False
        try:
            self.h5_file = h5py.File(self.filename, 'r')
            self.file_opened = True
        except:
            pass
        if self.file_opened:
            # # store the h5 datasets as attributes
            # self.holocube_framerate = holocube_framerate
            # store whether the datasets were completely loaded
            self.load_datasets()
            # grab default dataset frequently used
            if self.load_success:
                self.data = self.query()

    def add_dataset(self, name, arr):
        """Add a new dataset to the h5 file.

        Parameters
        ----------
        name : str
            The name of the dataset
        arr : np.ndarray
            The array to store.
        """
        # first re-load the dataset in readwrite mode
        self.h5_file.close()
        self.h5_file = h5py.File(self.filename, 'r+')
        # then add the dataset
        if name in self.h5_file.keys():
            del self.h5_file[name]
        self.h5_file.create_dataset(name=name, data=arr)
        del self.h5_file
        # finally, reload the file and datsets in read mode
        self.h5_file = h5py.File(self.filename, 'r')
        self.load_datasets()

    def add_attr(self, name, val):
        """Add an attribute to the whole trial dataset.

        Parameters
        ----------
        name : str
            The name of the dataset
        val : 
            The value to store.
        """
        # first re-load the dataset in readwrite mode
        if 'h5_file' in dir(self):
            self.h5_file.close()
        try:
            self.h5_file = h5py.File(self.filename, 'r+')
        except:
            breakpoint()
        # then add the dataset
        if name in self.h5_file.keys():
            del self.h5_file.attrs[name]
        self.h5_file.attrs[name] = val
        del self.h5_file
        # finally, reload the file and datsets in read mode
        self.h5_file = h5py.File(self.filename, 'r')
        self.load_datasets()

    def load_datasets(self):
        self.load_success = False
        # load datasets
        for key, val in self.h5_file.items():
            # store each dataset as an attribute
            self.__setattr__(key, np.squeeze(val[:]))
        # load attributes
        for key, val in self.h5_file.attrs.items():
            self.__setattr__(key, val)
        # trim time series to the same frame number
        min_frames = np.inf
        for attr in ['orientation', 'camera_heading', 'virtual_heading']:
            if attr in dir(self):
                min_frames = min(self.__getattribute__(attr).shape[1], min_frames)
        for attr in ['orientation', 'camera_heading', 'virtual_heading']:
            if attr in dir(self):
                self.__setattr__(attr, self.__getattribute__(attr)[:, :min_frames])
                if attr in ['orientation', 'camera_heading']:
                    # make an attribute indexing the time point of each frame
                    self.num_tests, self.num_frames = self.__getattribute__(attr).shape
        attr = 'camera_heading_offline'
        if attr in dir(self):
            self.num_tests, self.num_frames_offline = self.__getattribute__(attr).shape
            # make a count of each frame in order from start to end
            self.frame_ind_offline = np.arange(self.num_tests * self.num_frames_offline).reshape(
                self.num_tests, self.num_frames_offline)
        # measure the framerate directly
        duration = self.stop_exp - self.start_exp
        self.holocube_framerate = self.camera_heading.size / duration
        # store the approximate times
        if 'num_frames' in dir(self):
            self.test_ind = np.arange(self.num_tests)
            # make a count of each frame in order from start to end
            self.frame_ind = np.arange(self.num_tests * self.num_frames).reshape(
                self.num_tests, self.num_frames)
            # convert to time points using the framerate
            self.time = self.frame_ind * (1./ self.holocube_framerate)
            # success
            self.load_success = True
        # todo: check if the pickled bouts were saved
        bouts_fn = self.filename.replace(".h5", "_bouts.pkl")
        if os.path.exists(bouts_fn):
            self.bouts = pickle.load(open(bouts_fn, 'rb'))
        else:
            self.bouts = None

    def get_saccade_stats(self, key='camera_heading', rerun=False, **saccade_kwargs):
        """List saccades for each trial using peak angular velocities.
        
        Parameters
        ----------
        key : str, default='camera_heading'
            The variable to use for saccade data.
        rerun : bool, default=False
            Whether to re-analyze the start and stop points of saccades for each bout.
        **saccade_kwargs
        """
        # check if the bouts dataset is provided
        no_bouts = False
        if not isinstance(self.bouts, np.ndarray):
            no_bouts = True
        else:
            if len(self.bouts) != self.num_tests:
                no_bouts = True
        if rerun or no_bouts:
            # get saccades for each trial
            self.bouts = []
            times = getattr(self, 'time')
            output_vals = getattr(self, key)
            for time, test in zip(times, output_vals):
                bout = Bout(test, self.holocube_framerate)
                bout.process_saccades(**saccade_kwargs)
                bout.get_stats()
                self.bouts += [bout]
            self.bouts = np.array(self.bouts)
            # use pickle to save the list of bouts for next time 
            bout_fn = self.filename.replace(".h5", "_bouts.pkl")
            pickle.dump(self.bouts, open(bout_fn, 'wb'))
        if 'saccade_duration' not in dir(self) or rerun:
            # store relevant saccade parameters:
            lbls = ['saccade_duration', 'saccade_amplitude', 'saccade_peak_velocity',
                    'saccade_frequency', 'saccade_isi', 'saccading_left', 'saccading_right']
            variables = ['saccade_duration_avg', 'saccade_amps_avg', 'saccade_peak_velo_avg',
                        'saccade_frequency', 'inter_saccade_interval', 
                        'saccading_left', 'saccading_right']
            for lbl, var in zip(lbls, variables):
                vals = []
                for bout in self.bouts:
                    vals += [getattr(bout, var)]
                self.add_dataset(lbl, np.array(vals))

    def query_bouts(self, sort_by='test_ind', subset={'is_test': True}):
        """Return bouts that fit the subset conditions sorted by a specified variable.
    
        Parameters
        ----------
        sort_by : str, default = 'test_ind'
            The parameter to use for sorting the trials.
        subset : dict, default = {'is_test': True}
            The subset of parameters to include in the output.
        """
        # a bout corresponds to each test
        include = np.ones(self.num_tests, dtype=bool)
        if len(subset.keys()) > 0:
            for key, val in subset.items():
                logic, thresh = np.equal, val
                if isinstance(val, (str, bytes)):
                    logic, thresh = interprate_inequality(val)
                var = self.__getattribute__(key)
                if not isinstance(var, (list, tuple, np.ndarray)):
                    var = np.repeat(var, self.num_tests)
                if len(var) == len(include):
                    if isinstance(val, (list, tuple, np.ndarray)):
                        inds = np.array([test in val for test in var])
                    else:
                        inds = logic(var, thresh)
                        if inds.ndim > 1:
                            extra_dims = inds.ndim - 1
                            axes = tuple(np.arange(1, extra_dims+1))
                            # note: if any part satisfies the condition, include it
                            inds = np.any(inds, axis=axes)
                        if isinstance(val, float):
                            if np.isnan(val):
                                inds = np.isnan(var)
                    while inds.ndim > include.ndim: 
                        inds = np.any(inds, axis=-1)
                    pad = include.ndim - inds.ndim
                    index = [...]
                    index += [np.newaxis for p in range(pad)]
                    include = include * inds[tuple(index)]
                else:
                    print(f"A subset variable, {key}, has length {len(var)} but should be {len(include)}.")
        # grab the indexing variable
        sort_by = self.__getattribute__(sort_by)
        if isinstance(sort_by, (str, bytes)):
            sort_by = np.repeat(sort_by, self.num_tests)
        # select the specified subset
        # todo: fix the order array below
        assert sort_by.ndim == 1, f"the sort_by array must be flat. instead sort_by.ndim={sort_by.ndim}"
        while include.ndim > sort_by.ndim: 
            include = np.any(include, axis=-1).astype(bool)
        try:
            sort_by = sort_by[include]
        except:
            breakpoint()
        bouts = self.bouts[include]   # use only the subseted bouts
        # sort using the sort_by
        order = np.argsort(sort_by)
        return bouts[order]

    def query_saccades(self, output='heading', time_var='time', start=0, stop=np.inf, 
                       min_speed=350, max_speed=np.inf, **filter_kwargs):
        """Query the saccade data between the start and stop points.

        Parameters
        ----------
        output : str, default = 'heading'
            The output variable. Defaults to using the heading values but can also be velocity.
        time_var : str, default = 'time'
            The saccade time frame to use for selecting 
        start, stop : float, default=0., np.inf.
            The start and stop times to include in the saccade.
        min_speed, max_speed : float, default=350, np.inf
            The minimum and maximum peak saccade speed to include in the query
        **filter_kwargs
            Conditions for filtering the bouts before querying the saccades.
        """
        bouts = self.query_bouts(**filter_kwargs)
        # collect the saccades
        saccade_arr = []
        time_arr = []
        lengths_arr = []
        min_length = np.inf
        for bout in bouts:
            time, saccades = bout.query_saccades(output=output, start=start, stop=stop, min_speed=min_speed, max_speed=max_speed)
            saccade_arr += [saccades]
            time_arr += [time]
        return time_arr, saccade_arr
    
    def center_initial(self, key='camera_heading'):
        """Center the starting point for all tests.
        
        Parameters
        ----------
        key : str, default='camera_heading'
            The variable to center.
        """
        vals = self.query(key, sort_by='test_ind')
        starts = vals[..., 0]
        vals -= starts[..., np.newaxis]
        self.add_dataset(key + "_centered", vals)

    def unwrap(self, key='camera_heading', lower=-np.pi, upper=np.pi):
        """Unwrap the given time series.
        
        Parameters
        ----------
        key : str, default='camera_heading'
            The variable to unwrap.
        lower, upper : float, default=-np.pi, np.pi
            The bounds used for the unwrap function.
        """
        period = upper - lower
        vals = self.query(key, sort_by='test_ind')
        # shift up by pi so that the 2pi unwrapping is centered around 0
        vals_unwrapped = np.unwrap(vals - lower, axis=-1, period=period)
        vals_unwrapped += lower
        # comment here
        self.add_dataset(key + "_unwrapped", vals_unwrapped)

    def remove_saccades(self, key='camera_heading', invert=False, method='zero_velocity', min_speed=350, max_speed=np.inf):
        """Generate the same dataset but with saccades removed.
        
        To remove saccades, we first identify them in each bout. With this, 
        we can identify which frames are included in a saccade. We generate
        a velocity measurement, set velocity of those frames to 0, and then 
        take the cumulative sum to get the trajectory as if there were no 
        saccades. Then we add that dataset to the database.

        Parameters
        ----------
        key : str, default=camera_heading
            The variable to remove saccades from. This function will produce a new variable,
            called "{key}_no_saccades".
        invert : bool, default=False
            Whether to remove everything that is not a saccade instead.
        method : str, default='zero_velocity'
            Choose how to remove the saccades. There are two options so far:
                'zero_velocity' produces the same length array by finding velocities as the first
                    differences of heading, setting those velocities to 0, and then reverting back to 
                    position using the cumulative sum function. This has the effect of removing the 
                    saccades but maintaining the time course of responses. 
                'remove' actually removes the saccade segments, concatenates across them, and then 
                    pads the array with zeros to match the original array length.
        min_speed, max_speed : float, default=350, np.inf
            The minimum and maximum peak speeds for each saccade that's removed. Defaults to removing 
            saccades based on the Bender threshold of 350 degs/s.
        """
        if 'bouts' not in dir(self):
            # generate bout instances, which generates a list of Saccade instances
            self.get_saccade_stats(key=key)
        new_headings = []
        for bout in self.bouts:
            arr = np.unwrap(bout.arr)
            if method in ['zero_velocity', 'zero_torque']:
                # get the velocity of the bout
                offset = arr[0]
                velo = np.append([0], np.diff(arr))
            # set velo to 0 during saccades
            starts = np.array([saccade.start for saccade in bout.saccades])
            stops = np.array([saccade.stop for saccade in bout.saccades])
            peak_velos = np.array([saccade.peak_velocity for saccade in bout.saccades])
            # test: plot the heading data with the saccades highlighted
            # fig, axes = plt.subplots(nrows=2, sharex=True)
            # axes[0].plot(bout.arr)
            # ax = axes[1]
            # ax.plot(bout.velocity)
            # for start, stop in zip(starts, stops):
            #     ax.axvspan(start, stop, alpha=.2, color='gray')
            # ignore saccades with peak speed above the threshold speed
            fast_enough = abs(peak_velos) >= (min_speed * np.pi / 180.)
            slow_enough = abs(peak_velos) <= (max_speed * np.pi / 180.)
            include = fast_enough * slow_enough
            starts = starts[include]
            stops = stops[include]
            # for start, stop in zip(starts, stops):
            #     for ax in axes:
            #         ax.axvspan(start, stop, alpha=.5, color='gray')
            # plt.show()
            # breakpoint()
            # skip saccade if it starts at the beginning or end of the bout
            if invert:
                # invert the start and stop points
                new_stops = starts[1:]
                new_starts = stops[:-1]
                # replace 
                starts, stops = new_starts, new_stops
                # add the beginning to starts and end to stops
                starts = np.append([0], starts)
                stops = np.append(stops, -1)
            starts, stops = starts.astype(int), stops.astype(int)
            if len(starts) > 0:
                if method == 'zero_velocity':
                    for start, stop in zip(starts, stops):
                        # set velocity during saccade intervals to 0
                        velo[start:stop] = 0
                    # generate a new heading vector by taking the cumulative sum of the new velocity
                    new_heading  = np.cumsum(velo)
                    # new_heading += offset
                    # if new_heading[0] != offset:
                    #     breakpoint()
                    new_headings += [new_heading]
                elif method == 'zero_torque':
                    acc = np.append([0], np.diff(velo))
                    # test: plot acceleration highlighting the patches of saccades
                    # plt.plot(acc, color='k')
                    # for start, stop in zip(starts, stops): plt.axvspan(start, stop, color='gray', alpha=.3)
                    # plt.show()
                    for start, stop in zip(starts, stops):
                        # set acceleration to 0 during saccades
                        acc[start:stop] = acc[start:stop].mean()
                    # generate a new heading vector by taking the cumulative sum of the new velocity
                    new_velo = np.cumsum(acc)
                    new_heading  = np.cumsum(new_velo)
                    # new_heading += offset
                    # if new_heading[0] != offset:
                    #     breakpoint()
                    new_headings += [new_heading]
                elif method == 'remove':
                    # remove all of the 
                    # use the starts + the endpoint as the stopping points for inclusion
                    new_stops = np.append(starts, len(arr))
                    # use the begginging + the stopping points as starting points for inclusion
                    new_starts = np.append([0], stops)
                    # store values between the new starting and stopping points
                    new_vals = []
                    for start, stop in zip(new_starts, new_stops):
                        subset = arr[start:stop]
                        subset -= subset.mean()
                        new_vals += [arr[start:stop]]
                    new_vals = np.concatenate(new_vals)
                    new_heading = np.zeros(arr.shape)
                    new_heading[:len(new_vals)] = new_vals
                    new_headings += [new_heading]
            else:
                new_headings += [arr]
        new_headings = np.array(new_headings)
        if invert:
            lbl = key + ' saccades only'
        else:
            lbl = key + ' no saccades'
        lbl += f" {method}"
        self.add_dataset(lbl, new_headings)

    def query(self, output='camera_heading', sort_by='test_ind', subset={'is_test': True}):
        """Return the trials indexed by a given attribute.

        Parameters  
        ----------
        output : str, default = 'orientations'
            The parameter to output indexed by key.
        sort_by : str, default = 'time'
            The parameter to use for sorting the trials.
        subset : dict, default = {'is_test': True}
            The subset of parameters to include in the output.
        """
        ret = self.__getattribute__(output)
        if isinstance(ret, (str, bytes)):
            ret = np.repeat(ret, self.num_tests)
        include = np.ones(ret.shape, dtype=bool)
        logic_conv = {'<':np.less, '<=':np.less_equal, '==':np.equal,'>':np.greater,'>=':np.greater_equal}
        if len(subset.keys()) > 0:
            for key, val in subset.items():                
                logic, thresh = np.equal, val
                if isinstance(val, (str, bytes)):
                    logic, thresh = interprate_inequality(val)
                var = self.__getattribute__(key)
                if isinstance(var, (str, bytes)):
                    var = np.repeat(var, self.num_tests)
                # if np.any(np.isnan(val)):
                #     breakpoint()
                if len(var) == len(include):
                    if isinstance(val, (list, tuple, np.ndarray)):
                        inds = np.array([test in val for test in var])
                    # elif isinstance(val, (float, int, str, bool, bytes, np.integer, np.floating, np.str_)):
                    else:
                        inds = logic(var, thresh)
                        if isinstance(val, float):
                            if np.isnan(val):
                                inds = np.isnan(var)
                    while inds.ndim > include.ndim: 
                        inds = np.any(inds, axis=-1)
                    pad = include.ndim - inds.ndim
                    index = [...]
                    index += [np.newaxis for p in range(pad)]
                    include = include * inds[tuple(index)]
                else:
                    print(f"A subset variable, {key}, has length {len(var)} but should be {len(include)}.")
        # if self.dirname == 'Empty Sp Gal4' and 'img_id' in subset.keys():
        #     breakpoint()
        # grab the indexing variable
        sort_by = self.__getattribute__(sort_by)
        if isinstance(sort_by, (str, bytes)):
            sort_by = np.repeat(sort_by, self.num_tests)
        assert ret.size >= sort_by.size, (
            "The indexing variable cannot be larger than the output")
        # select the specified subset
        # if ret.shape != include.shape:
        #     new_ret = []
        #     for inds, arr in zip(include, ret): new_ret += [arr[inds]]
        #     ret = np.array(new_ret)
        # else:
        #     # new_ret = []
        #     # for arr, inds in zip(ret, include): 
        #     #     if isinstance(arr, np.ndarray):
        #     #         if np.any(inds): 
        #     #             new_ret += [arr[inds]]
        #     ret = ret[include]
        # index the return array using the include array
        new_ret = []
        for inds, arr, order in zip(include, ret, sort_by): 
            if isinstance(arr, np.ndarray):
                new_ret += [arr[inds]]
            elif inds:
                new_ret += [arr]
        # get inclusion index for the sort_by variable
        sort_include = np.copy(include)
        while sort_by.ndim < sort_include.ndim: sort_include = np.any(sort_include, axis=-1).astype(bool)
        sort_by = np.argsort(sort_by)[sort_include]
        # sort using the sort_by array
        new_ret_sorted = [new_ret[i] for i in sort_by]
        # if np.array(new_ret).ndim == 1 and output == 'camera_heading_offline_wrapped':
        #     breakpoint()
        # the array takes on strange shape if the indexing variable is not the same shape as the output
        ret = np.array(new_ret_sorted)
        # sort using the sort_by
        if sort_by.ndim == 1:
            order = np.argsort(sort_by)
            return ret[order]
        elif sort_by.ndim == 2:
            sort_by = sort_by.flatten()
            return ret.flatten()[np.argsort(sort_by)]

    def butterworth_filter(self, key='camera_heading', low=1, high=6,
                           sample_rate=60):
        """Apply a butterworth to the specified dataset.

        Parameters
        ----------
        key : str, default='camera_heading'
            The dataset to be filtered.
        low : float, default=1
            The lower bound of the bandpass filter in Hz.
        high : float, default=6
            The upper bound of the bandpass filter in Hz.
        sample_rate : float, default=60
            The sample rate used for calculating frequencies for the filter.
        """
        # frequencies must be at least 0
        assert low >= 0 and high > 0, "Frequencies bounds must be non-negative."
        # copy the values to filtered
        vals = np.copy(self.__getattribute__(key))
        # unwrap the vals first and then wrap again
        vals = np.unwrap(vals, axis=1)
        if low > 0 and high < np.inf:
            filter = scipy.signal.butter(5, (low, high),
                                   fs=sample_rate,
                                   btype='bandpass',
                                   output='sos')
        elif np.isinf(high):
            filter = scipy.signal.butter(5, low,
                                   fs=sample_rate,
                                   btype='highpass',
                                   output='sos')
        elif low == 0:
            filter = scipy.signal.butter(5, high,
                                   fs=sample_rate,
                                   btype='lowpass',
                                   output='sos')
        vals_smoothed = scipy.signal.sosfilt(filter, vals, axis=1)
        # apply the filter and store with a new name
        self.__setattr__(key+"_smoothed", vals_smoothed)


# todo: use nonlinear fitting to find the best jerk_std and measurement noise for a Kalman filter 
class KalmanFitter():
    def __init__(self, arr):
        self.arr = arr

    def fit(self):
        self.results = []
        # self.fmin = scipy.optimize.least_squares(self.compare, (100), bounds=(.1, np.inf))
        self.fmin = scipy.optimize.fmin(self.compare, 100)
        # todo: make a plot comparing the results with time on the x-axis
        # note: it seems like this method only really changes the noise variable. we probably have to decide on a jerk_std
        # and then minimize the noise or vice-versa
        fig, axes = plt.subplots(nrows=2)
        axes[0].scatter(range(len(self.arr)), self.arr, color='k', marker='.', s=.1)
        for num, res in enumerate(self.results): axes[0].plot(np.unwrap(res), color='k', alpha= (num+1)/len(self.results))
        axes[1].scatter(range(len(self.arr)), self.arr, color='k', marker='.', s=.1)
        axes[1].plot(np.unwrap(self.results[-1]), color='k')
        plt.tight_layout()
        plt.show()

    def error(self, modelled_vals):
        return sum((self.arr - modelled_vals)**2)

    def generate_vals(self, jerk_std=1, measurement_noise=1):
        # todo: apply KalmanAngle to the array with the given parameters
        self.filter = KalmanAngle(jerk_std=jerk_std, measurement_noise_x=measurement_noise)
        smoothed_vals = []
        for val in self.arr:
            if val != np.nan:
                self.filter.store(val)
            smoothed_vals += [self.filter.predict()]
        return np.array(smoothed_vals)

    def compare(self, jerk_std):
        # jerk_std, measurement_noise = params
        measurement_noise = 1
        self.modelled_vals = self.generate_vals(jerk_std, measurement_noise)
        self.results += [self.modelled_vals]
        return self.error(self.modelled_vals)

class Bout():
    def __init__(self, arr, framerate, original_times=None):
        """Analyze a flight bout isolating saccades and taking pertinent measurements.
        
        Parameters
        ----------
        arr : np.ndarray, shape=(num_frames)
            The array of heading values.
        framerate : float,
            The framerate of the heading time series.
        original_times : np.ndarray
            Optionally, provide the time values from the video to allow 
        """
        # store the parameters
        self.arr = arr
        self.framerate = framerate
        try:
            self.time = np.arange(len(arr))/framerate
        except:
            breakpoint()
        self.duration = self.time.max()

    def get_stats(self, **saccade_kwargs):
        """Calculate a bunch of bout statistics.

        Calculates:
        -----------
        -total angle subtended
        -average velocity
        -inter-saccade interval
        -saccade frequency
        -saccade durations
        -saccade amplitudes
        -peak velocities
        """
        # get all of the saccade data
        if 'saccades' not in dir(self):
            self.process_saccades(**saccade_kwargs)
        # total angle subtended
        self.extreme_frame = np.argmax(abs(self.arr - self.arr[0]))
        self.total_angle = self.arr[self.extreme_frame] - self.arr[0]
        # average velocity
        self.velocity = np.gradient(self.arr)
        self.velocitsy_avg = self.velocity.mean()
        self.velocity_std = self.velocity.std()
        # inter-saccade interval
        # measure the time from each stop to the next start
        starts = [saccade.stop for saccade in self.saccades][:-1]
        stops = [saccade.start for saccade in self.saccades][1:]
        intervals = []
        for start, stop in zip(starts, stops):
            intervals += [(stop - start)/self.framerate]
        self.inter_saccade_intervals = intervals
        self.inter_saccade_interval = np.mean(intervals)
        # saccade frequency
        self.saccade_frequency = len(self.saccades) / self.duration
        # saccade durations
        self.saccade_durations = [saccade.duration for saccade in self.saccades]
        self.saccade_duration_avg = np.mean(self.saccade_durations)
        self.saccade_duration_std = np.std(self.saccade_durations)
        # saccade amplitudes
        self.saccade_amps = [saccade.amplitude for saccade in self.saccades]
        self.saccade_amps_avg = np.mean(self.saccade_amps)
        self.saccade_amps_std = np.std(self.saccade_amps)
        # peak velocities
        self.saccade_peak_velo = [saccade.peak_velocity for saccade in self.saccades]
        self.saccade_peak_velo_avg = np.mean(self.saccade_peak_velo)
        self.saccade_peak_velo_std = np.std(self.saccade_peak_velo)
        # time series indicating when a 1) leftward and 2) rightward saccade is happening
        self.saccade_is_left = np.array(self.saccade_peak_velo) > 0
        self.saccading_left = np.zeros(self.arr.shape, dtype=float)
        self.saccading_right = np.zeros(self.arr.shape, dtype=float)
        for saccade, is_left in zip(self.saccades, self.saccade_is_left):
            storage = [self.saccading_left, self.saccading_right][1 * is_left]
            storage[saccade.start:saccade.stop] = True

    def process_saccades(self, threshold_speed=350, speed_noise_method=True, acceleration_method=False, 
                         maximum_saccade_frequency=3, kalman_method=False, de_lag=True, **saccade_kwargs):
        """Find saccades in an array of values. 
        
        We used a variant of the procedure from Bender and Dickinson (2006):
        0. apply a median filter (instead of the low-pass Butterworth filter).
        1. threshold the velocity gradient for speeds above 300 or 350 degs/s.
        2. find the peak velocity
        3. find the start and stop points to include velocities > .25*peak velocity
            * optionally, we can also account for baseline velocities by subtracting 
            the initial velocity
    
        Parameters
        ----------
        threshold_speed : float, default=350 degrees per second
            The minimum speed used for detecting saccades.
        speed_noise_method: bool, default=False
            Whether to use the speed distribution time series to detect saccades.
        acceleration_method : bool, default=False
            Whether to use acceleration data for better saccade extraction.
        maximum_saccade_frequency : float, default=3
            The maximum saccade frequency used for finding peak velocities pertaining 
            to saccade torque spikes.
        de_lag : bool, default=True
            If using the acceleration method, whether to use cross-correlation to 
            avoid phase errors due to smoothing.
        **saccade_kwargs
            Parameters for processing the saccades.
        """
        arr = np.copy(self.arr[np.newaxis])
        arr = np.unwrap(arr, axis=-1)
        if speed_noise_method:
            # todo: try using Kalman Filter instead
            # let's find the best jerk_std and measurement noise for fitting the heading data
            kfilter = KalmanFitter(arr[0])
            vals_filtered = np.unwrap(kfilter.generate_vals(100, .3))
            # vals_filtered = arr[0]
            # plt.scatter(range(len(arr[0])), arr[0]) 
            # plt.plot(np.unwrap(vals_filtered))
            # plt.show()
            self.saccades = []
            # use raw velocity to get the confidence interval based on a rolling window of variance
            # velocity = np.gradient(arr[0])
            velocity = np.gradient(vals_filtered)
            velocity *= self.framerate
            # find peak velocities using a peak finding algorithm
            # find the number of frames corresponding to 100 ms, because saccades are 
            # unlikely to occur that frequently
            dist = .5 * self.framerate
            peaks = scipy.signal.find_peaks(np.abs(velocity), distance=dist, width=4, prominence=(np.pi, 30), wlen=dist)
            # test:
            # fig, axes = plt.subplots(nrows=2, sharex=True)
            # axes[0].plot(arr[0])
            # plt.sca(axes[1])
            # plt.plot(velocity, zorder=2)
            # velos = velocity[peaks[0]]
            # plt.scatter(peaks[0], velos, marker='o', color=green, zorder=3)
            # # velos = velocity[neg_peaks[0]]
            # # plt.scatter(neg_peaks[0], velos, marker='o', color=red, zorder=3)
            # for lb, ub in zip(peaks[1]['left_ips'], peaks[1]['right_ips']): axes[0].axvspan(lb, ub, color='gray', alpha=.3, zorder=1); axes[1].axvspan(lb, ub, color='gray', alpha=.3, zorder=1)
            # plt.show()
            # remove outlier peaks
            starts, stops = peaks[1]['left_ips'], peaks[1]['right_ips']
            # find the rolling CI of the mean based on the past 5 velocities
            # series = pd.Series(velocity)
            # window = 5
            # rolling_mean = series.rolling(window, center=False).mean()
            # rolling_median = series.rolling(window, center=True).median()
            # rolling_std = series.rolling(window, center=False).std()
            # lower, upper = velocity - 2*rolling_std, velocity + 2*rolling_std
            # lower, upper = lower[:-1], upper[:-1]
            # upward = velocity[1:] > upper
            # downward = velocity[1:] < lower
            # test: can we simply use these cross points to get candidate saccade starts and then apply a duration filter
            # outlier_thresh = 1
            # outlier_up = (velocity - velocity.mean())/velocity.std() > outlier_thresh
            # outlier_down = (velocity - velocity.mean())/velocity.std() < -outlier_thresh
            # todo: find major peak velocities
            # peaks = scipy.signal.find_peaks(abs(velocity), prominence=.9)
            # peaks, lb, ub = peaks[0], peaks[1]['left_bases'], peaks[1]['right_bases']
            # for low, peak, high in zip(lb, peaks, ub): plt.axvline(low, color='g'); plt.axvline(high, color='red'); plt.axvline(peak, color='k'); plt.plot(velocity, color='k')
            # outlier_up = velocity > (np.pi / 180. * threshold_speed)
            # outlier_down = velocity < (np.pi / 180. * -threshold_speed)
            # go through each break in the cw and ccw boolean arrays
            # starts, stops = [], []
            # for dir_arr, direction, thresh, outliers in zip([np.where(upward)[0], np.where(downward)[0]], [np.greater, np.less], [upper, lower], [outlier_up, outlier_down]):
            #     ind = 0
            #     starts_pot = np.array(dir_arr)
            #     while len(starts_pot) > 0:
            #         start = starts_pot[0]
            #         velo = velocity[start+1:]
            #         bound = thresh[start]
            #         # the stop point is the last value within bounds
            #         in_bounds = direction(velo, bound)
            #         stops_pot = np.where((in_bounds[:-1] == True) * (in_bounds[1:] == False))[0]
            #         if len(stops_pot) > 0:
            #             stop = stops_pot[0]
            #             stop += start
            #             # check that the sequence has an outlier velocity
            #             is_fast = outliers[start:stop+1]
            #             # check if there is a peak within this range
            #             is_peak = (peaks[0] > start) * (peaks[0] < stop)
            #             if np.any(is_fast) or np.any(is_peak):
            #             # if np.any(is_peak):
            #                 # check that 
            #                 # plt.plot(velo)
            #                 # plt.axhline(bound)
            #                 # plt.axvspan(0, stop-start, color='gray', alpha=.3)
            #                 starts += [start]
            #                 stops += [stop]
            #             # remove all starts less than stop
            #             starts_pot = starts_pot[starts_pot > start]
            #         else:
            #             starts_pot = starts_pot[1:]
            # store the resulting saccade
            for start, stop in zip(starts, stops):
                saccade = Saccade(self.arr, self.framerate, start, stop, **saccade_kwargs)
                if abs(saccade.peak_velocity) < threshold_speed * np.pi / 180.:
                    saccade.success = False
                if saccade.success:
                    self.saccades += [saccade]
            # eliminate any overlapping saccades, keeping the longer one
            # saccade = self.saccades[1]
            # start, stop = saccade.start, saccade.stop
            # saccade = Saccade(self.arr, self.framerate, start, stop, display=True, **saccade_kwargs)
            # plt.show()
            new_saccades = []
            while len(self.saccades) > 0:
                saccade = self.saccades[0]
                self.saccades.remove(saccade)
                # get range of frames included in the other saccades
                spans = [range(sacc.start, sacc.stop+1) for sacc in self.saccades]
                # check if start or stop is within
                overlapping_saccades = [saccade]
                for other_saccade in self.saccades: 
                    other_span = range(other_saccade.start, other_saccade.stop+1)
                    is_contained = (saccade.start in other_span) or (saccade.stop in other_span)
                    span = range(saccade.start, saccade.stop+1)
                    contains_other = (other_saccade.start in span) or (other_saccade.stop in span)
                    same_span = (saccade.start == other_saccade.start) or (saccade.stop == other_saccade.stop)
                    if is_contained or contains_other or same_span:
                        overlapping_saccades += [other_saccade]
                for o_saccade in overlapping_saccades:
                    if o_saccade in self.saccades:
                        self.saccades.remove(o_saccade)
                if len(overlapping_saccades) > 1:
                    # we will remove all of these and replace with the longest duration saccade
                    durs = np.array([saccade.duration for saccade in overlapping_saccades])
                    ind_keep = np.argmax(durs)
                    new_saccades += [overlapping_saccades[ind_keep]]
                else:
                    new_saccades += [saccade]
            self.saccades = new_saccades
            if 'display' in saccade_kwargs.keys():
                if saccade_kwargs['display']:
                    fig, axes = plt.subplots(nrows=3, sharex=True)
                    axes[0].plot(self.time, np.gradient(velocity), color='k', marker='.')
                    axes[1].plot(self.time, velocity, color='k', marker='.')
                    axes[2].plot(self.time, arr[0])
                    # axes[1].fill_between(self.time + self.time[1], velocity - 2*rolling_std, velocity + 2*rolling_std, color='gray')
                    # axes[0].plot(self.time, rolling_median)
                    for saccade in self.saccades: axes[2].axvspan(self.time[saccade.start], self.time[saccade.stop], color='gray', alpha=.2)
                    # for start, stop in zip(np.round(starts).astype(int), np.round(stops).astype(int)): plt.axvspan(self.time[start], self.time[stop], color='gray', alpha=.2)
                    plt.show()
        elif kalman_method:
            kfilter = KalmanFitter(np.unwrap(arr[0]))
            smoothed_arr = np.unwrap(kfilter.generate_vals(100, .01))
            velocity = np.gradient(smoothed_arr)
            acceleration = np.gradient(velocity)
            # test: compare acceleration method of butterworth vs. kalman filter
            fig, axes = plt.subplots(nrows=3, sharex=True)
            plt.sca(axes[0])
            plt.plot(smoothed_arr, label='model')
            plt.scatter(range(len(arr[0])), arr[0], color='k', marker='.', label='data')
            plt.legend()
            plt.sca(axes[1])
            plt.plot(velocity, marker='.')
            plt.plot(np.gradient(arr[0]), '-k', marker='.')
            plt.sca(axes[2])
            plt.plot(acceleration, marker='.')
            plt.plot(np.gradient(np.gradient(arr[0])), '-k', marker='.')
            plt.show()
            breakpoint()
        elif acceleration_method:
            # we need a reliable measure of the acceleration data,
            # so let's apply a low-pass filter before taking gradients
            # smoothed_arr = butterworth_filter(np.unwrap(arr[0], axis=-1), 0, 6, 60)
            # smoothed_arr = smoothed_arr
            # breakpoint()
            # try using the kalman filter instead of the the butterworth
            kfilter = KalmanFitter(arr[0])
            smoothed_arr = np.unwrap(kfilter.generate_vals(300, .3))
            velocity = np.gradient(smoothed_arr)
            velocity *= self.framerate
            acceleration = np.gradient(velocity)
            acceleration *= self.framerate
            velocity_og = np.gradient(self.arr)
            # each step is idenfied as a local maximum in the velocity data
            # find peaks by setting the minimum time between saccades
            peaks = scipy.signal.find_peaks(np.abs(velocity), distance=self.framerate / maximum_saccade_frequency, prominence=.3)
            troughs = scipy.signal.find_peaks(-np.abs(acceleration), prominence=.3)[0]
            # add the in and out points of the bout to the troughs
            first_val, last_val = 0, len(smoothed_arr)
            if first_val not in troughs:
                troughs = np.append([first_val], troughs)
            if last_val not in troughs:
                troughs = np.append(troughs, [last_val])
            # for each peak, find the start and stop using the acceleration data
            self.saccades = []
            for peak in peaks[0]:
                # using the absolute value of acceleration, we now have a bimodal function in each direction
                # the start and stop are the previous and proceeding troughs in the acceleration data
                if np.any(troughs < peak) and np.any(troughs > peak):
                    start, stop = max(troughs[troughs < peak]), min(troughs[troughs > peak])
                    # todo: optionally, we can go back to the raw heading data and find the segment closest to the smoothed one here
                    # get the cross correlation of the smoothed segment with the original array
                    arr = smoothed_arr
                    if de_lag:
                        saccade = smoothed_arr[start:stop]
                        saccade_velo = velocity[start:stop]
                        # if max(abs(saccade_velo)) >= threshold_speed * np.pi / 180.:
                        saccade_velo_og = velocity_og[start:stop]
                        saccade_acceleration = np.gradient(saccade_velo)
                        saccade_acceleration_og = np.gradient(saccade_velo_og)
                        # plt.plot(acceleration)
                        # for trough in troughs: plt.axvline(trough, color='r')
                        # for peak in peaks[0]: plt.axvline(peak, color='g')
                        # plt.show() 
                        # corrs = scipy.signal.correlate(velocity_og, saccade_velo, mode='valid')
                        tmin, tmax = max(0, start - 10), min(len(arr), stop + 10)
                        offset = np.nanmean(saccade)
                        lags, corrs = normal_correlate(self.arr[tmin:tmax] - offset, saccade - offset, mode='valid', framerate=1)
                        # lags = scipy.signal.correlation_lags(len(self.arr), len(saccade), mode='valid')
                        # check lags within .5 seconds of the start
                        # included = (lags > start - .5 * self.framerate) * (lags < start + .5 * self.framerate)
                        included = (lags > 0) * (lags < 10)
                        # find the lag with the greatest 
                        if np.any(included):
                            peak_lag = round(lags[included][np.argmax(corrs[included])])
                            # peak_lag *= self.framerate
                            # peak_lag -= (stop - start)/2
                            peak_corr = corrs[peak_lag]
                            # test: check that cross-correlation will work
                            # fig, axes = plt.subplots(nrows=2, sharex=False)
                            # axes[0].plot(np.arange(peak_lag, peak_lag + len(saccade)), saccade - offset)
                            # axes[0].plot(self.arr[tmin:tmax] - offset)
                            # axes[1].scatter(lags * self.framerate, corrs, marker='.')
                            # axes[1].axhline(0, color='k', linestyle='--')
                            # plt.show()
                            # todo: fix the damned lag offset! It's not lining up correctyl!!
                            # only include the saccade if the peak correlation is substantial
                            # replace start and stop with the shifted times
                            # arr = self.arr
                            if peak_corr > .15:
                                dur = stop - start
                                start = round(tmin + peak_lag)
                                stop = round(start + dur)
                            stop = min(len(self.arr)-1, stop)
                            saccade = Saccade(self.arr, self.framerate, start, stop, display=True, **saccade_kwargs)
                            if abs(saccade.peak_velocity) < threshold_speed * np.pi / 180.:
                                saccade.success = False
                            if saccade.success:
                                self.saccades += [saccade]
            fig = plt.figure()
            plt.plot(self.arr)
            for saccade in self.saccades:
                plt.axvspan(saccade.start, saccade.stop, color='gray', alpha=.25)
            plt.show()
        else:
            # from Bender and Dickinson (2006)
            # 0. median filter
            # arr = scipy.signal.medfilt(arr, 101)
            arr = scipy.signal.medfilt(arr, 5)
            # 1. find all angular velocities > 350 degs/s
            velocity = np.gradient(arr)
            self.velocity = np.append([0], velocity)
            # convert to angular velocity
            self.velocity *= self.framerate  # convert to rads/second
            # todo: try using the smoothed velocity and acceleration 
            include = abs(self.velocity) > (threshold_speed * np.pi / 180.)  # Mongeau used 350 degs/second, but that excludes some of our slower saccades
            # plt.plot(original_arr); plt.plot(include); plt.show()     
            diffs = include.astype(int)[1:] - include.astype(int)[:-1]
            diffs = np.append([0], diffs)
            starts = np.where(diffs > 0)[0]
            stops = np.where(diffs < 0)[0]
            # collect the saccades
            self.saccades = []
            if len(starts) > 0 and len(stops) > 0:
                # if the first stop happens before the first start, remove it
                if stops[0] < starts[0]:
                    while stops[0] < starts[0]:
                        stops = stops[1:]
                # if the last start happens after the last stop, remove it
                if starts[-1] > stops[-1]:
                    while starts[-1] < stops[-1]:
                        starts = starts[:-1]
                # if any stops are `adjacent to starts, 
                while len(starts) > 0 and len(stops) > 0:
                    start = starts[0]
                    starts = starts[1:]
                    # find the next stop
                    stop = stops[0]
                    while stop < start and len(stops) > 1:
                        stops = stops[1:]
                        stop = stops[0]
                    if len(stops) > 0 and stop > start and start > 10:
                        inds = np.arange(start, stop)
                        # 2. find the ind that has the maximum velocity
                        if relative_start_velo:
                            start_velocity = self.velocity[start-10:start-5].mean()
                        else:
                            start_velocity = 0
                        relative_velocity = self.velocity - start_velocity
                        segment_velocity = relative_velocity[start:stop]
                        peak_ind = inds[np.argmax(abs(segment_velocity))]
                        peak_velocity = relative_velocity[peak_ind]
                        # 3. find the start and stop points to include velocities > .25*peak velocity
                        threshold_velocity = peak_velocity / 4
                        # test: plot the velocity and threshold. check that this behaves as expected
                        # fig, ax = plt.subplots()
                        # ax.plot(original_arr, 'ok')
                        # ax.plot(velocity)
                        # ax.axhline(threshold_velocity)
                        # plt.show()
                        # threshold depends on velocity sign
                        if peak_velocity > 0:
                            # included = self.velocity > threshold_velocity
                            included = relative_velocity > threshold_velocity
                        else:
                            # included = self.velocity < threshold_velocity
                            included = relative_velocity < threshold_velocity
                        included_inds = np.where(included)[0]
                        # find lowest index with velocity > threshold_velocity that is continuous with the peak
                        # remove any indices with diff > 1
                        diffs = np.diff(included_inds)
                        group_change = np.append([False], diffs > 1)
                        group_lbls = np.cumsum(group_change)
                        included_group = group_lbls[included_inds == peak_ind][0]
                        included_inds = included_inds[group_lbls == included_group]
                        # get the in point as the lowest ind in the included group
                        in_point, out_point = included_inds.min()-1, included_inds.max()+1, 
                        # include if the end point is not the last frame of the test
                        if out_point < 359:
                            # collect Saccades of each slice of the array
                            self.saccades += [Saccade(arr, self.framerate, start=in_point, stop=out_point, **saccade_kwargs)]


    def query_saccades(self, output='heading', time_var='time', start=0, stop=np.inf, min_speed=350,
                       max_speed=np.inf, **query_kwargs):
        """Grab saccade data between the start and stop times.
        
        Parameters
        ----------
        output : str, default='heading'
            The variable to output. Can also be velocity.
        time_var : str, default = 'time'
            The saccade time frame to use for selecting 
        start, stop : float, default=0., -1.
            The start and stop times to include in the saccade.
        sort_by : str, default = 'time'
            The parameter to use for sorting the trials.
        """
        times, saccades = [], []
        # allow for subseting and sorting just like the other query functions
        # we can only use parameters that are being collected by the Saccade object
        for saccade in self.saccades:
            include = True
            if 'subset' in query_kwargs:
                # check if the subset condition is met by this saccade
                subset = query_kwargs.pop('subset')
                # using the subset conditions, check if this saccade meets the criteria
                for key, val in subset.items():
                    # interpret val for inequalities
                    logic, thresh = interprate_inequality(val)
                    key_vals = saccade.__getattribute__(key)
                    include *= logic(key_vals, thresh)
            if include:                    
                peak_speed = abs(saccade.peak_velocity) * 180. / np.pi
                if (peak_speed > min_speed) and (peak_speed < max_speed):
                    time, saccade = saccade.query(output=output, time_var=time_var, start=start, stop=stop)
                    times += [time]
                    saccades += [saccade]
        return times, saccades

class Saccade():
    def __init__(self, arr, framerate=1, start=0, stop=-1, interpolate_velocity=True, baseline_comparison=False, display=False):
        """Wrapper for saccade time series and measurements.

        Parameters
        ----------
        arr : np.ndarray
            The heading time series from the whole trial.
        framerate : float, default=1
            The framerate of the time series.
        start, stop : int, default=0, -1
            The frames of original_arr marking the start and stop of the saccade. 
            Defaults to the whole array if unspecified.
        interpolate_velocity : bool, default=True
            Whether to interpolate the velocity value to find the velocity peak.
        baseline_comparison : bool, default=True
            Whether to use the baseline distribution of velocities to correct the
            start and stop points of the saccade.
        display : bool, default=False
            Whether to display the resultant saccade start and stop points.

        Attributes
        ----------
        time : np.ndarray
            The time from the start of the saccade.
        start_angle, stop_angle : float
            The first and last heading angles of the saccade.
        duration : float
            The duration of the time series based on the input framerate.
        velocity : np.ndarray
            The velocity time series assuming no initial motion.
        peak_velocity : float
            The maximum discrete velocity measured in the time series.
        amplitude : float
            The total displacement from start to finish.
        baseline_comparison : bool
            Whether to re-calculate the start and stop frames using the baseline 
            velocity.
        """
        # store the original time series and time
        arr = np.unwrap(arr)
        self.original_arr = arr
        self.start, self.stop = int(round(start)), int(round(stop))
        self.arr = np.asarray(np.copy(self.original_arr)[self.start:self.stop+1])
        self.framerate = framerate
        self.time = np.arange(len(self.original_arr)).astype(float)
        self.time -= self.start
        self.time /= self.framerate
        # get the start and stop times relative to the start of the bout
        self.start_time, self.stop_time = self.start / self.framerate, self.stop / self.framerate
        # calculate the starting and ending angle
        self.start_angle = self.arr[0]
        self.stop_angle = self.arr[-1]
        self.arr_relative = self.original_arr - self.start_angle
        # calculate the amplitude and duration  
        # self.amplitude = self.arr.ptp()
        self.amplitude = self.stop_angle - self.start_angle
        self.duration = (self.stop - self.start) / self.framerate
        # store the velocity time series
        self.velocity = np.gradient(self.original_arr)
        self.velocity *= self.framerate
        # store the acceleration time series
        self.acceleration = np.gradient(self.velocity)
        self.acceleration *= self.framerate
        if interpolate_velocity:
            # use a spline interpolation of the velocity to find the interpolated maximum
            start_frame, stop_frame = max(self.start - 10, 0), min(self.stop+11, len(self.time))
            # start_frame, stop_frame = max(self.start - 10, 0), min(self.stop+10, len(self.time))
            interp_func = scipy.interpolate.interp1d(self.time[start_frame: stop_frame], self.velocity[start_frame: stop_frame], kind='cubic')
            new_times = np.linspace(self.time[start_frame], self.time[stop_frame-1], 1000)
            new_velocity = interp_func(new_times)
            peak_ind = np.argmax(abs(new_velocity))
            self.peak_velocity = new_velocity[peak_ind]
            self.peak_time = new_times[peak_ind]
            self.relative_time = np.copy(self.time)
            self.relative_time -= self.peak_time
            # measure the max velocity from the few frames before the start and use as a threshold 
            # for the start of the saccade
            velos_included = self.velocity[start_frame:self.start]
            # assume the velocity is normally distributed
            if len(velos_included) > 0:
                velo_mean, velo_std = np.nanmean(velos_included), np.nanstd(velos_included)
                velo_floor, velo_ceiling = velo_mean - 2 * velo_std, velo_mean + 2 * velo_std
                # test: plot the headings, velocity, and angular acceleration highlighting the saccade interval and peak velocity
                ta, tb = max(0, self.start-10), min(len(self.velocity), self.stop + 11)
                if display:
                    fig, axes = plt.subplots(nrows=3, sharex=True)
                    axes[0].plot(self.time[ta: tb], 180./ np.pi * self.original_arr[ta:tb], 'ko-')
                    axes[1].plot(self.time[ta:tb], 180./ np.pi * self.velocity[ta: tb], 'ko-')
                    axes[1].plot(new_times, 180./ np.pi * new_velocity, 'r-')
                    axes[1].scatter(self.peak_time, 180./ np.pi * self.peak_velocity, marker='o', color='r')
                    for ax in axes: ax.axvline(self.peak_time, color='r')
                    axes[2].plot(self.time[ta: tb], 180./ np.pi * np.gradient(self.velocity[ta: tb]), 'ko-')
                    axes[0].axvspan(0, self.duration, color='gray', alpha=.25)  
                    axes[1].axvspan(0, self.duration, color='gray', alpha=.25)
                    axes[1].axhline(velo_ceiling * 180 / np.pi, color='k', linestyle='--')
                    axes[1].axhline(velo_floor * 180 / np.pi, color='k', linestyle='--')
                    axes[2].axvspan(0, self.duration, color='gray', alpha=.25)
                # todo: get the initial velocity and use this to update the start and stop points
                # the new start is the frame before the first frame with a high speed
                # the new stop is the first frame below threshold
                self.success = True
                if baseline_comparison:
                    offset = -5
                    xmin = max(0, self.start + offset)
                    xmax = min(len(self.velocity), self.stop + 10)
                    if self.peak_velocity > 0:
                        saccading = np.where(self.velocity[xmin: xmax] > velo_ceiling)[0]
                        not_saccading = np.where(self.velocity[xmin: xmax] <= velo_ceiling)[0]
                    else:
                        saccading = np.where(self.velocity[xmin: xmax] < velo_floor)[0]
                        not_saccading = np.where(self.velocity[xmin: xmax] >= velo_floor)[0]
                    not_saccading += self.start + offset
                    saccading += self.start + offset
                    if len(saccading) > 0:
                        self.start = saccading.min()
                        self.start_angle = self.original_arr[self.start]
                    # else:
                    #     self.success = False
                    if len(saccading) > 0 and np.any(not_saccading > self.start):
                        self.stop = not_saccading[not_saccading > self.start].min()
                        self.stop_angle = self.original_arr[self.stop]
                    self.duration = (self.stop - self.start) / self.framerate
                    if self.stop - self.start < 2 or self.duration > 1.5:
                        self.success = False
                    self.amplitude = self.stop_angle - self.start_angle
                if self.success and display:
                    # plot the new saccade spans
                    axes[0].axvspan(self.time[self.start], self.time[self.stop], color='gray', alpha=.5)
                    axes[1].axvspan(self.time[self.start], self.time[self.stop], color='gray', alpha=.5)
                    axes[2].axvspan(self.time[self.start], self.time[self.stop], color='gray', alpha=.5)
            else:
                self.success = False
        else:
            peak_ind = np.argmax(abs(self.velocity[self.start: self.stop]))
            self.peak_velocity = self.velocity[self.start: self.stop][peak_ind]
            self.peak_time = self.time[self.start: self.stop][peak_ind]
            self.relative_time = np.copy(self.time)
            self.relative_time -= self.peak_time

    def query(self, output='heading', time_var='time', start=0, stop=0):
        """Grab the saccade data between the start and stop times.

        In order to grab values relative to the start or stop point, which will vary between saccades,
        insert a string for the start or stop +/- the time difference. For example, to grab the values 
        between the start and .2 seconds after the start, set start='start' or 0 and stop='start+.2'. 
        Or to grab values between .1 second before and .3 seconds after the end of the saccade, set 
        start='stop-.1' and stop=.3 or stop='stop+.3'. 

        Parameters
        ----------
        output : str, default='heading'
            The output variable. Can also be velocity.
        time_var : str, default = 'time'
            The saccade time frame to use for selecting 
        start, stop : float or str, default=0.
            The start and stop times to include in the saccade relative to the start and stop of the 
            saccade, respectively. You can explicitly refer to times relative to the start or stop times

        """
        time = self.__getattribute__(time_var)
        # add feature for choosing time points relative to the start and stop of the saccade
        if isinstance(start, str):
            # default to the start
            base = 'start'
            if 'stop' in start:
                base = 'stop'
            start = start.replace(base, '')
            base = self.__getattribute__(base)
            base_val = self.time[base]
            # default to 0
            delta = 0
            if len(start) > 0:
                delta = float(start)
            start = base_val + delta
        if isinstance(stop, str):
            # default to the stop point
            base = 'stop'
            if 'start' in stop:
                base = 'start'
            stop = stop.replace(base, '')
            base = self.__getattribute__(base)
            base_val = self.time[base]
            # default to 0
            delta = 0
            if len(stop) > 0:
                delta = float(stop)
            stop = base_val + delta
        # get frames within the start and stop times
        include = (time >= start) * (time < stop)
        if output == 'heading':
            ret = self.original_arr - self.start_angle
        elif output == 'velocity':
            ret = self.velocity
        return time[include], ret[include]

def angle_rgb(angles, sat=.5, val=.7, period=2*np.pi):
    """Return a color for the given angle for a circular cmap."""
    hues = (angles % period) / period
    sats, vals = np.repeat(sat, len(hues)), np.repeat(val, len(hues))
    hsv = np.array([hues, sats, vals]).T
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    return rgb

def normal_correlate(arr1, arr2, framerate=60., mode='same',
                     circular=True):
    """Normalized cross correlation so that outcome is pearson correlation.

    Parameters
    ----------
    arr1, arr2 : np.ndarray
        The arrays to cross-correlate.
    framerate : int, default=60
        The framerate assumed for calculating lags.
    mode : str, default='same'
        A string indicating the size of the output. See the documentation
        scipy.signal.correlate for more information.
    circular : bool, default=True
        Whether to assume that the data are circular or periodic.
    """
    mean, std = np.nanmean, np.nanstd
    # if circular:
    #     mean, std = stats.circmean, stats.circstd
    arr1_std = (arr1 - mean(arr1)) / (
                std(arr1) * len(arr1))
    arr2_std = (arr2 - mean(arr2)) / (std(arr2))
    corr = scipy.signal.correlate(arr1_std, arr2_std, mode=mode)
    lags = scipy.signal.correlation_lags(len(arr1_std), len(arr2_std), mode=mode).astype('float')
    lags /= framerate
    return lags, corr

def butterworth_filter(vals, low=1, high=6, sample_rate=60):
    """Apply a butterworth to the specified dataset.

    Parameters
    ----------
    vals : np.ndarray
        The array to be filtered.
    low : float, default=1
        The lower bound of the bandpass filter in Hz.
    high : float, default=6
        The upper bound of the bandpass filter in Hz.
    sample_rate : float, default=60
        The sample rate used for calculating frequencies for the filter.
    """
    # frequencies must be at least 0
    assert low >= 0 and high > 0, "Frequency bounds must be non-negative."
    # copy the values to filtered
    vals = np.copy(vals)
    # unwrap the vals first and then wrap again
    vals = np.unwrap(vals, axis=1)
    if low > 0 and high < np.inf:
        filter = scipy.signal.butter(5, (low, high),
                                fs=sample_rate,
                                btype='bandpass',
                                output='sos')
    elif np.isinf(high):
        filter = scipy.signal.butter(5, low,
                                fs=sample_rate,
                                btype='highpass',
                                output='sos')
    elif low == 0:
        filter = scipy.signal.butter(5, high, fs=sample_rate, btype='lowpass', output='sos')
    # todo: remove initial offset and add after smoothing
    offset = vals[0, 0]
    vals_smoothed = scipy.signal.sosfilt(filter, vals - offset) + offset
    # apply the filter and store with a new name
    return vals_smoothed

def print_progress(part, whole):
    prop = float(part) / float(whole)
    sys.stdout.write('\r')
    sys.stdout.write('[%-20s] %d%%' % ('=' * int(20 * prop), 100 * prop))
    sys.stdout.flush()

def plot_diff_brackets(label, x1, x2, y1, y2, y_label, col='k',
                       vert=False, ax=None, lw=1, size='medium'):
    if ax is None:
        ax = plt.gca()
    if vert:
        ax.plot([y1, y_label, y_label, y2], [x1, x1, x2, x2],
                color=col, clip_on=False)
        ax.text(y_label, (x1+x2)*.5, label, ha='center', va='bottom',
                 color=col, rotation='vertical', size=size)
    else:
        ax.plot([x1, x1, x2, x2], [y1, y_label, y_label, y2],
                color=col, clip_on=False)
        ax.text((x1+x2)*.5, y_label, label, ha='center', va='bottom',
                 color=col, rotation='horizontal', size=size)

def moving_average(x, w): return np.convolve(x, np.ones(w), 'valid') / w

def vector_strength(arr, bins=100):
    """Convert array of angles into a time series of vector strength."""
    # convert angles into unit vectors
    xs, ys = np.cos(arr), np.sin(arr)
    # get cumulitive sum of x and y values 
    # xvals, yvals = np.cumsum(xs), np.cumsum(ys)
    # ts = np.arange(len(xvals)) + 1
    # dists = np.sqrt(xvals**2 + yvals**2)
    # normalize by the number of frames
    # test: what should we expect for a random walk???
    # random_pts = np.random.random((10000, 2, len(ts)))
    # norm = np.linalg.norm(random_pts, axis=1, keepdims=True)
    # random_pts /= norm
    # random_walks = np.cumsum(random_pts, axis=-1)
    # # measure the distance traveled, normalized by the length of the line
    # random_dists = np.sqrt(random_walks[:, 0]**2 + random_walks[:, 1]**2)
    # random_dists_normed = random_dists / ts
    # measure the rolling mean of x and y values
    xmeans, ymeans = moving_average(xs, bins), moving_average(ys, bins)
    dists = np.sqrt(xmeans**2 + ymeans**2)
    # bin the distances 
    return dists

def interprate_inequality(string):
    """Interpret a string as an inequality and return its partial function.

    This assumes that inequalities are provided as a string in this order:
        {variable} {inequality} {value}
    
    For example, "x < 5" returns partial(np.less, 5), which will only return
    True if x is less than 5. Note that spaces are ignored.
    """
    # string = string.replace(' ', '')
    # default to equality logic
    logic = np.equal
    logic_char = '=='
    # first, identify the specific logic
    logic_conv = {'<':np.less, '<=':np.less_equal, '==':np.equal,'>':np.greater,'>=':np.greater_equal}
    for key in logic_conv:
        if key in string:
            logic = logic_conv[key]
            logic_char = key
    # then, extract the variable and value
    val = string.split(logic_char)[-1]
    # convert to float if possible
    try:
        val = float(val)
    except:
        pass
    # return the partial function
    return logic, val

if __name__ == "__main__":
    tracker = OfflineTracker("..\\arena\\fourier feedback")
    # everything was shifted by pi/2, so the bounds are strange
    # let's update the camera_heading and camera_heading_offline datasets
    # experiment = TrackingExperiment("..\\arena\\fourier feedback", remove_incompletes=False)
    # for each trial, shift values to range -pi to pi
    # for trial in experiment.trials:
    #     for var in ['camera_heading', 'camera_heading_offline']:
    #         if var in dir(trial):
    #             arr = trial.__getattribute__(var)
    #             arr[arr < -np.pi] += 2 * np.pi
    #             trial.add_dataset(var, arr)
    tracker.process_vids(start_over=False)
    tracker.offline_comparison(smooth_offline=True)
