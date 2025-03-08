import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn.cluster import DBSCAN
import os
from torch.utils.data.dataset import Dataset
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def down_sampling_from_20Hz_to_10Hz(df):
    dw_size = int(len(df) // 2)
    # print(len(df))
    dw_indices = np.random.choice(len(df), dw_size, replace=False)

    dw_df = df[df.index.isin(dw_indices)]
    # print(len(dw_df))

    return dw_df

class denoising:
    def __init__(self, eps=0.15, min_samples=8):
        '''
        The class is used to remove the noise points of robot data
        inputs:
            eps: float, The maximum distance between two samples for one to be considered as in the neighborhood of the other
            min_samples: int, The number of samples in a neighborhood for a point to be considered as a core point
        '''
        self.eps = eps
        self.min_samples = min_samples

    def denoising_noise(self, x, y, z):
        '''
        inputs:
            x: numpy array of shape (N, )
            y: numpy array of shape (N, )
            z: numpy array of shape (N, )
        outputs:
            numpy array of shape (N, ), which contains the mask of the noise points
        '''
        xyz = np.stack((x, y, z), axis=1)
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(xyz)
        labels_ = clustering.labels_
        unique_label, counts = np.unique(labels_, return_counts=True)
        sorted_index = np.argsort(counts)[::-1]
        unique_label = unique_label[sorted_index]
        if unique_label[0] == -1:
            mask = labels_ == unique_label[1]
        else:
            mask = labels_ == unique_label[0]
        return mask

    def __call__(self, x, y, z):
        '''
        inputs:
            x: numpy array of shape (N, )
            y: numpy array of shape (N, )
            z: numpy array of shape (N, )
        outputs:
            numpy array of shape (N, ), which contains the mask of the points without noise and body
        '''
        removal_mask = self.denoising_noise(x, y, z)
        # only remove the noise points
        return removal_mask



class resampling:
    def  __init__(self, desired_num_points):
        self.desired_num_points = desired_num_points


    def down_sampling4one_frame(self, points):
        '''
        Down-sampling the points using K-means algorithm
        inputs:
            points: numpy array of shape (N, d), where N is the number of points and d is the dimension of each point
            n: the desired number of points after down-sampling
        outputs:
            numpy array of shape (n, d), which contains the down-sampled points
        '''
        # print(points.shape)
        kmeans = KMeans(n_clusters=self.desired_num_points, random_state=0).fit(points)
        return kmeans.cluster_centers_

    def up_sampling4one_frame(self, points):
        '''
        Up-sampling the points using Hierarchical Agglomerative Clustering algorithm
        inputs:
            points: numpy array of shape (N, d), where N is the number of points and d is the dimension of each point
            n: the desired number of points after up-sampling
        outputs:
            numpy array of shape (n, d), which contains the up-sampled points
        # '''
        # print(points.shape)

        # try:
        while len(points) < self.desired_num_points:
            # Apply AHC
            if self.desired_num_points - len(points) < len(points) // 2:
                clustering = AgglomerativeClustering(n_clusters=self.desired_num_points - len(points))

            else:
                clustering = AgglomerativeClustering(n_clusters=len(points) // 2)
            labels = clustering.fit_predict(points)

            # Compute centroids
            centroids = np.array([points[labels == i].mean(axis=0) for i in range(clustering.n_clusters)])

            # Add centroids to points
            points = np.vstack([points, centroids])

        # if the number of points is larger than the desired number of points, we need to down-sample the points
        if len(points) > self.desired_num_points:
            points = self.down_sampling4one_frame(points)

        return points
        # except ValueError:
        #     return np.zeros((self.desired_num_points, 3))

    def resampling(self, points):
        '''
        Resampling the points using K-means and Hierarchical Agglomerative Clustering algorithms
        inputs:
            points: numpy array of shape (N, d), where N is the number of points and d is the dimension of each point
            n: the desired number of points after resampling
        outputs:
            numpy array of shape (n, d), which contains the resampled points
        '''
        # print(points.shape)
        if len(points) > self.desired_num_points:
            return self.down_sampling4one_frame(points)
        else:
            return self.up_sampling4one_frame(points)

    def __call__(self, points):
        return self.resampling(points)

class argument4one_sample:
    def __init__(self):
        pass

    def normalize_data(self, data):
        """ Normalize the data, use coordinates of the block centered at origin,
            Input:
                Nx3 array
            Output:
                Nx3 array
        """
        centroid = np.mean(data, axis=0, keepdims=True)
        pc = data - centroid
        return pc
        # m = np.max(np.sqrt(np.sum(pc ** 2, axis=1, keepdims=True)))
        # normal_data = pc / m
        # return normal_data

    def scale_data(self, data, scale_low=0.8, scale_high=1.25):
        """ Scale the data, use the same scale for each batch
            Input:
                Nx3 array
            Output:
                Nx3 array
        """
        scales = np.random.uniform(scale_low, scale_high)
        data *= scales
        return data

    def shift_data(self, data, shift_range=0.2):
        """ Shift the data, use the same shift for each batch
            Input:
                Nx3 array
            Output:
                Nx3 array
        """
        shifts = np.random.uniform(-shift_range, shift_range, (1, 3))
        # print(shifts.shape)
        data += shifts
        return data

    def jitter_data(self, data, sigma=0.01, clip=0.05):
        """ Jitter the data, add noise to the data
            Input:
                Nx3 array
            Output:
                Nx3 array
        """
        jittered_data = np.clip(sigma * np.random.randn(*data.shape), -1 * clip, clip)
        jittered_data += data
        return jittered_data

    def rotate_point_cloud_z(self, data):
        """ Randomly rotate the point clouds to augument the dataset
                rotation is per shape based along up direction
                Input:
                  Nx3 array, original point cloud
                Return:
                  Nx3 array, rotated point cloud
            """
        rotation_angle = np.random.uniform() * np.pi * 0.5
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        zeros = np.zeros(1)[0]
        ones = np.ones(1)[0]
        # print(cosval, sinval, zeros, ones)
        rotation_matrix = np.array([[cosval, -sinval, zeros],
                                    [sinval, cosval, zeros],
                                    [zeros, zeros, ones]])
        # print(rotation_matrix)
        rotated_data = np.dot(data, rotation_matrix)
        return rotated_data


    def __call__(self, data):
        data = self.normalize_data(data)
        # print(data)
        # data = self.scale_data(data)
        # print(data)
        # data = self.shift_data(data)
        # print(data.shape)
        # data = self.jitter_data(data)
        # print(data)
        # data = self.rotate_point_cloud_z(data)
        # print(data)
        return data


def voxalize(x_points, y_points, z_points, x, y, z, velocity):

    '''
    Function to voxalize the data
    input:
        x_points: number of points in x direction, it defines the resolution of the voxel in x direction
        y_points: number of points in y direction, it defines the resolution of the voxel in y direction
        z_points: number of points in z direction, it defines the resolution of the voxel in z direction
        x: x coordinate of the points
        y: y coordinate of the points
        z: z coordinate of the points
    output:
        pixel: voxelized data
    '''

    x_min = np.min(x)
    x_max = np.max(x)

    y_min = np.min(y)
    y_max = np.max(y)

    z_max = np.max(z)
    z_min = np.min(z)

    z_res = (z_max - z_min)/z_points
    y_res = (y_max - y_min)/y_points
    x_res = (x_max - x_min)/x_points

    pixel = np.zeros([x_points,y_points,z_points])


    for i in range(y.shape[0]):
        x_current = x_min
        x_prev = x_min
        x_count = 0
        done=False

        while x_current <= x_max and x_count < x_points and done==False:
            y_prev = y_min
            y_current = y_min
            y_count = 0
            while y_current <= y_max and y_count < y_points and done==False:
                z_prev = z_min
                z_current = z_min
                z_count = 0
                while z_current <= z_max and z_count < z_points and done==False:
                    if x[i] < x_current and y[i] < y_current and z[i] < z_current and x[i] >= x_prev and y[i] >= y_prev and z[i] >= z_prev:
                        pixel[x_count,y_count,z_count] = pixel[x_count,y_count,z_count] + 1
                        done = True

                        #velocity_voxel[x_count,y_count,z_count] = velocity_voxel[x_count,y_count,z_count] + velocity[i]
                    z_prev = z_current
                    z_current = z_current + z_res
                    z_count = z_count + 1
                y_prev = y_current
                y_current = y_current + y_res
                y_count = y_count + 1
            x_prev = x_current
            x_current = x_current + x_res
            x_count = x_count + 1
    return pixel




class data_read:
    def __init__(self, N, F):
        '''
        inputs:
            N: the number of points in each frame after resampling
            F: the number of frames after dividing
        '''
        self.N = N
        self.F = F
        self.resampling = resampling(N)
        self.denoising = denoising()
        self.argument = argument4one_sample()
        # self.resampling = resampling(N)
        self.head_frame = 20
        self.tail_frame = 10

    def read_from_path(self, path):
        '''
        inputs:
            path: csv file path
        outputs:
            numpy array of shape (F, N, 3), which contains the resampled points
        '''

        # print(path)
        head_frame = self.head_frame
        tail_frame = self.tail_frame

        # grouping
        df = pd.read_csv(path)

        if '20hz' in path:
            df = down_sampling_from_20Hz_to_10Hz(df)
            head_frame *= 2
            tail_frame *= 2

        # print(df['reframe'])


        # denoising
        x = df['x']
        y = df['y']
        z = df['z']
        #
        removal_mask = self.denoising(x, y, z)
        # print(len(df))
        df = df[removal_mask]
        # print(len(df))



        unique_values = sorted(df['frame'].unique())
        mapping = {value: idx for idx, value in enumerate(unique_values)}
        df['frame_mapped'] = df['frame'].map(mapping)

        # erase the head and tail frames
        frame_max = df['frame_mapped'].max()


        df = df[~df['frame_mapped'].between(0, head_frame - 1)]

        df = df[~df['frame_mapped'].between(frame_max - tail_frame + 1, frame_max)]

        if len(df) < int(25):
            return None


        df['frame_mapped'] = df['frame_mapped'] - head_frame
        # print(df['frame_mapped'])
        # print(len(df))

        df['reframe'] = pd.cut(df['frame_mapped'], bins=self.F, labels=False)

        # argument
        x = df['x']
        y = df['y']
        z = df['z']
        points = np.stack((x, y, z), axis=1)
        # print(points.shape)
        # points = self.resampling(points)
        # points = self.argument(points)

        # a array to store the resampled points
        resampled_points = []

        # resampling
        for i in range(self.F):

            frame_points = points[df['reframe'] == i]

            frame_points = self.resampling(frame_points)
            # print(points.shape)
            resampled_points.append(frame_points)
        resampled_points = np.array(resampled_points)
        return resampled_points

def get_all_file_paths(directory):
    '''
    Get all the file paths in the directory
    '''
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:

            file_paths.append(os.path.join(root, file))
    return file_paths

def get_specific_label(nation, word):

    # the seven selected words for each nation
    FSL = ['fish', 'one', 'promise', 'know',
           'ten', 'tuesday',
           'yes']

    GSL = ['always',
           'library', 'lie', 'same', 'things',
           'us', 'your']

    MSL = ['badperson', 'february', 'hotel', 'lie', 'monday',
           'you', 'head']

    SSL = ['child', 'good', 'goodbye',
            'sugar',
           'thankyou', 'water', 'yellow']

    if nation == 'flb':
        if word.lower() in FSL:
            return FSL.index(word.lower())
        else:
            return -1
    elif nation == 'jn':
        if word.lower() in GSL:
            return GSL.index(word.lower())
        else:
            return -1
    elif nation == 'mxg':
        if word.lower() in MSL:
            return MSL.index(word.lower())
        else:
            return -1
    elif nation == 'nf':
        if word.lower() in SSL:
            return SSL.index(word.lower())
        else:
            return -1
    else:
        return -1




class RadarDataset(Dataset):
    def __init__(self, directory, N = 20, F = 5):
        super(RadarDataset, self).__init__()
        self.file_paths = get_all_file_paths(directory)
        # print(self.file_paths)

        self.data_Re = data_read(N = N, F = F)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        file_path = self.file_paths[idx]
        sample = {}
        # the file path structure is 'DIRECTORY/nation/setting/word/1.csv'
        # get the place, person_name, nation, word from the file path

        file_path = file_path.replace('\\', '/')
        # print(file_path.split('/'))

        nation = file_path.split('/')[-4]
        setting = file_path.split('/')[-3]
        word = file_path.split('/')[-2]

        sample['setting'] = setting
        sample['nation'] = nation
        sample['word'] = word

        # print(nation)

        sample['label'] = get_specific_label(nation, word)


        data = self.data_Re.read_from_path(file_path)
        sample['data'] = data
        return sample

class RadarDataset_offline(Dataset):
    def __init__(self, data_path, N = 20, F = 5,  if_argu = True):
        super(RadarDataset_offline, self).__init__()
        with open(data_path, 'r') as f:
            self.samples = json.load(f)
        # self.X = data_X
        # self.Y = data_Y
        self.N = N
        self.F = F
        self.if_argu = if_argu
        self.argument = argument4one_sample()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = np.array(sample['data'], dtype=np.float32)
        # data = sample['data']

        if self.if_argu:
            data = data.reshape(self.F * self.N, 3)
            data = self.argument(data)
            data = data.reshape(self.F, self.N, 3)

        label = sample['label']

        return data, label

def read_data_offine(dir_path, N, F, target_nation):
    '''
    Read the data from the directory
    inputs:
        dir_path: the directory path
        N: the number of points in each frame after resampling
        F: the number of frames after dividing
        target_nation: the target nation
    outputs:
        a list of dictionaries, each dictionary contains the place, object_name, nation, word, label, and data
    '''
    file_paths = get_all_file_paths(dir_path)
    data_Re = data_read(N, F)
    samples_X = []
    samples_Y = []
    samples = []
    for file_path in file_paths:
        sample = {}

        file_path = file_path.replace('\\', '/')
        # print(file_path.split('/'))

        setting = file_path.split('/')[-4]
        nation = file_path.split('/')[-3]
        word = file_path.split('/')[-2]

        label_index = get_specific_label(nation, word)

        # if target_nation is not None and nation != target_nation:
        #     continue

        if nation != target_nation or label_index == -1:
            continue

        sample_points = data_Re.read_from_path(file_path)
        if sample_points is None:
            continue

        sample['setting'] = setting
        sample['nation'] = nation
        sample['word'] = word


        sample['label'] = label_index
        sample['data'] = sample_points.tolist()
        samples_X.append(sample['data'])
        samples_Y.append(sample['label'])
        samples.append(sample)

    samples_X = np.array(samples_X)
    samples_Y = np.array(samples_Y)

    with open(f'samples_robot_{target_nation}_7class_{N}points_tailed_new.json', 'w') as f:
        json.dump(samples, f)

    return samples_X, samples_Y, samples

def read_data_offline_voxalize(dir_path, N, F, vox_x = 10, vox_y = 32, vox_z = 32):
    file_paths = get_all_file_paths(dir_path)
    data_Re = data_read(N, F)
    samples_X = []
    samples_Y = []
    samples = []
    for file_path in file_paths:
        sample = {}

        file_path = file_path.replace('\\', '/')
        # print(file_path.split('/'))

        nation = file_path.split('/')[-4]
        setting = file_path.split('/')[-3]
        word = file_path.split('/')[-2]

        sample['setting'] = setting
        sample['nation'] = nation
        sample['word'] = word

        sample['label'] = get_specific_label(nation, word)

        data = data_Re.read_from_path(file_path)

        vox_data = []

        for frame in data:
            x = frame[:, 0]
            y = frame[:, 1]
            z = frame[:, 2]
            velocity = np.zeros_like(x)
            pixel = voxalize(vox_x, vox_y, vox_z, x, y, z, velocity)
            vox_data.append(pixel)
            # sample['data'] = pixel.tolist()
            # samples_X.append(sample['data'])
            # samples_Y.append(sample['label'])
            # samples.append(sample)
        vox_data = np.array(vox_data)


        sample['data'] = vox_data.tolist()
        samples_X.append(sample['data'])
        samples_Y.append(sample['label'])
        samples.append(sample)

    with open('samples_vox.json', 'w') as f:
        json.dump(samples, f)

    return samples_X, samples_Y, samples


def load_data_offline(data_path):
    with open(data_path, 'r') as f:
        samples = json.load(f)

    train_size = int(0.8 * len(samples))
    test_size = len(samples) - train_size
    train_samples, test_samples = train_test_split(samples, [train_size, test_size])
    train_dataset = RadarDataset_offline(train_samples)
    test_dataset = RadarDataset_offline(test_samples)
    return train_dataset, test_dataset




