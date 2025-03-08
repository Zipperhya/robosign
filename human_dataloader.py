import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn.cluster import DBSCAN
import os
from torch.utils.data.dataset import Dataset
import json
from sklearn.model_selection import train_test_split

class denoising:
    def __init__(self, eps1=0.1, min_samples1=15, eps2=0.2, min_samples2=8):
        ''' DBSCAN(eps1, min_samples1) is used to remove the body points, DBSCAN(eps2, min_samples2) is used to remove the noise points
        eps1 < eps2, min_samples1 > min_samples2
        inputs:
            eps1: The maximum distance between two samples for one to be considered as in the neighborhood of the other
            min_samples1: The number of samples in a neighborhood for a point to be considered as a core point
            eps2: The maximum distance between two samples for one to be considered as in the neighborhood of the other
            min_samples2: The number of samples in a neighborhood for a point to be considered as a core point
        '''
        self.eps1 = eps1
        self.min_samples1 = min_samples1
        self.eps2 = eps2
        self.min_samples2 = min_samples2

    def denoising_first_stage(self, x, y, z):
        '''
        inputs:
            x: numpy array of shape (N, )
            y: numpy array of shape (N, )
            z: numpy array of shape (N, )
        outputs:
            numpy array of shape (N, ), which contains the mask of the body points
        '''
        xyz = np.stack((x, y, z), axis=1)
        clustering = DBSCAN(eps=self.eps1, min_samples=self.min_samples1).fit(xyz)
        labels_ = clustering.labels_
        unique_label, counts = np.unique(labels_, return_counts=True)
        sorted_index = np.argsort(counts)[::-1]
        unique_label = unique_label[sorted_index]
        if unique_label[0] == -1:

            if len(unique_label) == 1:
                # all the points are discrete points and no body part
                mask = np.zeros_like(labels_, dtype=bool)

            else:
                mask = labels_ == unique_label[1]
        else:
            mask = labels_ == unique_label[0]
        return mask

    def denoising_second_stage(self, x, y, z):
        '''
        inputs:
            x: numpy array of shape (N, )
            y: numpy array of shape (N, )
            z: numpy array of shape (N, )
        outputs:
            numpy array of shape (N, ), which contains the mask of the noise points
        '''
        xyz = np.stack((x, y, z), axis=1)
        clustering = DBSCAN(eps=self.eps2, min_samples=self.min_samples2).fit(xyz)
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
        body_mask = self.denoising_first_stage(x, y, z)
        removal_mask = self.denoising_second_stage(x, y, z)
        # mask = np.logical_and(removal_mask, np.logical_not(body_mask))
        return removal_mask

        # only remove the noise points
        # return removal_mask



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


    def __call__(self, data, if_argu):
        data = self.normalize_data(data)
        if if_argu:
            # print(data)
            data = self.scale_data(data)
            # print(data)
            data = self.shift_data(data)
            # print(data.shape)
            data = self.jitter_data(data)
            # print(data)
            data = self.rotate_point_cloud_z(data)
            # print(data)
            pass
        return data


class data_read:
    def __init__(self, N, F, if_denoise = True, if_filter = True):
        '''
        inputs:
            N: the number of points in each frame after resampling
            F: the number of frames after dividing
            if_denoise: bool, whether to denoise the data
            if_filter: bool, whether to give up the data if the number of points is less than threshold
        '''
        self.N = N
        self.F = F
        self.resampling = resampling(N)
        self.denoising = denoising()
        self.argument = argument4one_sample()
        # self.resampling = resampling(N)
        self.if_denoise = if_denoise
        self.if_filter = if_filter

    def read_from_path(self, path):
        '''
        inputs:
            path: csv file path
        outputs:
            numpy array of shape (F, N, 3), which contains the resampled points
        '''

        # grouping
        df = pd.read_csv(path)

        # denoising
        x = df['x']
        y = df['y']
        z = df['z']
        #
        if self.if_denoise:
            removal_mask = self.denoising(x, y, z)
            # print(len(df))
            df = df[removal_mask]
        # print(len(df))
        if self.if_filter:
            if len(df) < int(self.F * self.N // 2):
                return None

        unique_values = sorted(df['frame'].unique())
        mapping = {value: idx for idx, value in enumerate(unique_values)}
        df['frame_mapped'] = df['frame'].map(mapping)
        df['reframe'] = pd.cut(df['frame_mapped'], bins=self.F, labels=False)

        # argument
        x = df['x']
        y = df['y']
        z = df['z']

        if 'rent' in path:
            # Attention: the pose of radar from the rentroom is different from other places
            x = -x
            z = -z

        points = np.stack((x, y, z), axis=1)

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


class RadarDataset_offline(Dataset):
    def __init__(self, data_path, N = 20, F = 5,  if_argu = False):
        super(RadarDataset_offline, self).__init__()
        with open(data_path, 'r') as f:
            self.samples = json.load(f)
        # self.X = data_X
        # self.Y = data_Y
        self.argument = argument4one_sample()
        self.if_argu = if_argu
        self.N = N
        self.F = F

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = np.array(sample['data'], dtype=np.float32)


        data = data.reshape(self.F * self.N, 3)
        data = self.argument(data, self.if_argu)
        data = data.reshape(self.F, self.N, 3)

        # data = sample['data']
        label = sample['label']

        return data, label

def read_data_offine(dir_path, N, F, target_nation):
    '''
    Read the data from the directory
    inputs:
        dir_path: the directory path
        N: the number of points in each frame after resampling
        F: the number of frames after dividing
    outputs:
        a list of dictionaries, each dictionary contains the place, object_name, nation, word, label, and data
    '''
    file_paths = get_all_file_paths(dir_path)
    data_Re = data_read(N, F, if_denoise=True, if_filter=False)
    samples_X = []
    samples_Y = []
    samples = []
    for file_path in file_paths:
        sample = {}

        file_path = file_path.replace('\\', '/')
        # print(file_path.split('/'))

        place = file_path.split('/')[-5]
        object_name = file_path.split('/')[-4]
        nation = file_path.split('/')[-3]
        word = file_path.split('/')[-2]

        # print(file_path)

        label_index = get_specific_label(nation, word)
        if nation != target_nation or label_index == -1:
            continue

        sample_points = data_Re.read_from_path(file_path)
        # print(sample_points.shape)
        if sample_points is None:
            continue

        sample['place'] = place
        sample['object_name'] = object_name
        sample['nation'] = nation
        sample['word'] = word

        # sample['label'] = get_label(nation, word)
        sample['label'] = label_index
        sample['data'] = sample_points.tolist()

        samples_X.append(sample['data'])
        samples_Y.append(sample['label'])
        samples.append(sample)

    samples_X = np.array(samples_X)
    samples_Y = np.array(samples_Y)

    with open(f'samples_{target_nation}.json', 'w') as f:
        json.dump(samples, f)

    return samples_X, samples_Y, samples

def read_washdata_offine(dir_path, N, F, target_nation, selected_obj=None):
    '''
    Read the data from the directory
    inputs:
        dir_path: the directory path
        N: the number of points in each frame after resampling
        F: the number of frames after dividing
    outputs:
        a list of dictionaries, each dictionary contains the place, object_name, nation, word, label, and data
    '''
    file_paths = get_all_file_paths(dir_path)
    data_Re = data_read(N, F, if_denoise=False, if_filter=False)
    samples_X = []
    samples_Y = []
    samples = []
    for file_path in file_paths:
        sample = {}


        file_path = file_path.replace('\\', '/')


        nation = file_path.split('/')[-3]
        word = file_path.split('/')[-2]
        file_name = file_path.split('/')[-1]
        place = file_name.split('_')[0]
        object_name = file_name.split('_')[1]

        # print(file_path)

        label_index = get_specific_label(nation, word)
        if nation != target_nation or label_index == -1:
            continue

        if selected_obj is not None:
            if object_name not in selected_obj:
                continue

        sample_points = data_Re.read_from_path(file_path)
        # print(sample_points.shape)
        if sample_points is None:
            continue

        sample['place'] = place
        sample['object_name'] = object_name
        sample['nation'] = nation
        sample['word'] = word

        # sample['label'] = get_label(nation, word)
        sample['label'] = label_index
        sample['data'] = sample_points.tolist()

        samples_X.append(sample['data'])
        samples_Y.append(sample['label'])
        samples.append(sample)

    samples_X = np.array(samples_X)
    samples_Y = np.array(samples_Y)

    with open(f'samples_{target_nation}.json', 'w') as f:
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





