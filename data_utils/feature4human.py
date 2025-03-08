import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def denoising_first_stage(x, y, z):
    xyz = np.stack((x, y, z), axis=1)
    clustering = DBSCAN(eps=0.1, min_samples=15).fit(xyz)
    labels_ = clustering.labels_
    unique_label, counts = np.unique(labels_, return_counts=True)
    sorted_index = np.argsort(counts)[::-1]
    unique_label = unique_label[sorted_index]
    if unique_label[0] == -1:

        if len(unique_label) == 1:
            mask = np.zeros_like(labels_, dtype=bool)

        else:
            mask = labels_ == unique_label[1]
    else:
        mask = labels_ == unique_label[0]
    return mask

def denoising_second_stage(x, y, z):
    xyz = np.stack((x, y, z), axis=1)
    clustering = DBSCAN(eps=0.25, min_samples=6).fit(xyz)
    labels_ = clustering.labels_
    unique_label, counts = np.unique(labels_, return_counts=True)
    sorted_index = np.argsort(counts)[::-1]
    unique_label = unique_label[sorted_index]
    if unique_label[0] == -1:
        mask = labels_ == unique_label[1]
    else:
        mask = labels_ == unique_label[0]
    return mask

def plot_first_stage(df, pic_name, ax=None):

    x = df['x']
    y = df['y']
    z = df['z']

    # pic_name = get_name(csv_path)
    if ax is None:
        ax = plt.axes(projection="3d")


    # print(labels_)

    colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
              'hotpink']

    body_mask = denoising_first_stage(x, y, z)
    df = df[body_mask]
    ax.scatter3D(df['x'], df['y'], df['z'], color='#99a4bc')



    ax.set_xlim([-1, 1])
    ax.set_ylim([0, 2])
    ax.set_zlim([-1, 1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')



def plot_second_stage(df, pic_name, ax = None):

    x = df['x']
    y = df['y']
    z = df['z']
    removal_mask = denoising_second_stage(x, y, z)
    df = df[removal_mask]

    xyz = np.stack((x, y, z), axis=1)

    colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
              'hotpink']


    if ax is None:
        ax = plt.axes(projection="3d")



    ax.scatter3D(df['x'], df['y'], df['z'], color='green')

    ax.set_xlim([-1, 1])
    ax.set_ylim([0, 2])
    ax.set_zlim([-1, 1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def plot_segment_stage(csv_path, seg):

    '''
    plot the points in different segments with different colors, the segment order is based on the frame order
    Through the plot, we can see the different point distribution in continuous frames
    '''


    colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
              'hotpink']

    df = pd.read_csv(csv_path)
    pic_name = get_name(csv_path)
    x = df['x']
    y = df['y']
    z = df['z']

    body_mask = denoising_first_stage(x, y, z)
    removal_mask = denoising_second_stage(x, y, z)
    mask = np.logical_and(removal_mask, np.logical_not(body_mask))

    df = df[removal_mask]
    ax = plt.axes(projection="3d")
    ax.set_title(pic_name)

    unique_values = sorted(df['frame'].unique())
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    df['frame_mapped'] = df['frame'].map(mapping)
    df['reframe'] = pd.cut(df['frame_mapped'], bins=seg, labels=False)
    x = df['x']
    y = df['y']
    z = df['z']
    # points = np.stack((x, y, z), axis=1)

    for i in range(seg):
        # seg_df = df['reframe'] == i
        x_seg = x[df['reframe'] == i]
        y_seg = y[df['reframe'] == i]
        z_seg = z[df['reframe'] == i]

        if 'rent' in csv_path:
            ax.scatter3D(-x_seg, y_seg, z_seg, color=colors[i % len(colors)], label=f'{i}')
        else:
            ax.scatter3D(x_seg, y_seg, -z_seg, color=colors[i % len(colors)], label=f'{i}')

    ax.legend()
    plt.show()
    plt.close()

def count_points(csv_path):

    '''
    count the number of points after the second stage denoising
    '''

    df = pd.read_csv(csv_path)
    x = df['x']
    y = df['y']
    z = df['z']
    body_mask = denoising_first_stage(x, y, z)
    removal_mask = denoising_second_stage(x, y, z)
    mask = np.logical_and(removal_mask, np.logical_not(body_mask))
    df = df[mask]
    return len(df)

def plot_three_stage(csv_path):

    '''
    The whole pre-process for human point cloud data
    1. Denoising the noise points, it will remove the outlier points
    2. Detect the dense body part and it will be used in the next stage
    3. Decide the sign part with boundary and extract the sign part
    '''

    colors = ['#264A7D', '#990F1D', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', '#d35400', '#9b59b6', '#95a5a6',
              '#6c7a89']

    df = pd.read_csv(csv_path)
    pic_name = get_name(csv_path)
    x = df['x']
    y = df['y']
    z = df['z']

    body_mask = denoising_first_stage(x, y, z)
    removal_mask = denoising_second_stage(x, y, z)

    noise_points = df[np.logical_not(removal_mask)]

    human_points = df[removal_mask]

    body_points = df[body_mask]
    body_xyz = np.stack((body_points['x'], body_points['y'], body_points['z']), axis=1)


    body_close_y = np.min(body_xyz[:, 1])
    # body_low_z = np.min(body_xyz[:, 2])
    sign_points = human_points[human_points['y'] < body_close_y - 0.1]
    trunk_points = human_points[(human_points['y'] >= body_close_y - 0.1) & (~human_points.isin(body_points).all(axis=1))]


    # plot the noise_points, body_points, sign_points, trunk_points with different colors in one pic
    ax = plt.axes(projection="3d")
    ax.set_title(pic_name)
    # ax.scatter3D(noise_points['x'], noise_points['y'], -noise_points['z'], color=colors[3], label='noise', alpha=0.2)
    ax.scatter3D(sign_points['x'], sign_points['y'], -sign_points['z'], color=colors[1], label='sign', s = 300)
    ax.scatter3D(trunk_points['x'], trunk_points['y'], -trunk_points['z'], color=colors[0], label='trunk', s= 300)
    ax.scatter3D(body_points['x'], body_points['y'], -body_points['z'], color=colors[0], label='body', s=300)

    ax.set_axis_off()
    # plt.show()
    plt.close()

    sign_rate = len(sign_points) / len(human_points)

    return sign_rate

    pass


def get_name(path_name: str):


    path_name = path_name.replace('\\', '/')
    place = path_name.split('/')[-5]
    object_name = path_name.split('/')[-4]
    nation = path_name.split('/')[-3]
    word = path_name.split('/')[-2]
    trial = os.path.basename(path_name).split('.')[0]
    # print(trial)

    pic_name = place + '_' + object_name + '_' + nation + '_' + word + '_' + trial

    # print(label)
    return pic_name

def plot_denoise(csv_path, save_root_path=None, prefix_path = 'G:\手语视频新\雷达数据csv'):

    '''
    plot the original point cloud, the first stage denoising result and the second stage denoising result
    but this function may not be used in the future
    '''

    df = pd.read_csv(csv_path)
    # print(csv_path)
    pic_name = get_name(csv_path)


    x = df['x']
    y = df['y']
    z = df['z']


    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # pic_name = get_name(csv_path)

    fig, axs = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(6, 6))

    body_xzy = plot_first_stage(df, pic_name, axs[0, 0])
    removal_xyz = plot_second_stage(df, pic_name, axs[0, 1])


    axs[1, 0].scatter3D(x, y, z, color='#99a4bc')
    axs[1, 0].set_xlim([-1, 1])
    axs[1, 0].set_ylim([0, 2])
    axs[1, 0].set_zlim([-1, 1])
    axs[1, 0].set_xlabel('X')
    axs[1, 0].set_ylabel('Y')
    axs[1, 0].set_zlabel('Z')


    # body_xzy = plot_first_stage(x, y, z, pic_name)

    # removal_xyz = plot_second_stage(x, y, z, pic_name)

    mask = np.isin(removal_xyz, body_xzy).all(axis=1)
    body_and_move = removal_xyz[~mask]

    axs[1, 1].scatter3D(body_and_move[:, 0], body_and_move[:, 1], body_and_move[:, 2], color='green')
    axs[1, 1].set_xlim([-1, 1])
    axs[1, 1].set_ylim([0, 2])
    axs[1, 1].set_zlim([-1, 1])
    axs[1, 1].set_xlabel('X')
    axs[1, 1].set_ylabel('Y')
    axs[1, 1].set_zlabel('Z')
    # axs[1, 1].set_title(pic_name)
    fig.suptitle(pic_name)
    plt.tight_layout()
    # plt.show()

    if save_root_path is not None:
        relative_path = os.path.dirname(csv_path)

        relative_path = relative_path.replace('\\', '/')
        place = relative_path.split('/')[-4]
        object_name = relative_path.split('/')[-3]
        nation = relative_path.split('/')[-2]
        word = relative_path.split('/')[-1]


        dir_path = os.path.join(save_root_path, nation, word.lower())

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        plt.savefig(os.path.join(dir_path, pic_name + '.jpg'))

    plt.close(fig)

    pass

def get_specific_words(nation):
    # print(nation)

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
        return FSL
    elif nation == 'jn':
        return GSL
    elif nation == 'mxg':
        return MSL
    elif nation == 'nf':
        return SSL
    else:
        # give an error that the nation not exists
        raise ValueError('The nation not exists')

def read_root_dir(root_path, nation_index, word_index):

    dir_list = []

    for dirpath, dirnames, filenames in os.walk(root_path):

        if len(dirnames) == 0:

            relative_path = os.path.relpath(dirpath, root_path)


            relative_path = relative_path.replace('\\', '/')
            place = relative_path.split('/')[-4]
            object_name = relative_path.split('/')[-3]
            nation = relative_path.split('/')[-2]
            word = relative_path.split('/')[-1]

            if nation == nation_index and word.lower() == word_index:
                # print(dirpath)
                dir_list.append(dirpath)
    # print(nation_index, word_index)
    # print(len(dir_list))
    return dir_list

def plot_word(dir_list, save_path):

    points_number_sum = 0
    file_number = 0

    for dir in dir_list:

        for file in os.listdir(dir):

            if file.endswith('.csv'):
                csv_path = os.path.join(dir, file)
                plot_three_stage(csv_path)
                # exit(1)
                # print(len(df))

            else:
                continue
        # break
    return points_number_sum / file_number

def plot_nation(nation, dir_path):
    # word_list = get_words(nation)

    word_list = get_specific_words(nation)
    for word in word_list:
        dir_list = read_root_dir(dir_path, nation, word)
        # plot_word(dir_list)
        avg_number = plot_word(dir_list, None)
        print(f'{nation} {word} {avg_number}')
        # break
