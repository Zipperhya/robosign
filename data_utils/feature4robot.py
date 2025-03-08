import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def down_sampling_from_20Hz_to_10Hz(df):
    dw_size = int(len(df) // 2)
    # print(len(df))
    dw_indices = np.random.choice(len(df), dw_size, replace=False)

    dw_df = df[df.index.isin(dw_indices)]
    # print(len(dw_df))

    return dw_df


def denoising_first_stage(x, y, z, pic_name):
    xyz = np.stack((x, y, z), axis=1)



    clustering = DBSCAN(eps=0.15, min_samples=8).fit(xyz)



    labels_ = clustering.labels_

    unique_label, counts = np.unique(labels_, return_counts=True)

    sorted_index = np.argsort(counts)[::-1]
    # counts = counts[sorted_index]
    unique_label = unique_label[sorted_index]

    if unique_label[0] == -1:

        mask = labels_ == unique_label[1]
    else:
        mask = labels_ == unique_label[0]

    return mask


def erase_start_end(csv_path, head_frame=20, tail_frame=10):

    df = pd.read_csv(csv_path)



    if '20hz' in csv_path:
        df = down_sampling_from_20Hz_to_10Hz(df)
        head_frame *= 2
        tail_frame *= 2

    sum_points = len(df)

    pic_name = get_name(csv_path)
    x = df['x']
    y = df['y']
    z = df['z']
    removal_mask = denoising_first_stage(x, y, z, pic_name)
    df = df[removal_mask]
    unique_values = sorted(df['frame'].unique())
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    df['frame_mapped'] = df['frame'].map(mapping)
    frame_max = df['frame_mapped'].max()

    df = df[~df['frame_mapped'].between(0, head_frame-1)]
    df = df[~df['frame_mapped'].between(frame_max - tail_frame + 1, frame_max)]
    df['frame_mapped'] = df['frame_mapped'] - head_frame
    # print(len(df))
    df['reframe'] = pd.cut(df['frame_mapped'], bins=1, labels=False)

    # print(df['frame_mapped'])
    sign_rate = len(df) / sum_points

    return sign_rate

def plot_points_distribution(csv_path):

    df = pd.read_csv(csv_path)

    if '20hz' in csv_path:
        df = down_sampling_from_20Hz_to_10Hz(df)

    pic_name = get_name(csv_path)
    x = df['x']
    y = df['y']
    z = df['z']
    removal_mask = denoising_first_stage(x, y, z, pic_name)
    df = df[removal_mask]

    unique_values = sorted(df['frame'].unique())
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    df['frame_mapped'] = df['frame'].map(mapping)
    frame_mapped_counts = df['frame_mapped'].value_counts().sort_index()
    frame_mapped_counts.plot(kind='bar')
    plt.xlabel('Frame Mapped')
    plt.ylabel('Count')

    plt.title(pic_name)
    plt.show()
    plt.close()

def count_points(csv_path):
    df = pd.read_csv(csv_path)

    if '20hz' in csv_path:
        df = down_sampling_from_20Hz_to_10Hz(df)

    pic_name = get_name(csv_path)
    x = df['x']
    y = df['y']
    z = df['z']
    removal_mask = denoising_first_stage(x, y, z, pic_name)
    df = df[removal_mask]

    # print(len(df))
    return len(df)

def plot_segment_stage(csv_path, seg, save_root_path=None):

    colors = ['#264A7D', '#990F1D', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
              'hotpink']

    df = pd.read_csv(csv_path)

    if '20hz' in csv_path:
        df = down_sampling_from_20Hz_to_10Hz(df)

    pic_name = get_name(csv_path)
    x = df['x']
    y = df['y']
    z = df['z']
    removal_mask = denoising_first_stage(x, y, z, pic_name)
    df = df[removal_mask]
    ax = plt.axes(projection="3d")


    unique_values = sorted(df['frame'].unique())
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    df['frame_mapped'] = df['frame'].map(mapping)
    df['reframe'] = pd.cut(df['frame_mapped'], bins=seg, labels=False)
    x = df['x']
    y = df['y']
    z = df['z']


    for i in range(seg):

        x_seg = x[df['reframe'] == i][::4]
        y_seg = y[df['reframe'] == i][::4]
        z_seg = z[df['reframe'] == i][::4]

        if i == 2 or i == 3 or i == 1:
            ax.scatter3D(x_seg, y_seg, -z_seg, color=colors[1], label = 'End', s = 300)


    if save_root_path is not None:
        relative_path = os.path.dirname(csv_path)

        relative_path = relative_path.replace('\\', '/')
        place = relative_path.split('/')[-3]
        # object_name = relative_path.split('/')[-3]
        nation = relative_path.split('/')[-2]
        word = relative_path.split('/')[-1]


        dir_path = os.path.join(save_root_path, nation, word.lower())

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        plt.savefig(os.path.join(dir_path, pic_name + '.jpg'), transparent=True)

    ax.set_axis_off()
    plt.show()
    plt.close()

def plot_one_stage(df, pic_name, ax = None):


    x = df['x']
    y = df['y']
    z = df['z']
    removal_mask = denoising_first_stage(x, y, z, pic_name)
    df = df[removal_mask]

    if ax is None:
        ax = plt.axes(projection="3d")

    ax.scatter3D(df['x'], df['y'], df['z'], color='#99a4bc')

    ax.set_xlim([-1, 1])
    ax.set_ylim([0, 2])
    ax.set_zlim([-1, 1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')




def get_name(path_name: str):

    # print(path_name)

    path_name = path_name.replace('\\', '/')
    place = path_name.split('/')[-4]
    # object_name = path_name.split('/')[-4]
    nation = path_name.split('/')[-3]
    word = path_name.split('/')[-2]
    trial = os.path.basename(path_name).split('.')[0]
    # print(trial)

    pic_name = place + '_' + nation + '_' + word + '_' + trial

    return pic_name

def plot_denoise(csv_path, save_root_path=None, ):
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

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(6, 6))

    body_xzy = plot_one_stage(df, pic_name, axs[1])
    # removal_xyz = plot_second_stage(x, y, z, pic_name, axs[0, 1])


    axs[0].scatter3D(x, y, z, color='#99a4bc')
    axs[0].set_xlim([-1, 1])
    axs[0].set_ylim([0, 2])
    axs[0].set_zlim([-1, 1])
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].set_zlabel('Z')

    fig.suptitle(pic_name)
    plt.tight_layout()
    # plt.show()

    if save_root_path is not None:
        relative_path = os.path.dirname(csv_path)

        relative_path = relative_path.replace('\\', '/')
        place = relative_path.split('/')[-3]
        # object_name = relative_path.split('/')[-3]
        nation = relative_path.split('/')[-2]
        word = relative_path.split('/')[-1]


        dir_path = os.path.join(save_root_path, nation, word.lower())

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # plt.savefig(os.path.join(dir_path, pic_name + '.jpg'))
    plt.show()
    plt.close(fig)
    pass

def get_specific_words(nation):

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
    # print(root_path)

    for dirpath, dirnames, filenames in os.walk(root_path):

        if len(dirnames) == 0:

            relative_path = os.path.relpath(dirpath, root_path)
            # print(dirpath)


            relative_path = relative_path.replace('\\', '/')
            place = relative_path.split('/')[-3]
            # object_name = relative_path.split('/')[-3]
            nation = relative_path.split('/')[-2]
            word = relative_path.split('/')[-1]

            if nation == nation_index and word.lower() == word_index:
                # print(dirpath)
                dir_list.append(dirpath)

    return dir_list

def plot_word(dir_list):

    points_number_sum = 0
    file_number = 0

    for dir in dir_list:

        for file in os.listdir(dir):

            if file.endswith('.csv'):
                csv_path = os.path.join(dir, file)
                # print(csv_path)
                points_number_sum += erase_start_end(csv_path)
                # plot_points_distribution(csv_path)
                # points_number_sum += count_points(csv_path)
                file_number += 1

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
        avg_number = plot_word(dir_list)
        print(f'{nation} {word} {avg_number}')
        # break
