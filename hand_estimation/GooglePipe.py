import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
import os

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
NUM_HAND = 1


hand_angle = [
    [0, 1, 2],      # thumb
    [1, 2, 3],      # thumb
    [2, 3, 4],      # thumb
    [0, 5, 6],      # index
    [5, 6, 7],      # index
    [6, 7, 8],      # index
    [0, 9, 10],     # middle
    [9, 10, 11],    # middle
    [10, 11, 12],   # middle
    [0, 13, 14],    # ring
    [13, 14, 15],   # ring
    [14, 15, 16],   # ring
    [0, 17, 18],    # pinky
    [17, 18, 19],   # pinky
    [18, 19, 20]    # pinky
]

class threeD_coor:
    def __init__(self, x, y, z):

        self.x = x
        self.y = y
        self.z = z
        self.vec = np.array([x, y, z])



def Cal_Normal_3D(v1, v2, v3):
    # Calculate the vectors v2 - v1 and v3 - v1
    vec1 = v2 - v1
    vec2 = v3 - v1

    # Calculate the cross product of vec1 and vec2
    normal_vector = np.cross(vec1, vec2)

    # Normalize the normal_vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    normal_vector = threeD_coor(normal_vector[0], normal_vector[1], normal_vector[2])

    return normal_vector

def Cal_angel(coor1, coor2, coor3):
    # given three coordinates coor1, coor2, coor3
    # calculate the angle between coor1-coor2 and coor2-coor3

    v1 = coor1 - coor2
    v2 = coor3 - coor2

    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    return np.arccos(cos_theta)



def read_coor(hand_landmarks_list):


    x = []
    y = []
    z = []
    angles = []

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        # hand_landmarks have 21 points

        # print(hand_landmarks)
        # print('-------------------')

        if len(hand_landmarks) != 21:
            assert False, "The number of landmarks is not 21"
            # return None

        for landmark in hand_landmarks:
            x.append(landmark.x)
            y.append(landmark.y)
            z.append(landmark.z)


    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    xyz = np.stack((x, y, z),axis=1)

    new_x = x
    new_y = -z
    new_z = -y

    new_xyz = np.stack((new_x, new_y, new_z), axis=1)

    v1 = threeD_coor(new_x[0], new_y[0], new_z[0])
    v2 = threeD_coor(new_x[5], new_y[5], new_z[5])
    v3 = threeD_coor(new_x[17], new_y[17], new_z[17])

    vn = Cal_Normal_3D(v1.vec, v2.vec, v3.vec)
    # print(vn.vec)

    for idx in hand_angle:
        v1 = threeD_coor(new_x[idx[0]], new_y[idx[0]], new_z[idx[0]])
        v2 = threeD_coor(new_x[idx[1]], new_y[idx[1]], new_z[idx[1]])
        v3 = threeD_coor(new_x[idx[2]], new_y[idx[2]], new_z[idx[2]])

        # vn = Cal_Normal_3D(v1, v2, v3)
        # print(vn.vec)

        angle = Cal_angel(v1.vec, v2.vec, v3.vec)
        angles.append(angle)

        # print(Cal_angel(v1.vec, v2.vec, v3.vec))
    angles = np.array(angles)
    # print(angles)

    return xyz, new_xyz, angles, vn.vec

def pad_coor():

    xyz = np.zeros((21, 3))
    new_xyz = np.zeros((21, 3))
    angles = np.zeros(15)
    vn = threeD_coor(0, 0, 0)

    return xyz, new_xyz, angles, vn.vec


def extract_frames(video_path, frame_inerval, output_path=None):

    '''
    Extract frames from video file and save them as images.
    video_path: str, the path of the video file.
    frame_inerval: int, the interval of frames to extract.
    '''

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_to_skip = int(frame_inerval)

    frame_count = 0
    save_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frames_to_skip == 0:
            save_path = os.path.join(output_path, f"frame_{save_count}.jpg")
            cv2.imwrite(save_path, frame)
            save_count += 1

        frame_count += 1
    cap.release()


def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    # print(hand_landmarks_list)

    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]


    return annotated_image

def Detect_from_image_dir(Video_dirpath, Save_path=None):

    '''
    Detect hand landmarks from images in a directory.
    Video_dirpath: str, the path of the directory containing images of the video.
    '''

    if not os.path.exists(Save_path):
        os.makedirs(Save_path)

    # set four lists to save the xyz, new_xyz, angles, normal_vec
    xyz_list = []
    new_xyz_list = []
    angles_list = []
    normal_vec_list = []

    # STEP 2: Create an HandLandmarker object.
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                           num_hands=NUM_HAND)
    detector = vision.HandLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    images = os.listdir(Video_dirpath)

    images = sorted(images, key=lambda x: int(x.split('_')[1].split('.')[0]))

    # print(images)
    # image from video
    for image in images:
        # image from picture
        # print(image)

        if image.split('.')[-1] != 'jpg':
            continue

        mpImage = mp.Image.create_from_file(os.path.join(Video_dirpath, image))

        # STEP 4: Detect hand landmarks from the input image.
        detection_result = detector.detect(mpImage)

        # read coordinate



        if (len(detection_result.hand_landmarks) != 0):
            xyz, new_xzy, angels, normal_vec = read_coor(detection_result.hand_landmarks)

        else:
            xyz, new_xzy, angels, normal_vec = pad_coor()
        xyz_list.append(xyz)
        new_xyz_list.append(new_xzy)
        angles_list.append(angels)
        normal_vec_list.append(normal_vec)


        # STEP 5: Process the classification result. In this case, visualize it.
        annotated_image = draw_landmarks_on_image(mpImage.numpy_view()[:, :, :3], detection_result)


        bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        if Save_path is not None:

            cv2.imwrite(os.path.join(Save_path,f'{image}'), bgr_image)

    np.save(os.path.join(Save_path, 'xyz.npy'), np.array(xyz_list))
    np.save(os.path.join(Save_path, 'new_xyz.npy'), np.array(new_xyz_list))
    np.save(os.path.join(Save_path, 'angles.npy'), np.array(angles_list))
    np.save(os.path.join(Save_path, 'normal_vec.npy'), np.array(normal_vec_list))
    pass


def Detect_from_single_image(image_path, Save_path=None):

    image_path = image_path.replace('\\', '/')
    image_name = image_path.split('/')[-1].split('.')[0]
    # STEP 2: Create an HandLandmarker object.
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                           num_hands=NUM_HAND)
    detector = vision.HandLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    mpImage = mp.Image.create_from_file(image_path)

    # STEP 4: Detect hand landmarks from the input image.
    detection_result = detector.detect(mpImage)

    # STEP 5: Process the classification result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(mpImage.numpy_view()[:, :, :3], detection_result)

    # cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    if Save_path is not None:
        cv2.imwrite(os.path.join(Save_path, f'{image_name}.jpg'), bgr_image)

    return bgr_image


if __name__ == '__main__':


    for nation in ['nf']:

        landmark_path = f'./detect/{nation}_detect'

        video_path = f'./detect/{nation}_video'

        img_path = f'./detect/{nation}_image'

        for root, dirs, files in os.walk(video_path):
            for f in files:
                if f.split('.')[-1] == 'mp4':
                    print(f)
                    extract_frames(os.path.join(root, f), 1, os.path.join(img_path, f.split('.')[0]))
                    Detect_from_image_dir(os.path.join(img_path, f.split('.')[0]), os.path.join(landmark_path, f.split('.')[0]))
                    # arm_img_path = os.path.join(img_path, f.split('.')[0].lower())
                    # Detect_from_image_dir(os.path.join(arm_img_path, 'pose2D'), os.path.join(landmark_path, f.split('.')[0]))




