import glob
import math
import os
import shutil
import time
import socket
import struct
import traceback
import logging
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageShow
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from collections import OrderedDict
import darknet
import selectors
from os import listdir
from os.path import isfile, join, isdir
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D
from keras.models import Sequential, load_model
from keras.layers import LayerNormalization

print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print('device: ', device)
# device = torch.device('cpu')
# print('device: ', device)
# data setting
# class FishDataset(Dataset):
#     def __init__(self, folder_path, transform=None):
#         self.folder_path = glob.glob(folder_path + '\*.png')  #抓取路徑下所有檔案
#         self.transform = transform  #dataloader需要transform
#
#     def __len__(self):
#         return len(self.folder_path)
#
#     def __getitem__(self, idx):
#         file_path = self.folder_path[idx]
#         img = Image.open(file_path)
#         # label = file_path.split('\\')[-2]   #取最後一個\前的資料夾名稱
#         if self.transform:
#             img = self.transform(img)
#         return img
# class sharpen(object):
#     def __call__(self, img):
#         img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
#         sigma = 75
#         blur = cv2.GaussianBlur(img, (0, 0), sigma)
#         usm = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
#         img = cv2.cvtColor(usm, cv2.COLOR_BGR2RGB)
#         return Image.fromarray(img)
#     def __repr__(self):
#         return self.__class__.__name__+'()'
# imgtransform = transforms.Compose([
#     sharpen(),
#     transforms.CenterCrop((970, 1700)),
#     transforms.ToTensor(),
#     transforms.ConvertImageDtype(torch.float),
# ])
#
# # model
# class Autoencoder(nn.Module):  # 65-32-8-32-65
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.encoder_block = nn.Sequential(OrderedDict([
#             ('fc1', nn.Linear(1700, 1024)),
#             ('Relu1', nn.ReLU()),
#             ('fc2', nn.Linear(1024, 512)),
#             ('Relu2', nn.ReLU()),
#             ('fc3', nn.Linear(512, 256)),
#             ('Relu3', nn.ReLU())
#         ]))
#         self.decoder_block = nn.Sequential(OrderedDict([
#             ('fc4', nn.Linear(256, 512)),
#             ('Relu4', nn.ReLU()),
#             ('fc5', nn.Linear(512, 1024)),
#             ('Relu5', nn.ReLU()),
#             ('fc6', nn.Linear(1024, 1700)),
#             ('Sigmoid', nn.Sigmoid())
#         ]))
#
#     def forward(self, x):
#         z = self.encoder_block(x)
#         z = self.decoder_block(z)
#         return z  # in [0.0, 1.0]

# YOLO
def image_detection(image, network, class_names, class_colors, thresh):
    # width = darknet.network_width(network)
    # height = darknet.network_height(network)
    if type(image) == torch.Tensor:
        image = transforms.ToPILImage()(image)
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    # image = cv2.imread(image_path)
    # print(image.shape)
    # print(type(image))
    (height, width, rgb) = image.shape
    darknet_image = darknet.make_image(width, height, 3)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections
# 指定網路結構配置檔 yolov3.cfg，自建數據集設定檔 obj.data，事先訓練好的權重檔 yolov3.backup
network, class_names, class_colors = darknet.load_network(
    "./cfg/yolov4-fish.cfg",
    "./data/fish.data",
    "./backup/yolov4-fish_last.weights",
    1
)

# # multiple img testing
# def multiple_img(model, ds):
#     err_list = []
#     n_features = len(ds[0][0][0])  # 1700
#     for i in range(len(ds)):
#         X = ds[i]
#         prev_time = time.time()  # 用來計算辨識一張圖片的時間
#         # print('\npredicting...', prev_time)
#         # 進行影像辨識，回傳畫出方塊框的圖片以及辨識結果，辨識結果包含標籤、置信度以及方塊框的座標
#         image, detections = image_detection(
#             X, network, class_names, class_colors, 0.25
#         )
#         cv2.imwrite(test_path + 'yolo-{}'.format(i) + '.jpg', image)
#         # 顯示辨識時間
#         print('yolo predicted time: ', (time.time() - prev_time))
#         # 印出標籤、置信度以及方塊框的座標
#         # darknet.print_detections(detections, '--ext_output')
#         pos = []
#         for label, confidence, bbox in detections:
#             pos.append(bbox)
#         # print(pos)
#         # print(len(pos))
#
#         prev_time = time.time()
#         with torch.no_grad():
#             X = X.to(device)
#             Y = model(X)  # should be same as X
#         kernel_matrix = (X - Y) * (X - Y)
#
#         if pos:
#             localmax = []
#             for j in range(len(pos)):
#                 x, y, w, h = pos[j]
#                 # print('pos[{}]: {}'.format(j, pos[j]))
#                 subkernel = kernel_matrix[0:, int(y):int(y + h), int(x):int(x + w)]
#                 # print(subkernel.shape)
#                 localmax.append(torch.max(subkernel).item())
#                 # print('localmax[{}]: {}'.format(j, localmax[j]))
#             globalmax = np.argmax(localmax)
#             # print(globalmax)
#             # print(pos[globalmax])
#
#         err = torch.sum(kernel_matrix).item()  # SSE all features
#         err = err / n_features  # sort of norm'ed SSE
#         print('err: {}'.format(err))
#         if err < 143.65:
#             err_list.append((i, err, pos[globalmax]))  # idx of data item, err
#         print('test img time: ', (time.time() - prev_time))
#
#     err_list.sort(key=lambda x: x[1], reverse=True)
#     # print(len(err_list))
#     if err_list:
#         print("Largest reconstruction item / error: ")
#         (idx, err, mep) = err_list[0]
#         print(" [%5d]  %0.5f" % (idx, err))
#         # display_digit(data_ds, idx)
#         # print(len(err_list))
#         for i in range(3 if len(err_list)>3 else len(err_list)):
#             (idxQ, errQ, mepQ) = err_list[i]
#             print(i, (idxQ, errQ, mepQ))
#             a = test_dataset[idxQ][0]
#             # print(type(a))
#             a = transforms.ToPILImage()(a)
#             left, top, right, bottom = darknet.bbox2points(mepQ)
#             draw = ImageDraw.ImageDraw(a)
#             draw.rectangle(((left, top), (right, bottom)), fill=None, outline='red')
#             a.show()
#             center = (int((left + right) / 2), int((top + bottom) / 2))
#             return (idxQ, center)
#     else:
#         print('no error!')
# # single img testing
# def single_img(model, img_path):
#     img = Image.open(img_path)
#     img = imgtransform(img)
#     n_features = img.size(2)  # 1700
#     error_flag = 0
#     prev_time = time.time()  # 用來計算辨識一張圖片的時間
#     # print('\npredicting...', prev_time)
#     # 進行影像辨識，回傳畫出方塊框的圖片以及辨識結果，辨識結果包含標籤、置信度以及方塊框的座標
#     image, detections = image_detection(
#         img, network, class_names, class_colors, 0.25
#     )
#     cv2.imwrite(img_path + '-yolo-result.jpg', image)
#     # 顯示辨識時間
#     print('yolo predicted time: ', (time.time() - prev_time))
#     # 印出標籤、置信度以及方塊框的座標
#     # darknet.print_detections(detections, '--ext_output')
#     pos = []
#     for label, confidence, bbox in detections:
#         pos.append(bbox)
#
#     prev_time = time.time()
#     with torch.no_grad():
#         img = img.to(device)
#         out = model(img)  # should be same as X
#     kernel_matrix = (img - out) * (img - out)
#
#     if pos:
#         localmax = []
#         for j in range(len(pos)):
#             x, y, w, h = pos[j]
#             # print('pos[{}]: {}'.format(j, pos[j]))
#             subkernel = kernel_matrix[0:, int(y):int(y + h), int(x):int(x + w)]
#             # print(subkernel.shape)
#             localmax.append(torch.max(subkernel).item())
#             # print('localmax[{}]: {}'.format(j, localmax[j]))
#         globalmax = np.argmax(localmax)
#         # print(globalmax)
#         # print(pos[globalmax])
#     print('test img time: ', (time.time() - prev_time))
#
#     err = torch.sum(kernel_matrix).item()  # SSE all features
#     err = err / n_features  # sort of norm'ed SSE
#     if err < 143.65:
#         print('err: {}'.format(err))
#         error_flag = 1
#         print('*** error! ***')
#     else:
#         print('~~~ no error ~~~')
#
#     if error_flag:
#         print("Largest reconstruction error: %0.5f".format(err))
#         img = transforms.ToPILImage()(img)
#         left, top, right, bottom = darknet.bbox2points(pos[globalmax])
#         draw = ImageDraw.ImageDraw(img)
#         draw.rectangle(((left, top), (right, bottom)), fill=None, outline='red')
#         img.show()
#         center = (int((left+right)/2),int((top+bottom)/2))
#         return center
#
# # img_path = r"D:\so sad\_fish\train\normal1\1.png"
# # img = Image.open(img_path)
# # img = imgtransform(img)
# # image, detections = image_detection(
# #         img, network, class_names, class_colors, 0.25
# #     )
# # cv2.imwrite(img_path + '-yolo-result.jpg', image)
# # *attention* need to be refresh to the best model
# FILE = r"D:\so sad\_fish\result\model\model_state_dict.pt"
# autoenc = Autoencoder().to(device)
# # model = torch.load(FILE)
# # autoenc.load_state_dict(model['model_state_dict'])
# autoenc.load_state_dict(torch.load(FILE))
# # avg_std.load_state_dict(model['avg_std_error'])
# autoenc.eval()
#
# test_path = r'D:/so sad/_fish/test/'
# test_path = r'C:\Users\USER\OneDrive\文件\GitHub\projectShare2\unity\aquarium_URP\Assets\StreamingAssets'
# test_dataset = FishDataset(test_path, imgtransform)
# # test_loader = torch.utils.data.DataLoader(test_dataset,
# #                                     batch_size=5, shuffle=False)
# # multiple_img(autoenc, test_dataset)
# #
# #
# def sending_and_reciveing():
#     s = socket.socket()
#     socket.setdefaulttimeout(None)
#     print('socket created ')
#     port = 11000
#     s.bind(('127.0.0.1', port)) #local host
#     s.listen(30) #listening for connection for 30 sec?
#     print('socket listensing ... ')
#     while True:
#         try:
#             c, addr = s.accept() #when port connected
#             bytes_received = c.recv(4000) #received bytes
#             array_received = np.frombuffer(bytes_received, dtype=np.float32) #converting into float array
#
#             # nn_output = return_prediction(array_received) #NN prediction (e.g. model.predict())
#             output = multiple_img(autoenc, test_dataset)
#
#             bytes_to_send = struct.pack('%sf' % len(output), *output) #converting float to byte
#             c.sendall(bytes_to_send) #sending back
#             c.close()
#         except Exception as e:
#             logging.error(traceback.format_exc())
#             print("error")
#             c.sendall(bytearray([]))
#             c.close()
#             break
#
# sending_and_reciveing()
# #
# #
# #
# # HOST = '127.0.0.1' # 標準的迴環地址 (localhost)
# # PORT = 11000 # 監聽的埠 (非系統級的埠: 大於 1023)
# # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# # s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# # s.bind((HOST, PORT))
# # s.listen(5)
# # print('server start at: %s:%s' % (HOST, PORT))
# # print('wait for connection...')
# # while True:
# #     conn, addr = s.accept()
# #     with conn:
# #         print('\nConnected by', addr)
# #         while True:
# #             data = conn.recv(1024)
# #             if not data:
# #                 break
# #             print('recv: ' + data.decode())
# #             testing = single_img(autoenc, test_path + 'normal/110.png')
# #             if testing is None:
# #                 conn.send('normal')
# #             else:
# #                 conn.send('error: '+ testing)
#
# # img = Image.open(r"C:\Users\USER\OneDrive\文件\GitHub\projectShare2\unity\aquarium_URP\Assets\StreamingAssets\3.png")
# # img = imgtransform(img)
# # img.show()



# physical_devices = tf.config.experimental.list_physical_devices('CPU')
# for gpu in physical_devices:
#     tf.config.experimental.set_memory_growth(gpu, True)
#
# print("Num GPUs:", len(physical_devices))
#
# TF_GPU_ALLOCATOR = 'cuda_malloc_async'

def get_clips_by_stride(stride, frames_list, sequence_size):
    """ For data augmenting purposes.
    Parameters
    ----------
    stride : int
        The distance between two consecutive frames
    frames_list : list
        A list of sorted frames of shape 256 X 256
    sequence_size: int
        The size of the lstm sequence
    Returns
    -------
    list
        A list of clips , 10 frames each
    """
    clips = []
    sz = len(frames_list)
    # print(sz)
    clip = np.zeros(shape=(sequence_size, 256, 256, 1))
    cnt = 0
    for start in range(0, stride):
        for i in range(start, sz, stride):
            clip[cnt, :, :, 0] = frames_list[i]
            cnt = cnt + 1
            if cnt == sequence_size:
                clips.append(np.copy(clip))
                cnt = 0
    return clips

def get_training_set():
    """
    Returns
    -------
    list
        A list of training sequences of shape (NUMBER_OF_SEQUENCES,SINGLE_SEQUENCE_SIZE,FRAME_WIDTH,FRAME_HEIGHT,1)
    """
    clips = []
    # loop over the training folders (Train000,Train001,..)
    for f in sorted(listdir(Config.DATASET_PATH)):
        directory_path = join(Config.DATASET_PATH, f)
        if isdir(directory_path):
            all_frames = []
            # loop over all the images in the folder (0.tif,1.tif,..,199.tif)
            for c in sorted(listdir(directory_path)):
                # print(c)
                img_path = join(directory_path, c)
                if str(img_path)[-3:] == "png":
                    img = Image.open(img_path).resize((256, 256))
                    gray = ImageOps.grayscale(img)
                    gray = np.array(gray, dtype=np.float32) / 256.0
                    all_frames.append(gray)
            # get the 10-frames sequences from the list of images after applying data augmentation
            for stride in range(1, 3):
                clips.extend(get_clips_by_stride(stride=stride, frames_list=all_frames, sequence_size=10))
    return clips

def get_model(reload_model=True):
    """
    Parameters
    ----------
    reload_model : bool
        Load saved model or retrain it
    """
    if not reload_model:
        print("load model")
        return load_model(Config.MODEL_PATH,custom_objects={'LayerNormalization': LayerNormalization})
    training_set = get_training_set()
    print(len(training_set))
    training_set = np.array(training_set)
    seq = Sequential()
    seq.add(TimeDistributed(Conv2D(128, (11, 11), strides=4, padding="same"), batch_input_shape=(None, 10, 256, 256, 1)))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2D(64, (5, 5), strides=2, padding="same")))
    seq.add(LayerNormalization())
    # # # # #
    seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    seq.add(ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    # # # # #
    seq.add(TimeDistributed(Conv2DTranspose(64, (5, 5), strides=2, padding="same")))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2DTranspose(128, (11, 11), strides=4, padding="same")))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2D(1, (11, 11), activation="sigmoid", padding="same")))
    print(seq.summary())

    seq.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, decay=1e-5, epsilon=1e-6))
    print("finish compiling")
    seq.fit(training_set, training_set, batch_size=Config.BATCH_SIZE, epochs=Config.EPOCHS, shuffle=False)

    # history = seq.fit(training_set, training_set, batch_size=Config.BATCH_SIZE, epochs=Config.EPOCHS, shuffle=False)

    seq.save(Config.MODEL_PATH)
    return seq

def get_single_test():
    sz = 120
    test = np.zeros(shape=(sz, 256, 256, 1))
    image = np.zeros(shape=(sz, 1080, 1920, 3))
    cnt = 0
    for f in sorted(listdir(Config.SINGLE_TEST_PATH)):
        if f[:3]=='120':
            break
        if str(join(Config.SINGLE_TEST_PATH, f))[-3:] == "png":
            img = Image.open(join(Config.SINGLE_TEST_PATH, f))
            image[cnt, :, :, :] = img
            gray = ImageOps.grayscale(img).resize((256, 256))
            gray = np.array(gray, dtype=np.float32) / 256.0
            test[cnt, :, :, 0] = gray
            cnt = cnt + 1
    return test, image

def evaluate():
    print("evaluating...")
    model = get_model(False)
    print("got model\n", model)
    test, origin = get_single_test()
    print("got test\n", test.shape)
    sz = test.shape[0] - 10
    sequences = np.zeros((sz, 10, 256, 256, 1))
    # apply the sliding window technique to get the sequences
    for i in range(0, sz):
        clip = np.zeros((10, 256, 256, 1))
        for j in range(0, 10):
            clip[j] = test[i + j, :, :, :]
        sequences[i] = clip
    print("got data")
    # get the reconstruction cost of all the sequences
    reconstructed_sequences = model.predict(sequences,batch_size=4)
    error_map = np.subtract(sequences,reconstructed_sequences)
    sequences_reconstruction_cost = np.array([np.linalg.norm(np.subtract(sequences[i],reconstructed_sequences[i])) for i in range(0,sz)])
    sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(sequences_reconstruction_cost)
    print("sa: ", type(sa), sa.shape)
    sr = 1.0 - sa
    # plot the regularity scores
    plt.plot(sr)
    plt.ylabel('regularity score Sr(t)')
    plt.xlabel('frame t')
    # plt.ylim(0, 1)
    plt.show()
    if min(sr) < 0.9:
        minframe = np.where(sr==np.min(sr))
        print(type(minframe), minframe)
        print(minframe[0])
        error_map_plot = np.zeros(shape=(256, 256, 1))
        error_map_plot[:, :, :] = error_map[minframe[0][0] + 5, 5, :, :, :]
        plt.imshow(error_map_plot, interpolation='nearest')
        plt.show()
        error_frame = np.zeros(shape=(1080, 1920, 3))
        error_frame[:, :, :] = origin[minframe[0][0]+5, :, :, :]
        plt.imshow(error_map_plot, interpolation='nearest')
        plt.show()
        image, detections = image_detection(error_frame, network, class_names, class_colors, 0.25)   #YOLO
        cv2.imwrite(RESULT_PATH + '\\yolo-{}'.format(minframe) + '.jpg', image)
        pos = []
        for label, confidence, bbox in detections:
            pos.append(bbox)
        if pos:
            localmax = []
            for j in range(len(pos)):
                x, y, w, h = pos[j]
                # print('pos[{}]: {}'.format(j, pos[j]))
                subkernel = kernel_matrix[0:, int(y):int(y + h), int(x):int(x + w)]
                # print(subkernel.shape)
                localmax.append(torch.max(subkernel).item())
                # print('localmax[{}]: {}'.format(j, localmax[j]))
            globalmax = np.argmax(localmax)
            # print(globalmax)
            # print(pos[globalmax])



class Config:
    DATASET_PATH =r"C:\Users\USER\pythonProject\yolo\fishphoto\train"
    # SINGLE_TEST_PATH = r"C:\Users\USER\OneDrive\文件\GitHub\projectShare2\unity\aquarium_URP\Assets\StreamingAssets"
    SINGLE_TEST_PATH = r"C:\Users\USER\pythonProject\yolo\fishphoto3\test\Test001"
    RESULT_PATH = r"D:\so sad\_fish\result"
    BATCH_SIZE = 1
    EPOCHS = 5
    MODEL_PATH = r"D:\so sad\_fish\result\model\model_lstm-3.hdf5"
# DATASET_PATH = r"/home/csc/fish/img/fishphoto1/train"
# SINGLE_TEST_PATH = r"/home/csc/fish/img/fishphoto1/test/Test001"
# BATCH_SIZE = 4
# EPOCHS = 5
# MODEL_PATH = r"/home/csc/fish/fish_result/model_lstm-1.hdf5"

# while(True):
#     if os.path.exists(r'C:\Users\USER\OneDrive\文件\GitHub\projectShare2\unity\aquarium_URP\Assets\StreamingAssets'):
#         if os.path.exists(r"C:\Users\USER\OneDrive\文件\GitHub\projectShare2\unity\aquarium_URP\Assets\StreamingAssets\120.png"):
#             print("ecaluating...")
#             prev_time = time.time()
evaluate()
#             print('test img time: ', (time.time() - prev_time))
#             os.remove(r'C:\Users\USER\OneDrive\文件\GitHub\projectShare2\unity\aquarium_URP\Assets\StreamingAssets\*')
#         else:
#             print("not enough")
#             time.sleep(20)
#     else:
#         print("file not exist")
#         time.sleep(360)

