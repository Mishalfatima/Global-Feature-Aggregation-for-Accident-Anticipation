
from read_data import *
import importlib

import numpy as np
import tensorflow as tf
import skimage
import skimage.io
import skimage.transform
import vgg16
import pickle
import os
import numpy as np

path = []
labels =[]
n_classes = 2
batch_size=1
features_path = "/hdd/local/sda/mishal/pyflow/ECA_Data/f/"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def extract_features():
    infile = open("/hdd/local/sda/mishal/CrashCatcher/A3D_labels.pkl", 'rb')
    new_dict = pickle.load(infile)

    dashcam = Dashcam_data(train=False, batch_size=batch_size)

    abnormal_objects = "/hdd/local/sda/mishal/pyflow/objects/test/abnormal_objects.npy"
    normal_objects = "/hdd/local/sda/mishal/pyflow/objects/test/normal_objects.npy"

    abnormal_objects = np.load(abnormal_objects, allow_pickle=True)
    normal_objects = np.load(normal_objects, allow_pickle=True)

    images = tf.placeholder("float", [100, 224, 224, 3])
    vgg = vgg16.Vgg16()
    with tf.name_scope("content_vgg"):
        vgg.build(images)

    sess = tf.Session(
                config=tf.ConfigProto(gpu_options=(tf.GPUOptions(allow_growth=True)), log_device_placement=True))

    v = []
    for i in range(len(abnormal_objects)):

        k = abnormal_objects[i]
        for frame in range(len(k)):

            if (k[frame] != []):
                objects = k[frame]

                path = objects[0]


                f = path[0].split("/")


                v.append(f[-3])


                break
    a = []
    for i in range(len(normal_objects)):
        o = 0
        k = normal_objects[i]
        for frame in range(len(k)):
            o = 0

            if (k[frame] != []):
                o = o + 1
                objects = k[frame]

                path = objects[0]
                f = path[0].split("/")
                a.append(f[-2])

                break

        if (o == 0):
            w = str(i+1).zfill(6)
            a.append(w)

    tot_batches = dashcam.total_videos/batch_size


    for i in range(int(tot_batches)):

        batch, video_labels, frame_labels = dashcam.get_next_batch()
        x_scratch = np.zeros((batch_size, 100, 20, 224, 224, 3))
        for k in range(batch_size):


            video_path = batch[k]
            print(video_path)

            path, dirs, files = next(os.walk(video_path))
            paths = [video_path + "/" + x for x in files]

            frames = np.sort(paths)
            print(video_labels)
            print(frame_labels)
            print(frames)

            h = len(frames)-100


            if (h<0) or h == 0:
                h = 0
                ggg = len(frames)

            elif (h > 0):
                arr = frame_labels[h:len(frames)]
                arr1 = frame_labels[0:100]

                itemindex = np.argwhere(arr == 1)
                itemindex1 = np.argwhere(arr1 == 1)

                if(list(itemindex1) == []):
                    ggg = len(frames)
                else:
                    ggg = 100
                    h = 0


                '''if (len(itemindex)>len(itemindex1) and itemindex.min()!=0):
                    ggg = len(frames)
                else:
                    h = 0
                    ggg = 100'''

            g = video_labels[k][0][1]
            if g == 1:

                u = frames[0].split("/")
                video_name = u[-3]
                video = abnormal_objects[v.index(video_name)]

            elif (g == 0):

                u = frames[0].split("/")
                video_name = u[-2]

                video = normal_objects[a.index(video_name)]

            for file in range(h,ggg):

                imag = skimage.io.imread(frames[file])
                #imag = cv2.imread(frames[file])

                #imag1 = skimage.transform.resize(imag, (640, 640))
                img_height = imag.shape[0]
                img_width = imag.shape[1]


                imag = imag / 255.0
                assert (0 <= imag).all() and (imag <= 1.0).all()
                resized_img = skimage.transform.resize(imag, (224, 224))
                img_data = resized_img.reshape((1, 224, 224, 3))


                if (h== 0):
                    print("I am here ")
                    x_scratch[k, file, 0 ,:,:,:] = img_data

                else:
                    x_scratch[k, file-h, 0, :, :, :] = img_data

                frame = video[file]


                if (len(frame) > 19):
                    frame = frame[0:19]

                if (frame != []):

                    for objects in range(len(frame)):
                        item = np.zeros((4))
                        object = frame[objects]


                        if (object[5] == 1 or object[5] == 2 or object[5] == 3 or \
                                object[5] == 4 or object[5] == 6 or object[5] == 8):

                            item[0] = object[1]
                            item[1] = object[2]
                            item[2] = object[3]
                            item[3] = object[4]

                            item = [int(float(item[0]) * img_height), int(float(item[1]) * img_width),
                                        int(img_height * (float(item[2]) - float(item[0]))),
                                        int(img_width * (float(item[3]) - float(item[1])))]

                            box = [item[0], item[1], item[0] + item[2], item[1] + item[3]]

                            cropped_image =imag[box[0]:box[2], box[1]:box[3],:]
                            resized_img = skimage.transform.resize(cropped_image, (224, 224))
                            img_data = resized_img.reshape((1, 224, 224, 3))

                            if (h == 0):
                                x_scratch[k, file, objects+1, :, :, :] = img_data

                            else:
                                x_scratch[k, file - h, objects+1, :, :, :] = img_data


        x_ = x_scratch.reshape([-1,224,224,3])

        for f in range(0,2000,100):
            ba = x_[f:f+100,:,:,:]

            feed_dict = {images: ba}

            prob = sess.run(vgg.fc6, feed_dict=feed_dict)

            if (f == 0):
                temp1 = prob

            else:

                temp1 = np.concatenate((temp1,prob),axis=0)

        prob = temp1.reshape([100,20,4096])

        gg = batch[0].split("/")

        if (video_labels[0][0][1] == 1):
            video_name = gg[-1]
            array = new_dict[video_name]['target']

            if (len(array)) < 100:
                indexx = len(array)

                aa = np.zeros((100 - indexx, 20, 4096))

                prob[indexx:100, :, :] = aa

        path = features_path + "batch_" + str(i).zfill(3) + ".npz"

        np.savez(path, data=prob, labels=video_labels, paths=batch)
        print("videos ", i, "done out of", int(tot_batches))


if __name__=="__main__":
    #extract_features()
    #imag1 = cv2.imread("/hdd/local/sda/mishal/pyflow/ECA_Data/train_videos/TJBygxemAZI_000008/images/000001.jpg")       #"/hdd/local/sda/mishal/Anticipating-Accidents-master/dataset/videos/training/frames_test/000330/001.jpg")

    #imag2 = cv2.imread("/hdd/local/sda/mishal/Anticipating-Accidents-master/dataset/videos/training/frames_test/000330/002.jpg")
    #normal_objects = np.load("/hdd/local/sda/mishal/pyflow/objects/train/normal_objects.npy",allow_pickle=True)
    #print(normal_objects)


    '''a = []
    for i in range(len(normal_objects)):
        o = 0
        k = normal_objects[i]
        for frame in range(len(k)):
            o = 0

            if (k[frame] != []):
                o = o + 1
                objects = k[frame]


                path = objects[0]
                f = path[0].split("/")
                a.append(f[-2])

                break

        if (o == 0):
            w = str(i + 1).zfill(6)
            a.append(w)
    video = normal_objects[a.index("000772")]
    print(video[0])'''

    imag1 = skimage.io.imread("/hdd/local/sda/mishal/pyflow/ECA_Data/test_videos/uRHUXf93pfI_000002/images/000001.jpg")
    image = tf.placeholder("float", [1, 224, 224, 3])

    imag1 = imag1 / 255.0
    assert (0 <= imag1).all() and (imag1 <= 1.0).all()
    resized_img = skimage.transform.resize(imag1, (224, 224))
    img_data1 = resized_img.reshape((1, 224, 224, 3))

    '''imag2 = imag2 / 255.0
    assert (0 <= imag2).all() and (imag2 <= 1.0).all()
    resized_img = skimage.transform.resize(imag2, (224, 224))
    img_data2 = resized_img.reshape((1, 224, 224, 3))'''

    #imag = np.concatenate((img_data1, img_data2), axis=0)


    vgg = vgg16.Vgg16()
    with tf.name_scope("content_vgg"):
        vgg.build(image)

    sess = tf.Session(
                config=tf.ConfigProto(gpu_options=(tf.GPUOptions(allow_growth=True)), log_device_placement=True))

    prob = sess.run(vgg.fc6, feed_dict={image:img_data1})
    print(prob)

    '''infile = open("/hdd/local/sda/mishal/CrashCatcher/A3D_labels.pkl", 'rb')
    new_dict = pickle.load(infile)
    path = "/hdd/local/sda/mishal/pyflow/ECA_Data/features/training/"
    #features_path = "/hdd/local/sda/mishal/pyflow/training_zeros/features/"
    acc = 0
    a = 0
    #extract_features("batch_1076.npz", "Pkbqo1FfhXA_000005", vgg, sess)
    b = 0
    for file in os.listdir(path):

        f = path + file
        h = np.load(f)

        p = h['paths']
        data = h['data']
        labels =h['labels']

        g = p[0].split("/")


        if (labels[0][0][0]== 0):

            video_name = g[-2]
            array = new_dict[video_name]['target']

            if (len(array))<100:
                #index = len(array)

                #a = np.zeros((100 - index, 20, 4096))

                #data[index:100, :, :] = a
                itemindex = np.argwhere(array == 1)
                s = itemindex.min()
                acc = acc+s
                b+=1


            if (len(array)>100):


                #extract_features(file,video_name,vgg,sess)
                h = len(array)-100
                arr = array[h:len(array)]
                arr1 = array[0:100]

                itemindex = np.argwhere(arr == 1)
                itemindex1 = np.argwhere(arr1 == 1)

                if (list(itemindex1) == []):
                    s = itemindex.min()
                else:
                    s = itemindex1.min()


                acc=acc+s
                b+=1

            a+=1
        #pa = features_path + file

        #np.savez(pa, data=data, labels=labels, paths=p)

    print(acc)
    print(a)
    print(b)'''