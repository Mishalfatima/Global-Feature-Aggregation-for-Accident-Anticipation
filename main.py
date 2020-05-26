import tensorflow as tf
from RNN import *
import time
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

batch_size = 10
n_input = 4096
n_classes = 2
n_hidden = 512
learning_rate= 1e-3
lambda_loss_amount = 0.5
num_epochs=50
display_iter = 10
save_path = './models/'
n_frames = 100
train_num = 126
test_num = 46
n_objects = 10

train_path = "/hdd/local/sda/mishal/Dashcam/training/"
test_path = "/hdd/local/sda/mishal/Dashcam/testing/"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='accident_LSTM')
    parser.add_argument('--mode',dest='mode',help='train or test or viz',default='viz')
    parser.add_argument('--model',dest='model',default='./models/')
    parser.add_argument('--gpu',dest='gpu',default='1')
    parser.add_argument('--restore', type=bool, default=True)
    args = parser.parse_args()

    return args

def train(args):
    with tf.device('/device:GPU:1'):
        print('..training..')

        x, keep, y, loss, lstm_variables, soft_pred, zt, lr = LSTM_RNN(n_input,n_frames,n_objects,batch_size,n_classes)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss / n_frames)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

        previous_runs = os.listdir('output')
        if len(previous_runs) == 0:
            run_number = 1
        else:
            run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1

        logdir = 'run_%02d' % run_number
        '''if os.path.isdir(logdir) == False:
            os.mkdir(logdir)'''
        writer = tf.summary.FileWriter(os.path.join('output', logdir), sess.graph)

        if os.path.isdir(save_path) == False:
            os.mkdir(save_path)

        # Launch the graph
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(max_to_keep=100)

        model_path = args.model
        f = model_path.split("/")

        if (args.restore == True):
            saver.restore(sess, args.model)
            qq = f[-1][-1]

        else:
            qq = 0

        for epoch in range(int(qq),num_epochs):
            tStart_epoch = time.time()

            # To keep track of training's performance
            epoch_loss = np.zeros((train_num, 1), dtype=float)

            for step in range(1, train_num+1):

               file_name = '%03d' % (step)
               batch_data = np.load(train_path + 'batch_' + file_name + '.npz')

               batch = batch_data["data"]
               batch_xs = batch[:, :, 0:n_objects,:]
               batch_y = batch_data["labels"]

               if (epoch <= 10):
                   learning_rate = 0.0001

               elif (epoch > 10 and epoch < 20):

                   learning_rate = 0.0001

               elif (epoch >= 20):
                   learning_rate = 0.0001

               _, batch_loss = sess.run([optimizer, loss],
                                        feed_dict={x: batch_xs, y: batch_y, keep: [0.5], lr: learning_rate})
               epoch_loss[step - 1] = batch_loss / batch_size

               print("Batches done", step, " out of", int(128), "Epoch is ", epoch)

            epochloss = np.mean(epoch_loss)
            summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=epochloss)])
            # print one epoch
            print("Epoch:", epoch + 1, " done. Loss:", epochloss)
            writer.add_summary(summary, i)
            i += 1
            tStop_epoch = time.time()
            print("Epoch Time Cost:", round(tStop_epoch - tStart_epoch, 2), "s")
            sys.stdout.flush()
            if (epoch + 1) % 10 == 0:
                saver.save(sess, save_path + "model", global_step=epoch + 1)
            '''print("Training")
            test_all(sess,x, keep,end, y, loss, lstm_variables, soft_pred, train=True)'''

            if (epoch + 1) % 10 == 0:
                print("Testing")
                test_all(sess, x, keep, y, loss, lstm_variables, soft_pred, train=False)
        print("Optimization Finished!")
        saver.save(sess, save_path + "final_model")

def test_all(sess,x,keep, y,loss,lstm_variables, soft_pred,train=True):

    videos = []
    total_loss = 0.0
    acc = 0
    all_pred = []

    o = 0
    for num_batch in range(1,test_num+1):
        print(acc)
        acc = acc + 1

        file_name = '%03d' % (num_batch)
        test_all_data = np.load(test_path+'batch_'+file_name+'.npz')

        batch = test_all_data["data"]
        batch_xs = batch[:, :, 0:n_objects, :]
        test_labels = test_all_data["labels"]

        [temp_loss, pred] = sess.run([loss, soft_pred],
                                         feed_dict={x: batch_xs, y: test_labels, keep: [0.0]})

        if num_batch <= 1:
                all_pred = pred[:,0:90]
                all_labels = np.reshape(test_labels[:, 1], [batch_size, 1])

        else:
                all_pred = np.vstack((all_pred,pred[:,0:90]))
                all_labels = np.vstack((all_labels, np.reshape(test_labels[:, 1], [batch_size, 1])))

    '''np.save("all_pred_new_1.npy", all_pred)
    np.save("all_labels_new_1.npy", all_labels)

    all_pred = np.load("all_pred_new_1.npy", allow_pickle=True)
    all_labels = np.load("all_labels_new_1.npy", allow_pickle=True)'''

    evaluation(all_pred, all_labels)

def evaluation(all_pred,all_labels, total_time = 90, vis = False, length = None):
    ### input: all_pred (N x total_time) , all_label (N,)
    ### where N = number of videos, fps = 20 , time of accident = total_time
    ### output: AP & Time to Accident
    print(len(all_pred))
    length = []
    for i in range(len(all_pred)):
        a = len(all_pred[i])
        # r = [a]*a
        length.append(a)
    # length = [total_time] * all_pred.shape[0]
    # temp_shape = all_pred.shape[0]*total_time

    flat_list = [item for sublist in all_pred for item in sublist]
    # length = [item for sublist in length for item in sublist]
    temp_shape = len(flat_list)
    # print(all_labels)

    Precision = np.zeros((temp_shape))
    Recall = np.zeros((temp_shape))
    Time = np.zeros((temp_shape))
    cnt = 0
    AP = 0.0

    flat_list = [item for sublist in all_pred for item in sublist]
    a = 0
    for Th in sorted(flat_list):
        print(a, "out of ",len(flat_list) )
        a+=1
        if length is not None and Th == 0:
            continue
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0
        Fn = 0.0
        F_P = []
        Fp = 0.0

        for i in range(len(all_pred)):

            j = np.array(all_pred[i])
            tp = np.where(j * all_labels[i] >= Th)

            if (all_labels[i] == 1):
                Fn += float(len((np.where(j <= Th)[0] > 0)))

            if (all_labels[i] == 0):
                Fp = float(len((np.where(j >= Th)[0] > 0)))
                F_P.append(Fp)

            Tp += float(len(tp[0] > 0))
            
            if float(len(tp[0] > 0)) > 0:
                time += tp[0][0] / float(length[i])
                counter = counter + 1
            Tp_Fp += float(len(np.where(j >= Th)[0] > 0))
        if Tp_Fp == 0:
            Precision[cnt] = np.nan
        else:
            Precision[cnt] = Tp / Tp_Fp

        if np.sum(all_labels) == 0:
            Recall[cnt] = np.nan
        else:
            # Recall[cnt] = Tp/np.sum(all_labels)
            Recall[cnt] = Tp / (Tp + Fn)

        if counter == 0:
            Time[cnt] = np.nan
        else:

            Time[cnt] = (1 - time / counter)*4.5
        cnt += 1

    np.save("Precision_Dashcam_LSTM.npy", Precision)
    np.save("Recall_Dashcam_LSTM.npy", Recall)
    np.save("Time.npy",Time)
    '''Recall = np.load("Recall_Dashcam_LSTM.npy")'''
    index = np.argsort(Recall)
    TT = Time[index]
    RR = Recall[index]

    plt.figure()
    plt.plot(RR,TT)
    plt.xlim(0, 1)
    plt.ylim(0,4.5)
    plt.ylabel('TTA')
    plt.xlabel('Recall')
    name = "TTA-Recall.jpg"
    plt.savefig(name)

    print("Mean Precision is ", np.mean(Precision))
    print("Mean Recall is ", np.mean(Recall))
    print("Mean Time is ", np.mean(Time))

    np.save("Precision_Dashcam_LSTM.npy", Precision)
    np.save("Recall_Dashcam_LSTM.npy", Recall)


    a += 1

    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    _, rep_index = np.unique(Recall, return_index=1)
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index) - 1):
        new_Time[i] = np.max(Time[rep_index[i]:rep_index[i + 1]])
        new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i + 1]])

    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    new_Time = new_Time[~np.isnan(new_Precision)]
    new_Recall = new_Recall[~np.isnan(new_Precision)]
    new_Precision = new_Precision[~np.isnan(new_Precision)]


    if new_Recall[0] != 0:
        AP += new_Precision[0] * (new_Recall[0] - 0)
    for i in range(1, len(new_Precision)):
        AP += (new_Precision[i - 1] + new_Precision[i]) * (new_Recall[i] - new_Recall[i - 1]) / 2

    print("Average Precision= " + "{:.4f}".format(AP) + " ,mean Time to accident= " + "{:.4}".format(
        np.mean(new_Time) * 4.5))


def test(model_path):
    # load model
    x, keep, y, loss, lstm_variables, soft_pred, zt, lr = LSTM_RNN(n_input,n_frames,n_objects,batch_size,n_classes)
    # inistal Session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options))
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print("model restore!!!")
    #print("Training")
    #test_all(sess,x,keep,end, y,loss,lstm_variables, sub_lstm_variables,soft_pred,train=True)
    print("Testing")
    test_all(sess,x,keep, y,loss,lstm_variables, soft_pred,train=False)


def viz(args):

    '''pathh = "/hdd/local/sda/mishal/CrashCatcher/figs/"
    for i in range(all_pred.shape[0]):
        if all_labels[i] == 1:
            plt.figure(figsize=(14, 5))
            plt.plot(all_pred[i, 0:90], linewidth=3.0)
            plt.ylim(0, 1)
            plt.ylabel('Probability')
            plt.xlabel('Frame')
            name = pathh + str(i)+".jpg"
            plt.savefig(name)'''

    pathh = "./CrashCatcher/figs_all/"
    x, keep, y, loss, lstm_variables, soft_pred, zt, lr = LSTM_RNN(n_input,n_frames,n_objects,batch_size,n_classes)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    # restore model
    saver.restore(sess, args.model)
    path = "./Dashcam/testing/"
    # load data
    for num_batch in range(1, test_num):
        print(num_batch)
        file_name = '%03d' % num_batch
        all_data = np.load(path + 'batch_' + file_name + '.npz')
        data = all_data['data']
        data = data[:,:,0:10,:]
        labels = all_data['labels']
        ID = all_data['ID']

        # run result
        [all_loss, pred] = sess.run([loss, soft_pred], feed_dict={x: data, y: labels, keep: [0.0]})
        for i in range(len(ID)):
                time = 0.0

                plt.figure(figsize=(14, 5))
                prediction = pred[i,0:90]
                j = np.array(prediction)
                tp = np.where(j * labels[i][1] >= 0.8)
                if float(len(tp[0] > 0)) > 0:
                    time = (1 - (tp[0][0] / float(90))) * 4.5

                print(time)
                plt.plot(prediction, linewidth=3.0)
                yy = 0.8 * np.ones(90)
                #plt.axhline(y=0.8, color='r', linestyle='--')
                plt.plot(yy, color='r', linestyle='--')
                plt.ylim(0, 1)
                plt.xlim(0,90)
                #idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
                #plt.plot(x[idx], f[idx], 'ro')
                #idx = np.argwhere(np.diff(np.sign(yy - prediction))).flatten()
                #plt.plot(idx, prediction[idx], 'ro')
                plt.show()
                plt.ylabel('Probability')
                plt.xlabel('Frame')
                plt.xticks(np.arange(10,100,10))
                name = pathh + str(ID[i]) +"-"+str(labels[i][1])+"-"+str(time)+ ".jpg"
                plt.savefig(name)


if __name__ == '__main__':

    if __name__ == '__main__':
        args = parse_args()
        if args.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'

        if args.mode == 'train':
            train(args)
        elif args.mode == 'test':
            test(args.model)
        elif args.mode == 'viz':
            viz(args)


