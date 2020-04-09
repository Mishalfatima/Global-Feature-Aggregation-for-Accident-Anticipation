import tensorflow as tf

learning_rate= 1e-3
n_img_hidden = 256
n_hidden = 512
hidden = 256
n_att_hidden = 256

def LSTM_RNN(n_input, n_steps,n_objects, batch_size, n_classes):

     # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)

    # Define a lstm cell with tensorflow
    # Graph input/output
    x = tf.placeholder(tf.float32, [None, n_steps,n_objects, 4096])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep = tf.placeholder(tf.float32, [None])
    lr = tf.placeholder(tf.float32, [])

    # Graph weights
    weights = {

        'em_img': tf.Variable(tf.random_normal([n_input, n_img_hidden], mean=0.0, stddev=0.01)), #Hidden layer weights
        'em_obj': tf.Variable(tf.random_normal([n_input, n_att_hidden], mean=0.0, stddev=0.01)),
        'att_wa': tf.Variable(tf.random_normal([n_hidden, n_att_hidden], mean=0.0, stddev=0.01)),
        'Spatial_W_g': tf.Variable(tf.random_normal([hidden, hidden], mean=0.0, stddev=0.01)),
        'Spatial_W_theta': tf.Variable(tf.random_normal([hidden, hidden], mean=0.0, stddev=0.01)),
        'Spatial_W_phi': tf.Variable(tf.random_normal([hidden, hidden], mean=0.0, stddev=0.01)),
        'out':tf.Variable(tf.random_normal([n_hidden, n_classes], mean=0.0, stddev=0.01))
    }
    biases = {

        'em_img': tf.Variable(tf.random_normal([n_img_hidden], mean=0.0, stddev=0.01)),
        'em_obj': tf.Variable(tf.random_normal([n_att_hidden], mean=0.0, stddev=0.01)),
        'Spatial_W_g': tf.Variable(tf.random_normal([hidden], mean=0.0, stddev=0.01)),
        'Spatial_W_theta': tf.Variable(tf.random_normal([hidden], mean=0.0, stddev=0.01)),
        'Spatial_W_phi': tf.Variable(tf.random_normal([hidden], mean=0.0, stddev=0.01)),
        'out': tf.Variable(tf.random_normal([n_classes], mean=0.0, stddev=0.01))
    }

    lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                        use_peepholes=True, state_is_tuple=False)

    lstm_cell_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1 - keep[0])

    zeros_object = tf.to_float(
         tf.not_equal(tf.reduce_sum(tf.transpose(x[:, :, 1:n_objects, :], [1, 2, 0, 3]), 3), 0))

    # init LSTM parameters
    istate = tf.zeros([batch_size, lstm_cell.state_size])
    h_prev = tf.zeros([batch_size, n_hidden])
    loss = 0.0


    for t in range(n_steps):
        with tf.variable_scope('spatial_model', reuse=tf.AUTO_REUSE):

            X = tf.transpose(x[:, t, :, :], [1, 0, 2])

            full_frame = X[0, :, :]

            #linear embedding of full frame features
            image = tf.matmul(full_frame,weights['em_img']) + biases['em_img']
            x1 = X[1:n_objects,:, :]

            x2 = tf.reshape(x1, [-1, n_input])

            #linear embedding of object features
            n_object = tf.matmul(x2, weights['em_obj']) + biases['em_obj']

            n_object = tf.reshape(n_object, [n_objects - 1, batch_size, n_att_hidden])
            n_object = tf.multiply(n_object, tf.expand_dims(zeros_object[t], 2))


            image_part1 = tf.matmul(n_object, tf.tile(tf.expand_dims(weights['Spatial_W_theta'], 0), [n_objects - 1, 1, 1]))+biases['Spatial_W_theta']


            k = tf.matmul(h_prev, weights['att_wa'])
            theta = tf.tanh(k + image_part1)

            image_part2 = tf.matmul(n_object,
                                    tf.tile(tf.expand_dims(weights['Spatial_W_phi'], 0), [n_objects - 1, 1, 1])) + \
                          biases['Spatial_W_phi']

            phi = tf.tanh(tf.matmul(h_prev, weights['att_wa']) + image_part2)

            g = tf.matmul(n_object, tf.tile(tf.expand_dims(weights['Spatial_W_g'], 0), [n_objects - 1, 1, 1]))+biases['Spatial_W_g']

            theta = tf.transpose(theta, perm=[1,0,2])
            phi = tf.transpose(phi, perm=[1,2,0])

            c = tf.nn.softmax(tf.matmul(theta, phi), axis = 2)
            g = tf.transpose(g, perm=[1, 0, 2])

            vector = tf.matmul(c,g)
            vector = tf.transpose(vector, perm=[1, 0, 2])

            final = tf.reduce_sum(vector + n_object,0)
            fusion = tf.concat([image, final], 1)
            with tf.variable_scope("LSTM") as vs:
                outputs, istate = lstm_cell_dropout(fusion, istate)
                lstm_variables = [v for v in tf.global_variables() if v.name.startswith(vs.name)]

            h_prev = outputs

            zt = tf.matmul(outputs, weights['out']) + biases['out']  # b x n_classes
            # save the predict of each time step

            if t == 0:
                soft_pred = tf.reshape(tf.gather(tf.transpose(tf.nn.softmax(zt), (1, 0)), 1), (batch_size, 1))

            else:
                temp_soft_pred = tf.reshape(tf.gather(tf.transpose(tf.nn.softmax(zt), (1, 0)), 1), (batch_size, 1))

                soft_pred = tf.concat([soft_pred, temp_soft_pred], 1)

            pos_loss = -tf.multiply(tf.exp(-(n_steps - t - 1) / 20.0),
                                    -tf.nn.softmax_cross_entropy_with_logits(logits=zt, labels=y))


            neg_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=zt, labels=y)

            temp_loss = tf.reduce_mean(tf.add(tf.multiply(pos_loss, y[:, 1]), tf.multiply(neg_loss, y[:, 0])))
            loss = tf.add(loss, temp_loss)


    return x, keep, y, loss, lstm_variables, soft_pred, zt, lr








