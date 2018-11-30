import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import inception_preprocessing
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import os
import time
slim = tf.contrib.slim

from datasets import flowers
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import slim
import os
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
import time


# GUIDE ! https://kwotsin.github.io/tech/2017/02/11/transfer-learning.html

#================ DATASET INFORMATION ======================
#State dataset directory where the tfrecord files are located
dataset_dir = '/home/aewin/work/anaconda3/code'

#State where your log file is at. If it doesn't exist, create it.
log_dir = '/home/aewin/work/anaconda3/code'
DATA_DIR = log_dir

#State where your checkpoint file is
checkpoint_file = '/home/aewin/work/anaconda3/code/inception_resnet_v2_2016_08_30.ckpt'
checkpoint_file = '/home/aewin/work/anaconda3/code/inception_resnet_v2_2016_08_30.ckpt'

#State the image size you're resizing your images to. We will use the default inception size of 299.
image_size = 299

#State the number of classes to predict:
num_classes = 5

#State the labels file and read it
labels_file = '/home/aewin/work/anaconda3/code/models/research/slim/smorking/labels.txt'
labels = open(labels_file, 'r')

#Create a dictionary to refer each label to their string name
labels_to_name = {}


# a dictionary of flowers_data_dir name
flowers_data_dir = '/home/aewin/work/anaconda3/code/models/research/slim/smorking'



#Create the file pattern of your TFRecord files so that it could be recognized later on
file_pattern = 'flowers_%s_*.tfrecord'


#================= TRAINING INFORMATION ==================
#State the number of epochs to train
num_epochs = 1

#State your batch size
batch_size = 8

#Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 0.0002
learning_rate_decay_factor = 0.7
num_epochs_before_decay = 2


# In[10]:



#============== DATASET LOADING ======================
#We now create a function that creates a Dataset class which will give us many TFRecord files to feed in the examples into a queue in parallel.
def get_split(split_name, dataset_dir, file_pattern=file_pattern, file_pattern_for_counting='flowers'):
    '''
    Obtains the split - training or validation - to create a Dataset class for feeding the examples into a queue later on. This function will
    set up the decoder and dataset information all into one Dataset class so that you can avoid the brute work later on.
    Your file_pattern is very important in locating the files later. 

    INPUTS:
    - split_name(str): 'train' or 'validation'. Used to get the correct data split of tfrecord files
    - dataset_dir(str): the dataset directory where the tfrecord files are located
    - file_pattern(str): the file name structure of the tfrecord files in order to get the correct data
    - file_pattern_for_counting(str): the string name to identify your tfrecord files for counting

    OUTPUTS:
    - dataset (Dataset): A Dataset class object where we can read its various components for easier batch creation later.
    '''

    #First check whether the split_name is train or validation
    if split_name not in ['train', 'validation']:
        raise ValueError('The split_name %s is not recognized. Please input either train or validation as the split_name' % (split_name))

    #Create the full path for a general file_pattern to locate the tfrecord_files
    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))

    #Count the total number of examples in all of these shard
    num_samples = 0
    file_pattern_for_counting = file_pattern_for_counting + '_' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    #Create a reader, which must be a TFRecord reader in this case
    reader = tf.TFRecordReader

    #Create the keys_to_features dictionary for the decoder
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    #Create the items_to_handlers dictionary for the decoder.
    items_to_handlers = {
    'image': slim.tfexample_decoder.Image(),
    'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    #Start to create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    #Create the labels_to_name file
    labels_to_name_dict = labels_to_name

    #Actually create the dataset
    dataset = slim.dataset.Dataset(
        data_sources = file_pattern_path,
        decoder = decoder,
        reader = reader,
        num_readers = 4,
        num_samples = num_samples,
        num_classes = num_classes,
        labels_to_name = labels_to_name_dict,
        items_to_descriptions = items_to_descriptions)

    return dataset


def load_batch(dataset, batch_size, height=image_size, width=image_size, is_training=True):
    '''
    Loads a batch for training.

    INPUTS:
    - dataset(Dataset): a Dataset class object that is created from the get_split function
    - batch_size(int): determines how big of a batch to train
    - height(int): the height of the image to resize to during preprocessing
    - width(int): the width of the image to resize to during preprocessing
    - is_training(bool): to determine whether to perform a training or evaluation preprocessing

    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).

    '''
    #First create the data_provider object
    
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity = 24 + 3 * batch_size,
        common_queue_min = 24)
        
    #BY using data_provider TO GET raw image
    raw_image, label = data_provider.get(['image', 'label'])

    #Perform the correct preprocessing for this image depending if it is training or evaluating
    image = inception_preprocessing.preprocess_image(raw_image, height, width, is_training)

    #As for the raw images, we just do a simple reshape to batch it up
    ## keep raw image that is not preprocessed for the inception model so that
    #we can display it as an image in its original form. We only do a 
    #simple reshaping so that it fits together nicely in one batch. 
    #tf.expand_dims will expand the 3D tensor 
    #from a [height, width, channels] shape to [1, height, width, channels] shape, 
    #while tf.squeeze will simply eliminate all the dimensions with the number ‘1’, 
    #which brings the raw_image back to the same 3D shape after reshaping
    
    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)


    #Finally, we just create the images and labels batch, using multiple threads to dequeue the examples 
    #for training. The capacity is simply the capacity for the internal FIFO queue that exists 
    #by default when you create a tf.train.batch, and a higher capacity is recommended if you have an unpredictable data input/output. This can data I/O stability can be seen through a summary created by default on TensorBoard when you use the tf.train.batch function. We also let allow_smaller_final_batch be True to use the last few examples even if they are insufficient to make a batch.
    images, raw_images, labels = tf.train.batch(
        [image, raw_image, label],
        batch_size = batch_size,
        num_threads = 4,
        capacity = 4 * batch_size,
        allow_smaller_final_batch = True)

    return images, raw_images, labels

def run():
    #Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    ## Now we are finally ready to construct the graph! We first start by setting the logging level to INFO (which gives us sufficient information for training purposes), and load our dataset.
    

    #======================= TRAINING PROCESS =========================
    #Now we start to construct the graph and build our model
    ##Now we are finally ready to construct the graph! We first start by setting the logging level to INFO (which gives us sufficient information for training purposes), and load our dataset.
    #https://blog.gtwang.org/programming/tensorflow-read-write-tfrecords-data-format-tutorial/

    '''  WHY using  tf.Graph().as_default():!!!!!!!!!!!!!!!
       Since a default graph is always registered, every op and variable is placed into the default graph. 
       The statement of  tf.Graph().as_default(): , however,creates a new graph and places everything (declared inside its scope) into this graph.
       If the graph is the only graph, it's useless. 
       But it's a good practice because if you start to work with many graphs it's easier to understand where ops and 
       vars are placed. 
       Since this statement costs you nothing, it's better to write it anyway. 
       Just to be sure that if you refactor the code in the future, 
       the operations defined belong to the graph you choose initially
    '''
    with tf.Graph().as_default() as graph:

    
        dataset = flowers.get_split('traning', flowers_data_dir)
        images, _, labels = load_batch(dataset, batch_size=batch_size)

        #Know the number steps to take before decaying the learning rate and batches per epoch
        #Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch #Because one step is one batch processed
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

        #Create the pretrain-model  inference
        #This function is used to init model and input necessary papermate
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(images, num_classes = dataset.num_classes, is_training = True)
        '''    
            logit = w*x + b,
            x: input, w: weight, b: bias. That's it.
            
            logit is defined as the output of a neuron without applying activation function:
            Define the scopes that you want to exclude for restoration (force to train!)
        '''
        
        '''
        when you are training on grayscale images, you would have to remove the initial input convolutional layer, which assumes you have an RGB image with 3 channels, if you set the argument channels=3 for the Image decoder in the get_split function. In total, here are the 3 scopes that you can exclude:

        InceptionResnetV2/AuxLogits
        InceptionResnetV2/Logits
        InceptionResnetV2/Conv2d_1a_3x3 (Optional, for Grayscale images)
        '''
        
        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
        
        variables_to_restore = slim.get_variables_to_restore(exclude = exclude)

        #Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
        '''
        why we use one_hot_lables?  in order to count out the:cross-entropy
        The main benefits of this are:
        1.Solved the problem classifier is not good at processing attribute data
        2.expanding features.
        '''
        
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)

        #Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
        # counte level of softmax_cross_entropy
        loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits)
        total_loss = tf.losses.get_total_loss()    #obtain the regularization losses as well
        #The total loss is defined as the cross entropy loss plus all of the weight
        
        #Create the global step variable for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        #Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate = initial_learning_rate,
            global_step = global_step,
            decay_steps = decay_steps,
            decay_rate = learning_rate_decay_factor,
            staircase = True)

        #Now we can define the optimizer that takes on the learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)

        '''        
        create_train_op perform more functions like gradient clipping or multiplication to 
        prevent exploding or vanishing gradients. 
        This is done rather than simply doing an Optimizer.minimize function, 
        which simply just combines compute_gradients and 
        apply_gradients without any gradient processing after compute_gradients.
        '''
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        ''' 
        Now we simply get the predictions through extracting the probabilities predicted 
        from end_points['Predictions'], 
        and perform an argmax function that returns us the index of the highest probability,
        which is also the class label.
        '''
        
        predictions = tf.argmax(end_points['Predictions'], 1)
        probabilities = end_points['Predictions']
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        
        metrics_op = tf.group(accuracy_update, probabilities)


        #Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', lr)
        my_summary_op = tf.summary.merge_all()

        #Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        # train_step function takes in a session and runs all these ops together .
        def train_step(sess, train_op, global_step):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            #Check the time for each sess run
            start_time = time.time() 
            
            '''invoke sess.run to execute matrix multiplication to do cauculate 3 op train_op, global_step, metrics_op into a numpy arry and retun it!!'''
            total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
            
            time_elapsed = time.time() - start_time
            logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

            return total_loss, global_step_count

        #Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        #we have defined our variables to restore  .
        #we need asign our variables into model ervertime sinec we load pre traing model of  inception_resnet_v2_2016_08_30

        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        #Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        #supervisor is especially useful when you are training your models for many days. I
        #supervisor helps you deal with summaryWriter and the initialization of your global and local variables
        sv = tf.train.Supervisor(logdir = log_dir, summary_op = None, init_fn = restore_fn)


        #Run the managed session
        ''' Why we need to use managed_session
            auto check checkpoint to save data
            auto saver checkpoint
            auto summary_computed
            
            
        '''
        #Run the managed session
        with sv.managed_session() as sess:
            for step in range(num_steps_per_epoch * num_epochs):
                #At the start of every epoch, show the vital information:
                if (step % num_batches_per_epoch) == 0:
                    print('Epoch %s/%s' %(step/num_batches_per_epoch + 1, num_epochs))
                    learning_rate_value, accuracy_value = sess.run([lr, accuracy])
                    print('Current Learning Rate: %s' %(learning_rate_value))
                    print('Current Streaming Accuracy: %s' %(accuracy_value))

                    # optionally, print your logits and predictions for a sanity check that things are going fine.
                    logits_value, probabilities_value, predictions_value, labels_value = sess.run([logits, probabilities, predictions, labels])
                    print( 'logits: \n', logits_value)
                    print( 'Probabilities: \n', probabilities_value)
                    print( 'predictions: \n', predictions_value)
                    print( 'Labels:\n:', labels_value)

                #Log the summaries every 10 step.
                if (step % 10) == 0:
                    loss, _ = train_step(sess, train_op, sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)
                    
                #If not, simply run the training step
                else:
                    loss, _ = train_step(sess, train_op, sv.global_step)

            #We log the final training loss and accuracy
            print('Final Loss: %s' %loss)
            print('Final Accuracy: %s' %sess.run(accuracy))
                          

            #Once all the training has been done, save the log files and checkpoint model
            print('Finished training! Saving model to disk now.')
            # saver.save(sess, "./flowers_model.ckpt")
            sv.saver.save(sess, sv.save_path, global_step = sv.global_step)

if __name__ == '__main__':
    run()

                

