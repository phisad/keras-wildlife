# keras-wildlife

Binary and multiclass animal detection on a wildlife camera trap dataset using Keras and TensorFlow.

A simple baseline for animal detection
- creating the dataset splits for training, validation and test
- learning the animal categories (using weights pre-trained on ImageNet)

# Project Setup

### Clone and install the project

Clone the project to your machine

    git clone git@github.com:phisad/keras-wildlife.git

Install the project scripts by running from the project directory

    python setup.py install clean -a
    
If you want to install to a custom local directory, then create the local site-packages directory and run the install script with the custom local directory as a prefix. Afterwards you have to update the python path to make the custom directory available.

    mkdir -p $HOME/.local/lib/python3.5/site-packages
    python3 setup.py install --prefix=$HOME/.local
    export PYTHONPATH=$HOME/.local/lib/python3.5/site-packages:$PYTHONPATH
    
After installation you can run the following commands from the command line

    wildlife-prepare-images
    wildlife-session
   
The scripts will automatically look for a wildlife-configuration.ini in the users home directory, but you can also specify a configuration file using the `-c` option. The program will prompt you, if no configuration file can be found (see next steps).

### Prepare the configuration file

The commands require a configuration file to be setup. You can find a template file from within the egg file at

    $HOME/.local/lib/python3.5/site-packages/wildlife-0.0.1-py3.5.egg/wildlife/configuration/configuration.ini.template
    
The recommendation is to copy this file to the user directory and rename it to 

    $HOME/wildlife-configuration.ini

The configuration file describes attributes for the model, training and preparation.
    
# Image dataset preparation

The images are down-sized and stored into TFRecord files with their according labels. Notice, that the actual training mode (binary or multiclass) is specified during  the actual training session. In this way, only the training specifies the mapping form animal labels to output classes. The image preparation stores the images along with their original label. 

### Preparing the Wildlife dataset as a target domain

Download the wildlife dataset files put them in a directory like

    # The wildlife dataset directory is supposed to contain a label.csv and the camera split directories 
    WildlifeDatasetDirectoryPath = /data/wildlife

The wildlife preparation script uses the following configuration for the image preparation

    # The dataset directory where split files and tf record file will be written to
    DatasetDirectoryPath = /data/wildlife-project
    
    # The expected image input shape. The image get resized to this shape.
    ImageInputShape = (224, 224, 3)
    
Now you can run the preparation script: 

    wildlife-prepare-images all -d wildlife -m weighted -f target_train target_dev target_test
    
1. The script will list the categories exposed by the prepared *label.csv* in the  configured *WildlifeDatasetDirectoryPath* folder. 
1. Then the script creates for each split (training, validation, testing) an according CSV file (target\_train, target\_dev, target\_test) in the target *DatasetDirectoryPath* folder. 
1. Then the images are read and resized to the configured image input shape. 
1. Finally, the images are put into a tf records file for each split.

Notice: The splits are prefixed with *target*, because this is the targeted image domain for the trained model prediction.

### Preparing ImageNet dataset images as source domain

Download the [ImageNet dataset](http://image-net.org/index) files put them folder-wise in a directory like
    
    # The imagenet dataset directory is supposed to an image folder for each imagenet category to be used 
    ImagenetDatasetDirectoryPath = /data/imagenet
    
The project assumes the following sub directory structure, where each animal category images are contained in a separate folder

     /data/imagenet
     +- deer
     +- cat
     +- dog
     +- ...

Now you can run the preparation script: 

    wildlife-prepare-images all -d imagenet -f source_train source_dev
    
1. Then the script creates for each split (training, validation) an according CSV file (source\_train, source\_dev) in the target *DatasetDirectoryPath* folder. 
1. Then the images are read and resized to the configured image input shape. 
1. Finally, the images are put into a tf records file for each split.

Notice: The source_test split is omitted here, because these are ImageNet images which are not targeted for prediction, but only for narrower pre-training. The splits are prefixed with *source*, because the ImageNet images represent the source domain for pre-training (and thus transfer learning).

# Running the training session

When the dataset is prepared, then the training session can be started.

The wildlife session uses the following configuration for training

    # Which model to run (default: baseline)
    ModelType = baseline
    
    # The classifier to use on top of the vgg model (default: softmax)
    ModelClassifier = softmax
    
    # Whether to apply normalization before classifiers
    UseBatchNormalization = True
    
    # The number of epochs (default: 20)
    Epochs = 20
    
    # The batch size (default: 64)
    BatchSize = 64

You can also specify where to log the tensorboard events and which GPU to use

    # Determine which GPU to use. The default GPU is 0. For CPU only specify -1.
    GpuDevices = 0
    
    # The directory where the metrics will be written as tensorboard logs
    TensorboardLoggingDirectory = /cache/tensorboard-logdir

Now you can run the training script for binary classification training as

    wildlife-session training -s wl-c11

For multiclass training run the same command with the *-m* option:

    wildlife-session training -s wl-c11 -m

The training will automatically prepare and start everything based on the configuration. Notice: For now, the training is by loading all prepared splits into memory.

The script will try to load the dataset from the configured *DatasetDirectoryPath* folder. At the dataset directory path a folder with the given *split_name* must exist. The split name folder contains the TFRecord files and is for example *wl-c11* for wildlife by default. 

After training an evaluation can be performed on the test set:

    wildlife-session evaluate -f /cache/tensorboard-logdir/wildlife/wl-c11/softmax/binary/19-51-33/wildlife-baseline-softmax-bn.h5 -s wl-c11 [-m]

The evaluation is loading the trained model and applied it by default against the *target_test* split. A specific split could be specified with the *split_files* option. In the end, the script creates a report about the models prediction performance.
 
# ToDo
- allow the user to specify the *split_files* when creating the csv files (for now only the selection of the file is possible; on csv creation during prepare always the same names are used)
- allow the user to specify the *split_name* when creating the csv files
- allow the user to configure the category splits and the mappings (for now they are hard coded)
- allow the user to perform a prediction and create an according results file (for now only the analysis report is shown)
