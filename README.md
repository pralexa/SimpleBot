# Readme

#### Description
This includes a very small start at abstract 'dataset' and 'bot' classes, a quick / dirty implementation of a seq2seq bot,
 an 'echo bot' which just repeats the user's words, and
a data pipeline for the Cornell movie dialog dataset.

Everything will change, this is just fun / starting playing around.

The seq2seq bot and data pipeline are mostly ripped from the Tensorflow fr-en translation example located [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/rnn "rnn").

#### Requires:
Tensorflow

Internet connection for python to download the data

#### Usage:

##### Echo Bot:

Interactive testing / chat:

    python echo_bot.py

##### Seq2Seq Bot:

Training:

    python seq2seq_bot.py

Interactive testing / chat:

    python seq2seq_bot.py --converse=True
    
#### Config:

See seq2seq\_bot.py for available flags (e.g. layers, units, batch size) and defaults.

#### Notes:
It will train very slowly on a laptop. Better to use a GPU server for all but debugging.

On Han Liu's Tesla K40c server, it looks like train time will be on the order of days, depending on # layers etc.

#### Sample conversation trace
After ~100,000 train steps (~8 hours) - should probably train for ~3 days on this dataset.

    (tensorflow)[abeatson@smile simple_bot]$ python seq2seq_bot.py --reload=True
    I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcublas.so locally
    I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcudnn.so locally
    I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcufft.so locally
    I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcuda.so.1 locally
    I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcurand.so locally
    I tensorflow/core/common_runtime/gpu/gpu_device.cc:951] Found device 0 with properties:
    name: Tesla K40c
    major: 3 minor: 5 memoryClockRate (GHz) 0.745
    pciBusID 0000:81:00.0
    Total memory: 11.17GiB
    Free memory: 11.10GiB
    I tensorflow/core/common_runtime/gpu/gpu_device.cc:972] DMA: 0
    I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] 0:   Y
    I tensorflow/core/common_runtime/gpu/gpu_device.cc:1041] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K40c, pci bus id: 0000:81:00.0)
    Reading model parameters from tmp/bot_train/translate.ckpt-95400
    > Hello
    Hello .
    > What is your name ?
    Uh huh ?? William name does .
    > Who are you?
    My name is _UNK .
    > What do you do?
    I ' m retired . . .
    > That's nice
    Yeah .
    > Tell me a story
    Uh . . . uh . . .
    > You're boring
    I ' m not .
    > How is your day today?
    Oh , my good .
    > Goodbye
    Goodbye .


