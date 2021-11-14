# ADAPTING THE SOLUTION FROM HW 4, since there is already logic to 1. load and split datasets, 2. create models of various sizes and methods (DFF and CNN), and 3. plot history and metrics!


#MODIFIED FROM CHOLLETT P120 
from math import floor
from keras import layers, regularizers, models
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")


# ------------------------------------
# Flags for Settings
# ------------------------------------
dataset = 'MNIST' # MNIST, FASHION-MNIST, or CIFAR-10
n_classes = 10 # 10 for MNIST, FASHION-MNIST, and CIFAR-10
method = 'CNN' # CNN, DFF, or 'LOAD path/to/model' (Convolutional Neural Network or Dense Feed Forward or Load a saved model)
save='my_model_' + dataset + '_' + method # None if no save, path/to/model to save
L2_CONSTANT = 1e-4
ACT_TYPE = 'relu'
N_HIDDEN = 4
IS_AUGMENTED = False
epochs=5
bottleneck_size = 32
N_NODES = bottleneck_size

#-------------------------------------
#GET DATA AND REFORMAT
#-------------------------------------
#from keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical
validation_percentage=0.2

if dataset == 'MNIST':
    from keras.datasets import mnist, fashion_mnist
    (images_train, labels_train), (test_images, test_labels) = mnist.load_data()
    (test_images, _), (test_val_images, _) = fashion_mnist.load_data() # For the AutoEncoder problem, we are training an autoencoder on MNIST, and testing it on Fashion MNIST!
elif dataset == 'FASHION-MNIST':
    from keras.datasets import fashion_mnist
    (images_train, labels_train), (test_images, test_labels) = fashion_mnist.load_data()
elif dataset == 'CIFAR-10':
    from keras.datasets import cifar10
    (images_train, labels_train), (test_images, test_labels) = cifar10.load_data()
else:
    print("Error! Please specify: dataset 'MNIST', 'FASHION-MNIST', or 'CIFAR-10'")
    exit()


train_total = labels_train.shape[0]
train_num = floor(train_total * (1 - validation_percentage))
test_num = test_images.shape[0]

image_dim = images_train.shape[1:]
if len(image_dim) == 2:
    image_dim = image_dim + (1,)

print('Image Dimensions: ' + str(image_dim))

train_images, val_images, train_labels, val_labels = train_test_split(images_train, labels_train, test_size=validation_percentage)

print('Train shape before: ' + str(images_train.shape))
print('Train shape after: ' + str(train_images.shape))

train_images = train_images.reshape((train_num, image_dim[0], image_dim[1], image_dim[2]))
val_images = val_images.reshape((val_labels.shape[0], image_dim[0], image_dim[1], image_dim[2]))
test_images = test_images.reshape((test_num, image_dim[0], image_dim[1], image_dim[2]))

#NORMALIZE
train_images = train_images.astype('float32') / 255 
val_images = val_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255  


# Show random image -- tested this on a Linux box so not entirely sure if it'll show!
img_path = '/Users/fchollet/Downloads/cats_and_dogs_small/test/cats/cat.1700.jpg'
from keras.preprocessing import image
import numpy as np
img_tensor = train_images[1]
#img_tensor = np.expand_dims(img_tensor, axis=0)
print(img_tensor.shape)
plt.imshow(img_tensor[0])
plt.show()


#-------------------------------------
#BUILD MODEL SEQUENTIALLY (LINEAR STACK)
#-------------------------------------
model = models.Sequential()

#BUILD LAYER ARRAYS FOR ANN
ACTIVATIONS=[]; LAYERS=[]
for i in range(0,N_HIDDEN):
    LAYERS.append(N_NODES)
    ACTIVATIONS.append(ACT_TYPE)

if method == 'CNN':
    # Test the number of layers is valid
    start = min(image_dim)
    is_valid = True
    for i in range(1, len(LAYERS)):
        start = floor(start / 2)
        start -= 2
        print(i)
        print(start)
        if start <= 0:
            is_valid = False
            start = i
    if not is_valid:
        print('Not valid -- Convolutional layers will reduce dimensions too much')
        del LAYERS[-1 * (len(LAYERS) - start)]
        del ACTIVATIONS[-1 * (len(LAYERS) - start)]
        print('Now using: ' + str(len(LAYERS)) + ' layers')
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_dim[0], image_dim[1], image_dim[2])))
#    model.add(layers.MaxPooling2D((2, 2)))
#    model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
#    model.add(layers.MaxPooling2D((2, 2)))
#    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    for i in range(1, len(LAYERS)):
        model.add(layers.MaxPooling2D(2,2))
        model.add(layers.Conv2D(N_NODES, (3, 3), activation=ACTIVATIONS[i], kernel_regularizer=regularizers.l2(L2_CONSTANT)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
elif method == 'DFF':
    model.add(layers.Dense(LAYERS[0], activation=ACTIVATIONS[0], input_shape=(image_dim[0], image_dim[1], image_dim[2]), kernel_regularizer=regularizers.l2(L2_CONSTANT)))
    model.add(layers.Flatten())
    for i in range(1,len(LAYERS)):
        model.add(layers.Dense(LAYERS[i], activation=ACTIVATIONS[i], kernel_regularizer=regularizers.l2(L2_CONSTANT)))
    #OUTPUT LAYER
    #model.add(layers.Dense(y.shape[1], activation=OUTPUT_ACTIVATION, kernel_regularizer=regularizers.l2(L2_CONSTANT)))
elif 'LOAD' in method:
    load_path = method[5:]
    print(load_path)
    model = models.load_model(load_path)
else:
    print("Error! Please specify 'CNN' or 'DFF' method for the NN")
    exit()
if 'LOAD' not in method:
    model.add(layers.Dense(image_dim[0] * image_dim[1] * image_dim[2], activation='sigmoid'))

model.summary()

if save is not None and save != '':
    model.save(save)

#DEBUGGING
NKEEP=train_num
batch_size=int(0.1*NKEEP)
#epochs=20
print("batch_size",batch_size)
rand_indices = np.random.permutation(train_images.shape[0])
train_images=train_images[rand_indices[0:NKEEP],:,:]
train_labels=train_labels[rand_indices[0:NKEEP]]
# exit()


#CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
tmp=train_labels[0]
train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)
test_labels = to_categorical(test_labels)
print(tmp, '-->',train_labels[0])
print("train_labels shape:", train_labels.shape)

#-------------------------------------
#COMPILE AND TRAIN MODEL
#-------------------------------------
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
if IS_AUGMENTED:
    from keras import ImageDataGenerator
    datagen = ImageDataGenerator()
    iterator = datagen.flow(train_images, train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2]))
    history = model.fit_generator(iterator, steps_per_epoch=floor(train_total / N_NODES), validation_data=(val_images, val_images.reshape(val_images.shape[0], val_images.shape[1] * val_images.shape[2])))
#model.compile(optimizer='rmsprop',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])
else:
    history = model.fit(train_images, train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2]), epochs=epochs, batch_size=batch_size, validation_data=(val_images, val_images.reshape(val_images.shape[0], val_images.shape[1] * val_images.shape[2])))


#-------------------------------------
#EVALUATE ON TEST DATA
#-------------------------------------
train_loss, train_acc = model.evaluate(train_images, train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2]), batch_size=batch_size)
val_loss, val_acc = model.evaluate(val_images, val_images.reshape(val_images.shape[0], val_images.shape[1] * val_images.shape[2]), batch_size=floor(validation_percentage*batch_size))
test_loss, test_acc = model.evaluate(test_images, test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2]), batch_size=test_images.shape[0])
print('train_acc:', train_acc)
print('val_acc: ', val_acc)
print('test_acc:', test_acc)

#BASIC PLOTTING
I_PLOT=True
#if(k==N_KFOLD-1): I_PLOT=True
if(I_PLOT):
    #LOSS
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig('loss_' + dataset + '_' + method + '.png')

    #METRICS
    metric = 'accuracy'
    plt.clf()
    MT = history.history[metric]
    MV = history.history['val_'+metric]
    plt.plot(epochs, MT, 'bo', label='Training '+metric)
    plt.plot(epochs, MV, 'b',  label='Validation '+metric)
    plt.title('Training and validation '+metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.show()
    plt.savefig('accuracy_' + dataset + '_' + method + '.png')




# Visualize covnet layers -- skip if not CNN!
if method != 'CNN':
    exit()


# Skipping this entirely. There seems to be an issue with keras.backend.gradients in TF versions 2+, as documented here: https://github.com/tensorflow/tensorflow/issues/33135 and here: https://stackoverflow.com/questions/66221788/tf-gradients-is-not-supported-when-eager-execution-is-enabled-use-tf-gradientta
# The below code was adapted from the Chollet book, chapter 5.4 (it must use an older version of TF, or I have the wrong book version!)
# exit called to cut off script execution early for a clean exit
exit()



# Define loss tensor
#from keras.applications import VGG16
from keras import backend as K
from tensorflow import GradientTape
#from tensorflow import compat
#model = VGG16(weights='imagenet',include_top=False)
layer_name = 'conv2d_1'
filter_index = 0
layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])

# Get gradient of loss wrt input
#compat.v1.disable_eager_execution()
with GradientTape() as gtape:

    grads = K.gradients(loss, model.input)[0]

    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5) # 1e-5 prevents division by 0

# Get numpy outputs from the inputs
    iterate = K.function([model.input], [loss, grads])
    loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

# Basic Stochastic Grad Descent
    input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

# Re-convert tensor into plotable image
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Generate the covnet visualization pattern
def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]
    return deprocess_image(img)

plt.imshow(generate_pattern('conv2d_1', 0))

# Further zoom out to see all filter response patterns in a specified layer!
layer_name = 'conv2d_1'
size = 64
margin = 5
results = np.zeros((8 * size+7* margin, 8 * size+7* margin, 3))
for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end,vertical_start: vertical_end, :] = filter_img
plt.figure(figsize=(20, 20))
plt.imshow(results)
