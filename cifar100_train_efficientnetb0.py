import tensorflow as tf
from tensorflow.keras.datasets import cifar100, cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
import albumentations as albu

# Load CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
channel_mean = np.mean(x_train/255, (0,1,2))
channel_var = np.var(x_train/255, (0,1,2))

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for keras'
    def __init__(self, images , labels = None, mode = 'fit', batch_size = 32,
                 dim = (32,32), channels = 3, n_classes = 100,
                 shuffle = True, augment = False):
        self.images = images
        self.labels = labels
        self.mode = mode
        self.batch_size = batch_size
        self.dim = dim
        self.channels = channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        
        self.on_epoch_end()
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.images.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.images) / self.batch_size))
        
    def __getitem__(self, index):
        'Generate one batch of data'
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # =========================================================== #
        # Generate mini-batch of X
        # =========================================================== #
        X = np.empty((self.batch_size, *self.dim, self.channels))
        for i, ID in enumerate(batch_indexes):
            # Generate a preprocessed image
            img = self.images[ID]
            X[i] = img
            
        
        # =========================================================== #
        # Generate mini-batch of y
        # =========================================================== #
        if self.mode == 'fit':
            y = self.labels[batch_indexes][:,0]
            # y = to_categorical(y, 100)
            '''
            y = np.zeros((self.batch_size, self.n_classes), dtype = np.uint8)
            for i, ID in enumerate(batch_indexes):
                # one hot encoded label
                y[i, self.labels[ID]] = 1
            '''
            # Augmentation should only be implemented in the training part.
            if self.augment == True:
                X = self.__augment_batch(X)                
            
            return X,y
        
        elif self.mode == 'predict':
            return X       
        
        else:
            raise AttributeError('The mode parameters should be set to "fit" or "predict"')
            
    def __random_transform(self, img):
        composition = albu.Compose([albu.HorizontalFlip(p = 0.5),
                                    albu.VerticalFlip(p = 0.5),
                                    albu.GridDistortion(p = 0.2),
                                    albu.ElasticTransform(p = 0.2)])
        
        return composition(image = img)['image']
        
    
    def __augment_batch(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i] = self.__random_transform(img_batch[i])
            
        return img_batch
    
train_generator = DataGenerator(x_train, y_train, augment = True,batch_size=256)
valid_generator = DataGenerator(x_test, y_test, augment = False,batch_size=256)


# tf.keras.layers.Resizing()
def get_eff_net(output_dimension=100):    
    inputs = tf.keras.Input((32,32,3))
    x = tf.keras.layers.Resizing(224,224)(inputs)
    # data_augmentation = tf.keras.Sequential([
    #   tf.keras.layers.RandomFlip("horizontal"),
    #   tf.keras.layers.RandomRotation(0.2),
    # ])
    # x = data_augmentation(x)
    x = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False,weights='imagenet')(x)
    # x = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False,weights='imagenet')(x)
    # x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    # x = tf.keras.layers.Dense(1000,activation='relu')(x)
    x = tf.keras.layers.Dense(output_dimension)(x)
    
    return Model(inputs=inputs, outputs=x)

# model = get_eff_net()
# model(tf.ones((1,32,32,3))).shape



# model = Model(inputs=inputs, outputs=outputs)
strategy = tf.distribute.MirroredStrategy()
# Variable created inside scope:
with strategy.scope():
    # model = tf.keras.models.load_model('./cifar100_cnn.h5')
    # model = get_resnet(100,[1,1,4,2,5],channel_mean,channel_var)
    model = get_eff_net()
# Compile the model
model.summary()

optimizer = Adam(1e-4)
# lr_schedule = tf.keras.optimizers.schedules.CosineDecay(1e-3, 1000)
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
# model.summary()
# Train the model
# model.fit(x_train, y_train, batch_size=256, epochs=50, validation_data=(x_test, y_test))
hist = model.fit(train_generator,validation_data = valid_generator, 
                           epochs = 50)# Save the model to a .h5 file
# model.save('./cifar100_cnn.h5')
