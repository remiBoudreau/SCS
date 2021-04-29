
'''
Import dependencies
'''
import os
import warnings
import time
import numpy as np
import random
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

from keras.initializers import TruncatedNormal
from keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Flatten, Input, LeakyReLU, Reshape
from keras.models import Model
from keras.optimizers import Adam
# 
warnings.filterwarnings("ignore")

'''
Hyperparameters
'''
IMAGE_SIZE = 64
NOISE_SIZE = 100
LR_D = 0.00004
LR_G = 0.0004
BATCH_SIZE = 64
EPOCHS = 50
BETA1 = 0.5
WEIGHT_INIT_STDDEV = 0.02
EPSILON = 0.00005
SAMPLES_TO_SHOW = 5

'''
GAN
'''
def generator_model(z=(NOISE_SIZE,)):
    # Input Layer [100]
    input_layer=Input(z)
    # Fully connected layer [100 -> 4x4x1024]
    fully_connected = Dense(4*4*1024) (input_layer)
    fully_connected = Reshape((4, 4, 1024)) (fully_connected)
    # Default alpha of keras is 0.3;
    # Default alpha of tf is 0.2;
    # For consistency with original code, alpha defined as 0.2
    fully_connected = LeakyReLU(alpha=0.2) (fully_connected)
    
    # Convolutional Layer 1 [4x4x1024 -> 8x8x512]
    trans_conv1 = Conv2DTranspose(filters=512,
    			   			      kernel_size=[5,5],
	    					      strides=[2,2],
    						      padding="same",
    						      kernel_initializer=TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)) (fully_connected)
    batch_trans_conv1 = BatchNormalization(epsilon=EPSILON) (trans_conv1)
    trans_conv1_out = LeakyReLU(alpha=0.2) (batch_trans_conv1)

    # Convolutional Layer 2 [8x8x512 -> 16x16x256]
    trans_conv2 = Conv2DTranspose(filters=256,
    						  	  kernel_size=[5,5],
	    					  	  strides=[2,2],
    						  	  padding="same",
    						  	  kernel_initializer=TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)) (trans_conv1_out)
    batch_trans_conv2 = BatchNormalization(epsilon=EPSILON) (trans_conv2)
    trans_conv2_out = LeakyReLU(alpha=0.2) (batch_trans_conv2)

    # Convolutional Layer 3 [16x16x256 -> 32x32x128]
    trans_conv3 = Conv2DTranspose(filters=128,
    						  	  kernel_size=[5,5],
	    					  	  strides=[2,2],
    						  	  padding="same",
    						  	  kernel_initializer=TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)) (trans_conv2_out)
    batch_trans_conv3 = BatchNormalization(epsilon=EPSILON) (trans_conv3)
    trans_conv3_out = LeakyReLU(alpha=0.2) (batch_trans_conv3)

    # Convolutional Layer 4 [32x32x128 -> 64x64x64]
    trans_conv4 = Conv2DTranspose(filters=64,
    						  	  kernel_size=[5,5],
	    					  	  strides=[2,2],
    						  	  padding="same",
    						  	  kernel_initializer=TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)) (trans_conv3_out)
    batch_trans_conv4 = BatchNormalization(epsilon=EPSILON) (trans_conv4)
    trans_conv4_out = LeakyReLU(alpha=0.2) (batch_trans_conv4)

    # Output layer [64x64x64 -> 64x64x3]
    logits = Conv2DTranspose(filters=3,
    						 kernel_size=[5,5],
	    					 strides=[1,1],
    						 padding="same",
    						 kernel_initializer=TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)) (trans_conv4_out)
    out = Activation('tanh') (logits)
    
    #
    model = Model(inputs=input_layer, outputs=out)
    model.summary()
    return model

def discriminator_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)):
    # Input Layer [64x64x3]
    input_layer = Input(input_shape)
    
	# Convolutional Layer 1 [64x64x3 -> 32x32x64]
    conv1 = Conv2D(filters=64,
				   kernel_size=[5,5],
				   strides=[2,2],
				   padding="same",
    			   kernel_initializer=TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)) (input_layer)
    batch_norm1 = BatchNormalization(epsilon=EPSILON) (conv1)
    conv1_out = LeakyReLU(alpha=0.2) (batch_norm1)

	# Convolutional Layer 2 [32x32x64 -> 16x16x128]
    conv2 = Conv2D(filters=128,
				   kernel_size=[5,5],
				   strides=[2,2],
				   padding="same",
    			   kernel_initializer=TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)) (conv1_out)
    batch_norm2 = BatchNormalization(epsilon=EPSILON) (conv2)
    conv2_out = LeakyReLU(alpha=0.2) (batch_norm2)

	# Convolutional Layer 3 [16x16x128 -> 8x8x256]
    conv3 = Conv2D(filters=256,
				   kernel_size=[5,5],
				   strides=[2,2],
				   padding="same",
    			   kernel_initializer=TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)) (conv2_out)
    batch_norm3 = BatchNormalization(epsilon=EPSILON) (conv3)
    conv3_out = LeakyReLU(alpha=0.2) (batch_norm3)

	# Convolutional Layer 4 [8x8x256 -> 8x8x512]
    conv4 = Conv2D(filters=512,
				   kernel_size=[5,5],
				   strides=[1,1],
				   padding="same",
    			   kernel_initializer=TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)) (conv3_out)
    batch_norm4 = BatchNormalization(epsilon=EPSILON) (conv4)
    conv4_out = LeakyReLU(alpha=0.2) (batch_norm4)

	# Convolutional Layer 5 [8x8x512 -> 4x4x1024]
    conv5 = Conv2D(filters=1024,
				   kernel_size=[5,5],
				   strides=[2,2],
				   padding="same",
    			   kernel_initializer=TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)) (conv4_out)
    batch_norm5 = BatchNormalization(epsilon=EPSILON) (conv5)
    conv5_out = LeakyReLU(alpha=0.2) (batch_norm5)
    
    # Flattened Convolutional Filter Output Layer [4x4x1024 -> 16384] ###?#?#?#?#??#
    flatten = Flatten() (conv5_out)
    
    # Fully connected layer [16384 -> 1]
    out = Dense(1, activation='sigmoid') (flatten)
    
    # ?????? 
    model = Model(inputs=input_layer, outputs=out)
    model.summary()
    return model

# Discriminator
discriminator = discriminator_model((IMAGE_SIZE, IMAGE_SIZE, 3))
discriminator.compile(loss='binary_crossentropy',optimizer=Adam(lr=LR_D, beta_1=BETA1),metrics=['accuracy'])
discriminator.trainable = False

# Generator
generator = generator_model((NOISE_SIZE,))

# GAN
gan_input = Input(shape=(NOISE_SIZE,))
x = generator(gan_input)
gan_out = discriminator(x)
gan = Model(gan_input, gan_out)
gan.summary()

gan.compile(loss='binary_crossentropy',optimizer=Adam(lr=LR_G, beta_1=BETA1))

'''
PLOT SAMPLE IMAGES
'''
def show_samples(sample_images, name, epoch):
    figure, axes = plt.subplots(1, len(sample_images), figsize = (IMAGE_SIZE, IMAGE_SIZE))
    for index, axis in enumerate(axes):
        axis.axis('off')
        image_array = sample_images[index]
        axis.imshow(image_array)
        image = Image.fromarray(image_array)
        image.save(name+"_"+str(epoch)+"_"+str(index)+".png") 
    plt.savefig(name+"_"+str(epoch)+".png", bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
    
'''

'''
def test(input_z, epoch, OUTPUT_DIR):
    samples = generator.predict(input_z[:SAMPLES_TO_SHOW])
    sample_images = [((sample + 1.0) * 127.5).astype(np.uint8) for sample in samples]
    show_samples(sample_images, OUTPUT_DIR + "samples", epoch)

def summarize_epoch(d_losses, g_losses , data_shape, epoch, duration, input_z, OUTPUT_DIR):
    minibatch_size = int(data_shape[0]//BATCH_SIZE)
    print("Epoch {}/{}".format(epoch, EPOCHS),
          "\nDuration: {:.5f}".format(duration),
          "\nD Loss: {:.5f}".format(np.mean(d_losses[-minibatch_size:])),
          "\nG Loss: {:.5f}".format(np.mean(g_losses[-minibatch_size:])))
    fig, ax = plt.subplots()
    plt.plot(d_losses, label='Discriminator', alpha=0.6)
    plt.plot(g_losses, label='Generator', alpha=0.6)
    plt.title("Losses")
    plt.legend()
    plt.savefig(OUTPUT_DIR + "losses_" + str(epoch) + ".png")
    plt.show()
    plt.close()
    test(input_z, epoch, OUTPUT_DIR)
    
def get_batches(data):
    batches = []
    for i in range(int(data.shape[0]//BATCH_SIZE)):
        batch = data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        augmented_images = []
        for img in batch:
            image = Image.fromarray(img)
            if random.choice([True, False]):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            augmented_images.append(np.asarray(image))
        batch = np.asarray(augmented_images)
        normalized_batch = (batch / 127.5) - 1.0
        batches.append(normalized_batch)
    return np.array(batches)

# Training
def train(): 
    '''
LOAD DATA
'''
    INPUT_DATA_DIR = "./simpsons-faces/cropped/" # Path to the folder with input images. For more info check simspons_dataset.txt
    OUTPUT_DIR = './model/'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Training
    # Import Data
    exclude_img = ["9746","9731","9717","9684","9637","9641","9642","9584","9541","9535",
    "9250","9251","9252","9043","8593","8584","8052","8051","8008","7957",
    "7958""7761","7762","9510","9307","4848","4791","4785","4465","2709",
    "7724","7715","7309","7064","7011","6961","6962","6963","6960","6949",
    "6662","6496","6409","6411","6406","6407","6170","6171","6172","5617",
    "4363","4232","4086","4047","3894","3889","3493","3393","3362","2780",
    "2710","2707","2708","2711","2712","2309","2056","1943","1760","1743",
    "1702","1281","1272","772","736","737","691","684","314","242","191"]
    exclude_img = [s + ".png" for s in exclude_img]
    input_images = np.asarray([np.asarray(Image.open(file).resize((IMAGE_SIZE, IMAGE_SIZE))) for file in glob(INPUT_DATA_DIR + '*') if file not in exclude_img])
    print ("Image Samples\nInput: " + str(input_images.shape))
    np.random.shuffle(input_images)

    print("Training Starts!")
    # Load weights from last completed epoch if they exists
    discriminator_weights = glob('discriminator_simpson_weights.h5_*')
    gan_weights = glob('generator_simpson_weights.h5_*')
    if discriminator_weights:
        discriminator_weights.sort()
        discriminator.load_model(discriminator_weights[-1])
    if gan_weights:
        gan_weights.sort()
        gan.load_model(gan_weights[-1])
    
    d_losses = []
    g_losses = []
    cum_d_loss = 0
    cum_g_loss = 0

    #EPOCHS = 50
    for epoch in range(EPOCHS):
        epoch += 1
        start_time = time.time()
        for batch_images in get_batches(input_images):
            noise_data = np.random.normal(0, 1, size=(BATCH_SIZE, NOISE_SIZE))

            # We use same labels for generated images as in the real training batch
            generated_images = generator.predict(noise_data)

            # Prepare real target labels
            noise_prop = 0.05 # Randomly flip 5% of targets
            real_labels = np.zeros((BATCH_SIZE, 1)) + np.random.uniform(low=0.0, high=0.1, size=(BATCH_SIZE, 1))
            flipped_idx = np.random.choice(np.arange(len(real_labels)), size=int(noise_prop*len(real_labels)))
            real_labels[flipped_idx] = 1 - real_labels[flipped_idx]
            
            # Train discriminator on real data
            d_loss_real = discriminator.train_on_batch(batch_images, real_labels)
    
            # Prepare labels for generated data
            fake_labels = np.ones((BATCH_SIZE, 1)) - np.random.uniform(low=0.0, high=0.1, size=(BATCH_SIZE, 1))
            flipped_idx = np.random.choice(np.arange(len(fake_labels)), size=int(noise_prop*len(fake_labels)))
            fake_labels[flipped_idx] = 1 - fake_labels[flipped_idx]
            
            # Train discriminator on generated data
            d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            cum_d_loss += d_loss
            d_losses.append(d_loss[0])
            
            # Train generator
            noise_data = np.random.normal(0, 1, size=(BATCH_SIZE, NOISE_SIZE))
            g_loss = gan.train_on_batch(noise_data, np.zeros((BATCH_SIZE, 1)))
            cum_g_loss += g_loss
            g_losses.append(g_loss)
            
        if epoch > 0 and epoch % 5 == 0 :
            print("saving model")
            discriminator.save_weights(OUTPUT_DIR + "discriminator_simpson_weights.h5_" + str(epoch))
            gan.save_weights(OUTPUT_DIR + "gan_simpson_model.h5_" + str(epoch))
            summarize_epoch(d_losses, g_losses, input_images.shape, epoch, time.time()-start_time, noise_data, OUTPUT_DIR)

train()
