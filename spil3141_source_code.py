from PIL import Image
import numpy as np 
import h5py 
import os 
import pandas 
import matplotlib.pyplot as plt
import tensorflow as tf

train_feature_folder_dir = "data/train_images"
train_label_folder_dir = "data/train_labels.csv"
valid_feature_folder_dir = "data/train_images"
valid_label_folder_dir = "data/train_labels.csv"
output_folder_dir = "data/pre_processed_validate_images"

LIST_OF_SPECIES  = list(pandas.read_csv("round4_classes.csv")["scientific_name"])
height = 100 
weight = 100
##############################################################################
# Total Number of Samples ( 245185 )
# 1. Load in the images from the image folder using a python generator
# 2. Resize images to a fixed size (100x100)
# 3. convert python generator to tensorflow.data
# 3. Feature Scaling ( Normalization or Standardization) dataset using tf.data map function
# 4. Add Labels ( 783 )
##############################################################################
#python generator
def train_gen(feature_dir,label_dir):
    label_df = pandas.read_csv(str(label_dir,"utf-8"))
    for image in os.listdir(feature_dir):
        x = np.array((Image.open(os.path.join(feature_dir,image)).convert('RGB')).resize((height,weight)),dtype="float32")
        y = np.asarray(LIST_OF_SPECIES.index((label_df.values[np.where(label_df.values[:,2] == str(image[:-4],"utf-8"))])[0][0]))
        yield (x,y)
def valid_gen(feature_dir,label_dir):
    label_df = pandas.read_csv(str(label_dir,"utf-8"))
    for image in os.listdir(feature_dir):
        x = np.array((Image.open(os.path.join(feature_dir,image)).convert('RGB')).resize((height,weight)),dtype="float32")
        y = np.asarray(LIST_OF_SPECIES.index((label_df.values[np.where(label_df.values[:,2] == str(image[:-4],"utf-8"))])[0][0]))
        yield (x,y)

# Convert python gen to tf.data 
train_dataset = tf.data.Dataset.from_generator(train_gen,args=[train_feature_folder_dir,train_label_folder_dir],
                                          output_types=('float32',"int32"),
                                          output_shapes=(tf.TensorShape((height,weight,3)),tf.TensorShape(()))
                                          )

valid_dataset = tf.data.Dataset.from_generator(valid_gen,args=[valid_feature_folder_dir,valid_label_folder_dir],
                                          output_types=('float32',"int32"),
                                          output_shapes=(tf.TensorShape((height,weight,3)),tf.TensorShape(()))
                                          )

#Normalization 
# Pre-processing 
def norm(x,_min,_max):
   return (x - _min)/(_max - _min)
def normalize_samples(feature,label):
   X = norm(feature,0,255)
   Y = label
   return X,Y
#Apply Feature Scaling
train_dataset_norm = train_dataset.map(normalize_samples)
valid_dataset_norm = valid_dataset.map(normalize_samples)

train_dataset_norm = train_dataset_norm.batch(100)
valid_dataset_norm = valid_dataset_norm.batch(100)


# for i in train_dataset_norm:
#     print(i[0].shape, " : " , i[1])
#     a = i[0]
#     break

# count = 0
# for i in features:
#     print(i[1])
#     if count == 10:
#         break
#     count += 1


##############################################################################
# Model
##############################################################################

def create_model_spil_cnn():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3),padding = "same", activation='relu', input_shape= (height,weight,3)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3),padding = "same", activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.45))
    model.add(tf.keras.layers.Conv2D(64, (3, 3),padding = "same", activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3),padding = "same", activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.45))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(222, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.45))
    model.add(tf.keras.layers.Dense(len(LIST_OF_SPECIES), activation='softmax')) # softmax activation function goes with categorical_crossentropy/ predicts prob
    # model.add(tf.keras.layers.Dense(1, activation = "sigmoid")) # if i use a sigmoid activation function i must pass a loss function of binary_crossentropy
    return model
def create_InceptionResNetV2() -> tf.keras.models.Model:
    base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False,
                                                                             input_shape= (height,weight,3),
                                                                             weights="imagenet",
                                                                             input_tensor= tf.keras.layers.Input(shape= (height,weight,3)),
                                                                             pooling=None,
                                                                             classes=len(LIST_OF_SPECIES)
                                                                             )
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    predictions = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    return model

model = create_model_spil_cnn()
model.summary()
#print("Input Shapes: ")
#for i in model.layers:
#    print(i.name,"\t" ,i.input_shape)

##############################################################################
# Define the checkpoint directory to store the checkpoints
##############################################################################
checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# Function for decaying the learning rate.
EPOCHS = 10
def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif epoch >= 3 and epoch < 7:
    return 1e-4
  else:
    return 1e-5

# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]

##############################################################################
# Compiling and Optimizers 
##############################################################################
# model.compile(optimizer= tf.keras.optimizers.SGD(learning_rate=0.01,
#                                                          momentum=0.0,
#                                                          nesterov=False),
#                       loss="categorical_crossentropy",
#                       metrics = ["accuracy"])
# model.compile(optimizer= tf.keras.optimizers.RMSprop(lr=0.001,
#                                                     rho=0.9,
#                                                     epsilon=None,
#                                                     decay=0.0),
#               loss='categorical_crossentropy',
#               metrics = ["accuracy"])

# model.compile(optimizer= tf.keras.optimizers.RMSprop(lr=0.045,
#                                                     rho=0.9,
#                                                     epsilon=0.9,
#                                                     decay=1.0),
#               loss='categorical_crossentropy',
#               metrics = ["accuracy"])
model.compile(optimizer= tf.keras.optimizers.Adam(lr=0.001,
                                                  beta_1=0.9,
                                                  beta_2=0.999,
                                                  epsilon=None,
                                                  decay=0.0,
                                                  amsgrad=False),
              loss='categorical_crossentropy',
#                loss = "binary_crossentropy",
              metrics=['accuracy'])

##############################################################################
# Training the Model
##############################################################################
# Fit with tf.data """
history = model.fit(valid_dataset_norm,
                    epochs=EPOCHS,
                    callbacks= callbacks,
                    verbose=1
                   )
"""### Saving history ###"""
hist_df = pandas.DataFrame(history.history)
# or save to csv:
hist_csv_file = 'history-extra.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)




