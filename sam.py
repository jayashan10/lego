# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import pandas as pd
from os.path import join
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
import glob
from IPython.display import Image, display
import matplotlib.pyplot as plt
from tensorflow import keras
#from tensorflow.python.keras.applications.ResNet50 import preprocess_input
#get_ipython().run_line_magic('matplotlib', 'inline')


# %%
img_dir='./question_1_dataset/train/'


# %%
total_imgs=len(glob.glob(img_dir+'/*.png'))
total_imgs


# %%
fnames=['{}.png'.format(i) for i in range(1,total_imgs+1)]
len(fnames)


# %%
img_paths=[join(img_dir,filename) for filename in fnames]


# %%
imgs=[load_img(img_path,target_size=(100,100)) for img_path in img_paths]


# %%
len(imgs)


# %%
img_array = np.array([img_to_array(img) for img in imgs])
img_array=img_array/255


# %%
img_array.shape


# %%
#display(Image(img_paths[4464]))


# %%
from tensorflow.keras.utils import to_categorical
x=img_array
df=pd.read_csv('./question_1_dataset/train.csv')
y_out=np.array(df['category'])
y=to_categorical(y_out)
y=y[:,1:]

x_train,x_cv,y_train,y_cv=train_test_split(x,y,test_size=0.2,random_state=4)


# %%
data_generator_with_aug = ImageDataGenerator( horizontal_flip = True,
                                              width_shift_range = 0.1,
                                              height_shift_range = 0.1,validation_split=0.2)
            
data_generator_no_aug = ImageDataGenerator()


# %%
model=Sequential()
model.add(Conv2D(20,kernel_size=3,activation='relu',input_shape=(100,100,3)))
model.add(Conv2D(20,kernel_size=3,activation='relu'))
model.add(Conv2D(20,kernel_size=3,activation='relu'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(16,activation='softmax'))


# %%
image_size=100
i=0
#train_generator = data_generator_with_aug.flow_from_directory(
 #       './question_1_dataset/train/',target_size=(150,150),batch_size=32,class_mode='categorical'
  #      )
train_generator = data_generator_with_aug.flow(
        x_train,y_train,batch_size=100
        )


# %%
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit_generator(
        train_generator,
        epochs = 10,
        steps_per_epoch=30,validation_data=(x_cv,y_cv))


# %%
yhat=model.predict(x_cv)
yhat=yhat.argmax(axis=1)+1
y_cv=y_cv.argmax(axis=1)+1


# %%
from sklearn.metrics import f1_score
f1_score(y_cv, yhat, average='weighted')


# %%
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x,y,batch_size=50,epochs=10,validation_split=0.2)


# %%

yhat=model.predict(x)
yhat=yhat.argmax(axis=1)+1

from sklearn.metrics import f1_score
f1_score(y_out, yhat, average='weighted')


# %%

img_dir_test='./question_1_dataset/test/'
total_imgs_test=len(glob.glob(img_dir_test+'/*.png'))
fnames_test=['{}.png'.format(i) for i in range(total_imgs+1,total_imgs+total_imgs_test+1)]
img_paths_test=[join(img_dir_test,filename) for filename in fnames_test]
imgs_test=[load_img(img_path_test,target_size=(100,100)) for img_path_test in img_paths_test]
img_array_test = np.array([img_to_array(img) for img in imgs_test])
img_array_test=img_array_test/255


# %%
x_test=img_array_test
yhat_test=model.predict(x_test)
yhat_test=yhat_test.argmax(axis=1)+1


# %%
y_final=np.hstack([yhat,yhat_test])
y_final.shape


# %%
import csv


# %%
with open("./question_1_dataset/Testfinal.csv","w") as file:
    writer=csv.writer(file)
    for i in range(total_imgs):
        writer.writerow([fnames[i],yhat[i]])
    for i in range(total_imgs_test):
        writer.writerow([fnames_test[i],yhat_test[i]])


# %%
df=pd.read_csv('./question_1_dataset/Testfinal.csv')


# %%
df


# %%
plt.show()

