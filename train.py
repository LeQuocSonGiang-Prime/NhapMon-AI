import numpy as np
import keras
import keras.backend as k
from keras.layers import Conv2D,MaxPooling2D,SpatialDropout2D,Flatten,Dropout,Dense
from keras.models import Sequential,load_model
from keras.optimizers import adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


#tao ra mot model
model=Sequential() 
 #(fillter) - kernel size - 
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D() ) #
#trích xuất đặt trưng từ ảnh to sang ảnh nhỏ
#giảm parammeter 
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten()) #dung de làm phẳng
model.add(Dense(100,activation='relu')) # tại sao 100/1 relu -sigmid
model.add(Dense(1,activation='sigmoid')) #kich hoạt ra  0 1

#Hoan thanh model voi ham loss ca ham optimizer
#huấn luyện - 
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#optimizer: hàm điều chỉnh learning rate và trọng số #adam 1/n lớp
#loss: tính độ mất mác của con AI(train_test)
#metrics: độ đo lường mất mác
#xem so do model
model.summary()
train_datagen = ImageDataGenerator( #đảo chiều
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory( 
        'train',
        target_size=(150,150), 
        batch_size=16 ,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'test',
        target_size=(150,150),
        batch_size=16,
        class_mode='binary')
model_saved=model.fit_generator( #chạy model
        training_set,
        epochs=10,
        validation_data=test_set,
        )
model.save('model_face_mask.h5',model_saved)
