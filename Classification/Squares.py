import tkinter
import cv2
import tensorflow
import numpy, random
import shutil, os, sys, pathlib
from functools import partial
import matplotlib.pyplot as plot
import sklearn
import time
import pickle


AUTOTUNE = tensorflow.data.experimental.AUTOTUNE

class Window():
    def __init__(self, num_pics_per_click, delete_old_pics):
        '''
        Variable delete_old_pics should be true only if changing users or changing the number of squares
        '''
        self.window = tkinter.Tk()
        self.window.attributes('-zoomed', True)
        self.build_main_frame()
        self.num_rows = 3
        self.num_cols = 3
        self.labels = [x for x in range(self.num_rows * self.num_cols)]
        self.colors = ['red', 'blue', 'brown4', 'green', 'orange', 'purple', 'pink', 'cyan', 'orange2']
        if delete_old_pics:
            self.clean_imgs()
        self.num_pict_dict = self.update_num_pictures()
        self.collect_data(num_pics_per_click)
        self.window.mainloop()

    def build_main_frame(self):
        '''
        Reset window to have only one main empty frame
        '''
        try:
            self.main_frame.destroy()
        except:
            True
        self.main_frame = tkinter.Frame(master=self.window, bg='white')
        self.main_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

    def collect_data(self, num_pics_per_click):
        '''
        Main stage for taking pictures while looking at different squares
        '''
        index = 0
        # Make frames into variable so we can use later
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                curr_color = self.colors[index]
                frame = tkinter.Frame(master=self.main_frame, bg=curr_color)
                frame.place(relx=j/self.num_cols, rely=i/self.num_rows, relwidth=1/self.num_cols, relheight=1/self.num_rows)
                num_pics_widget = tkinter.Label(frame, text=f'Num Pics: {self.num_pict_dict[index]}')
                num_pics_widget.place(relx=0.5, rely=0.4, anchor='center')
                button = tkinter.Button(frame, text=f'Square {index}', command=partial(Window.take_picture, self, num_pics_widget, index, num_pics_per_click))
                button.place(relx=0.5, rely=0.5, anchor='center')
                index += 1
        train_button = tkinter.Button(self.main_frame, text='Begin Training', command=self.start_training)
        train_button.place(relx=0.5, rely=0, anchor='n')
    def start_training(self):
        '''
        Start training the model with the images in the Images/ folder
        '''
        # Display training progress to window
        self.build_main_frame()
        print('Counting Images...')
        total_num_img = 0
        for label in self.labels:
            for path in pathlib.Path(f"Images/Square{label}").iterdir():
                total_num_img += 1
        random_indices = random.sample(range(total_num_img), total_num_img)
        print('Loading Images...')
        random_index = 0
        self.IMG_SIZE = 200
        images = [-1 for num in range(total_num_img)]
        labels = [-1 for num in range(total_num_img)]
        for label in self.labels:
            for path in pathlib.Path(f"Images/Square{label}").iterdir():
                new_image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                new_image = cv2.resize(new_image, (self.IMG_SIZE, self.IMG_SIZE))
                
                images[random_indices[random_index]] = new_image
                labels[random_indices[random_index]] = label 
                random_index += 1
        X = numpy.array(images).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        X = X / 255
        y = numpy.array(labels)
        print('Building Model...')       
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        model = tensorflow.keras.models.Sequential()

        model.add(tensorflow.keras.layers.Conv2D(30, (3, 3), input_shape=(self.IMG_SIZE, self.IMG_SIZE, 1)))
        model.add(tensorflow.keras.layers.Activation('relu'))
        model.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        
        model.add(tensorflow.keras.layers.Conv2D(30, (3, 3), input_shape=(self.IMG_SIZE, self.IMG_SIZE, 1)))
        model.add(tensorflow.keras.layers.Activation('relu'))
        model.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tensorflow.keras.layers.Flatten())
        model.add(tensorflow.keras.layers.Dense(64))
        model.add(tensorflow.keras.layers.Dense(len(self.labels)))
        model.add(tensorflow.keras.layers.Activation('sigmoid'))
        model.summary()
        model.compile(loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        optimizer='adam',
                        metrics=['accuracy'])
        print('Training Model...')
        model.fit(x=X, y=y, batch_size=32, epochs=6, validation_split=0.05)
        self.model = model
        print('Model Built!')
        train_button = tkinter.Button(self.main_frame, text='Eye Tracking Phase', command=self.start_tracking)
        train_button.place(relx=0.5, rely=0.5, anchor='center')

    def start_tracking(self):
        self.build_main_frame()
        index = 0
        frames = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                frame = tkinter.Frame(master=self.main_frame, bg=self.colors[index])
                frame.place(relx=j/self.num_cols, rely=i/self.num_rows, relwidth=1/self.num_cols, relheight=1/self.num_rows)
                frames.append(frame)
                square_label = tkinter.Label(frame, text=f'Square {index}')
                square_label.place(relx=0.5, rely=0.5, anchor='center')
                index += 1
        self.window.update()
        video = cv2.VideoCapture(0)
        index = 0
        while True:
            # Fix last frame that was looked at
            frames[index].config(bg=self.colors[index])
            cap, frame = video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (self.IMG_SIZE, self.IMG_SIZE))
            frame = (numpy.expand_dims(frame,0))
            frame = frame.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
            frame = frame / 255
            # Predict current frame being looked at and update
            predictions = self.model.predict(frame)
            print(f'\n\n{predictions}\n\n')
            index = numpy.argmax(predictions)
            frames[index].config(bg='black')
            self.window.update()

    def take_picture(self, num_pics_widget, square_index, num_pictures):
        '''
        Take num_pictures while assumed looking at square square_index
        '''
        # Open camera
        video = cv2.VideoCapture(0)
        # Take num_pictures
        for i in range(num_pictures):
            # Take picture
            cap, frame = video.read()
            # Save to appropriate location (e.g. Images/Square1/img13.jpg) using number of images for that square
            cv2.imwrite(f"Images/Square{square_index}/img{self.num_pict_dict[square_index]}.jpg",frame)
            # Update the number of pictures for that square
            self.num_pict_dict[square_index] += 1
            num_pics_widget.config(text=f'Num Pics: {self.num_pict_dict[square_index]}')
            self.window.update()
        # Close camera
        video.release()
    
    def clean_imgs(self):
        '''
        Only use if removing old images, like if changing users or changing the number of squares
        '''
        # Delete previous directory of images
        try:
            shutil.rmtree('Images/')
        except:
            print('Faild to remove Images folder')
        # Make directory images
        os.mkdir('Images')
        # Make directories for different squares
        for label in self.labels:
            os.mkdir(f'Images/Square{label}')

    def update_num_pictures(self):
        '''
        Begin program by counting the number of pictures already contained for that square
        '''
        num_pic_dict = {}
        total_num_img = 0
        for label in self.labels:
            for path in pathlib.Path(f"Images/Square{label}").iterdir():
                total_num_img += 1
            num_pic_dict[label] = total_num_img
            total_num_img = 0
        return num_pic_dict

    '''
    class RedirectPrint(object):
        def __init__(self, window, widget):
            self.widget = widget
            self.window = window
            self.buff = ""
        def write(self, string):
            self.widget.config(state='normal')
            self.widget.insert('end', string)
            self.widget.config(state='disabled')
            self.window.update()
        def buffer(self, string):
            self.buffer += string
        def flush(self):
            self.write(self.buff)
            self.buff  = ""
    '''
main_window = Window(200, False)
