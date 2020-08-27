import tkinter, time, os, shutil, pathlib, cv2, tensorflow, numpy, random
from functools import partial
import matplotlib.pyplot as plot

'''
Run this program to see the results of using regression and moving the cursor 
along the screen 
'''

class Window():
    def __init__(self, clean_cols, clean_rows):
        '''
        Initialize all configuration variables here, which determine accuracy of model
        and how fast the model is trained.
        If you have a configured GPU for tensorflow, you can probably increase
        the img height and width here. OpenCV and the program will handle the rest,
        just don't go over the maximimum definition of your camera.
        With 100x100, it can still interestingly change the location of the horizontal bar
        with head swings.

        Switch clean_cols or clean_rows to true if you want to remove the training data from
        a previous run

        '''
        self.window = tkinter.Tk()
        self.window.attributes('-zoomed', True)
        # Set up constants
        self.num_repeats = 10
        self.time_btw_pics = 0.05
        self.num_cols = 100
        self.num_rows = 30
        self.BATCH_SIZE = 10
        # 480 x 640 is the maximum for my computer
        self.IMG_WIDTH = 100
        self.IMG_HEIGHT = 100
        # See if images need to be cleaned
        if clean_cols:
            self.clean_imgs('Columns')
        if clean_rows:    
            self.clean_imgs('Rows')
        # Count number of pictures in each folder
        self.col_dict = self.count_pics('Columns')
        self.row_dict = self.count_pics('Rows')
        self.collect_data()
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
    
    def collect_data(self):
        '''
        Landing page for application here
        '''
        self.build_main_frame()
        self.main_frame.config(bg='peach puff')
        # Column Data Info
        col_data_frame = tkinter.Frame(self.main_frame, bg='cyan3')
        col_data_frame.place(relx=0.5, rely=0.3, relheight=0.3, relwidth=0.9, anchor='center')
        col_data_info = tkinter.Label(col_data_frame, text=f'Num Pics Per Column: {self.col_dict[0]}', bg='cyan3')
        col_data_info.place(relx=0.5, rely=0.3, anchor='center')
        col_data_btn = tkinter.Button(col_data_frame, text='Take Column Data', command=partial(Window.collect_col_data, self))
        col_data_btn.place(relx=0.5, rely=0.5, anchor='center')
        # Row Data Info
        row_data_frame = tkinter.Frame(self.main_frame, bg='sienna2')
        row_data_frame.place(relx=0.5, rely=0.65, relheight=0.3, relwidth=0.9, anchor='center')
        row_data_info = tkinter.Label(row_data_frame, text=f'Num Pics Per Row: {self.row_dict[0]}', bg='sienna2')
        row_data_info.place(relx=0.5, rely=0.3, anchor='center')
        row_data_btn = tkinter.Button(row_data_frame, text='Take Row Data', command=partial(Window.collect_row_data, self))
        row_data_btn.place(relx=0.5, rely=0.5, anchor='center')
        # Start training button
        row_data_frame = tkinter.Frame(self.main_frame, bg='DarkOliveGreen4')
        row_data_frame.place(relx=0.5, rely=0.9, relheight=0.1, relwidth=0.9, anchor='center')
        train_btn = tkinter.Button(self.main_frame, text='Start Training', command=partial(Window.train_big_dataset, self))
        train_btn.place(relx=0.5, rely=0.9, anchor='center')

    def collect_col_data(self):
        '''
        Moves bar accross column, and user looks at bar to collect data

        DONT GIVE ANY BAD DATA
        '''
        self.build_main_frame()
        # Take pictures for columns
        nc = self.num_cols
        new_frame = tkinter.Frame(self.main_frame, bg='black')
        video = cv2.VideoCapture(0)
        for r in range(self.num_repeats):
            for i in range(nc):
                # Move column
                new_frame.place(relx=i/nc, rely=0, relwidth=2/nc, relheight=1)
                self.window.update()
                # Sleep to allow user to look
                time.sleep(self.time_btw_pics)
                # Take picture of user
                self.take_picture(video, 'Columns', i)
            for i in range(nc):
                # Move column
                new_frame.place(relx=(nc - (i+1))/nc, rely=0, relwidth=2/nc, relheight=1)
                self.window.update()
                # Sleep to allow user to look
                time.sleep(self.time_btw_pics)
                # Take picture of user
                self.take_picture(video, 'Columns', nc - (i+1))
        self.collect_data()

    def collect_row_data(self):
        '''
        Moves bar up and down for row data
        '''
        self.build_main_frame()
        # Take pictures for rows
        nr = self.num_rows
        new_frame = tkinter.Frame(self.main_frame, bg='black')
        video = cv2.VideoCapture(0)
        for r in range(self.num_repeats):
            for i in range(nr):
                # Move row
                new_frame.place(relx=0, rely=i/nr, relwidth=1, relheight=2/nr)
                self.window.update()
                # Sleep to allow user to look
                time.sleep(self.time_btw_pics)
                # Take picture of user
                self.take_picture(video, 'Rows', i)
            for i in range(nr):
                # Move row
                new_frame.place(relx=0, rely=(nr - (i+1))/nr, relwidth=1, relheight=2/nr)
                self.window.update()
                # Sleep to allow user to look
                time.sleep(self.time_btw_pics)
                # Take picture of user
                self.take_picture(video, 'Rows', nr - (i+1))
        self.collect_data()
    
    def take_picture(self, video, folder, index):
        '''
        Take picture and save it to the folder
        folder: Either Columns or Rows
        index:  Column / row number (b/c finite number of cols or rows for data collection)
        '''
        num_pics = 0
        if folder == 'Columns':
            num_pics = self.col_dict[index]
            self.col_dict[index] += 1
        elif folder == 'Rows':
            num_pics = self.row_dict[index]
            self.row_dict[index] += 1
        cap, pic = video.read()
        cv2.imwrite(f'{folder}/index{index}/img{num_pics}.jpg', pic)
        
    def train_big_dataset(self):
        ''' Define functions for loading data '''
        def get_label(file_path):
            '''
            file path is tensor of type string
                tensorflow keeps things as tensors bc they are how they handle the equations
                need to keep as tensor and use tensor functions to manipulate and get
                the number from the file path needed for regression
            '''
            # convert the path to a list of path components
            parts = tensorflow.strings.split(file_path, os.path.sep)
            # convert tensor with 'indexXX' to 'XX'
            str_num = tensorflow.strings.split(parts[-2], 'index')[-1]
            # convert tensor with XX as dtype string to tensor with XX to dtype int
            label = tensorflow.strings.to_number(str_num)
            # convert label to number [0,1] for regression
            return tensorflow.math.divide(label, (self.num_cols - 1))
        def decode_img(img):
            # convert the compressed string to a 3D uint8 tensor
            img = tensorflow.image.decode_jpeg(img, channels=1)
            # Use `convert_image_dtype` to convert to floats in the [0,1] range.
            img = tensorflow.image.convert_image_dtype(img, tensorflow.float32)
            # resize the image to the desired size.
            return tensorflow.image.resize(img, [self.IMG_HEIGHT, self.IMG_WIDTH])
        def process_path(file_path):
            # file_path is a tensor with dtype string
            label = get_label(file_path)
            # load the raw data from the file as a string
            img = tensorflow.io.read_file(file_path)
            img = decode_img(img)
            return img, label     
        def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
            # This is a small dataset, only load it once, and keep it in memory.
            # use `.cache(filename)` to cache preprocessing work for datasets that don't
            # fit in memory.
            if cache:
                if isinstance(cache, str):
                    ds = ds.cache(cache)
                else:
                    ds = ds.cache()

            ds = ds.shuffle(buffer_size=shuffle_buffer_size)

            # Repeat forever
            ds = ds.repeat()

            ds = ds.batch(self.BATCH_SIZE)

            # `prefetch` lets the dataset fetch batches in the background while the model
            # is training.
            ds = ds.prefetch(buffer_size=tensorflow.data.experimental.AUTOTUNE)

            return ds
        def show_batch(image_batch, label_batch):
            plot.figure(figsize=(10,10))
            for n in range(self.BATCH_SIZE):
                ax = plot.subplot(5,5,n+1)
                plot.xticks([])
                plot.yticks([])
                plot.imshow(numpy.squeeze(image_batch[n], axis=2), cmap=plot.cm.gray)
                plot.title(label_batch[n])
                plot.axis('off')
            plot.show()
        
        ''' Load data using above private functions '''
        self.window.destroy()
        print('Loading Images...')
        data_dir = pathlib.Path('Columns/')
        image_count = len(list(data_dir.glob('*/*.jpg')))
        list_ds = tensorflow.data.Dataset.list_files(str(data_dir/'*/*'))
        labeled_ds = list_ds.map(process_path, num_parallel_calls=tensorflow.data.experimental.AUTOTUNE)
        train_ds = prepare_for_training(labeled_ds)
        # Build Model
        print('Building Column Model...')
        col_mod = tensorflow.keras.Sequential()
        # Convolutional Layer 1
        col_mod.add(tensorflow.keras.layers.Conv2D(30, (3, 3), input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 1)))
        col_mod.add(tensorflow.keras.layers.Activation('relu'))
        col_mod.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # Convolutional Layer 2
        col_mod.add(tensorflow.keras.layers.Conv2D(30, (3, 3)))
        col_mod.add(tensorflow.keras.layers.Activation('relu'))
        col_mod.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # Dense Layer 1
        col_mod.add(tensorflow.keras.layers.Flatten())
        col_mod.add(tensorflow.keras.layers.Dense(60))
        col_mod.add(tensorflow.keras.layers.Activation('relu'))
        # Regression Layer
        col_mod.add(tensorflow.keras.layers.Dense(1))
        col_mod.summary()
        
        col_mod.compile(loss='mse',
                        optimizer='adam',
                        metrics=['mse'])
        # Train Model
        print('Training Model...')
        STEPS_PER_EPOCH = numpy.ceil(image_count/self.BATCH_SIZE)
        col_mod.fit(train_ds, epochs=4, steps_per_epoch=STEPS_PER_EPOCH)
        self.col_mod = col_mod

        # Reopen window
        self.window = tkinter.Tk()
        self.window.attributes('-zoomed', True)
        self.build_main_frame()
        track_btn = tkinter.Button(self.main_frame, text='Start Tracking', command=partial(Window.track, self))
        track_btn.place(relx=0.5, rely=0.5, anchor='center')
        self.window.mainloop()

    def track(self):
        '''
        Tracking stage, applies model to current pictures of user
        '''
        self.build_main_frame()
        video = cv2.VideoCapture(0)
        col_frame = tkinter.Frame(self.main_frame, bg='black')
        while True:
            cap, pic = video.read()
            # Convert picture to gray scale to match training data
            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            # Resize to match our training data
            pic = cv2.resize(pic, (self.IMG_WIDTH, self.IMG_HEIGHT))
            # Expand bc tensorflow predict expects array of images to predict
            pic = (numpy.expand_dims(pic,0))
            # Dont know if we have to reshape
            pic = pic.reshape(-1, self.IMG_WIDTH, self.IMG_HEIGHT, 1)
            # Rescale to have values between 0 and 255
            pic = pic / 255
            # Predict 
            prediction = self.col_mod.predict(pic)
            print(prediction)
            col_frame.place(relx=prediction[0][0], rely=0, relheight=1, relwidth=1/self.num_cols)
            self.window.update()

    def clean_imgs(self, folder):
        '''
        Only use if removing old images, like if changing users or changing the number of squares
        '''
        # Delete previous directory of images
        try:
            shutil.rmtree(f'{folder}/')
        except:
            print('Failed to remove folder')
        # Make directory images
        os.mkdir(folder)
        if folder == 'Columns':
            # Make directories for different columns
            for col in range(self.num_cols):
                os.mkdir(f'Columns/index{col}')
        elif folder == 'Rows':
            # Make directories for different columns
            for row in range(self.num_rows):
                os.mkdir(f'Rows/index{row}')

    def count_pics(self, folder):
        '''
        Begin program by counting the number of pictures for every row / column
        '''
        num_pic_dict = {}
        indices = 0
        if folder == 'Columns':
            indices = self.num_cols
        elif folder == 'Rows':
            indices = self.num_rows
        total_num_img = 0
        for index in range(indices):
            for path in pathlib.Path(f"{folder}/index{index}").iterdir():
                total_num_img += 1
            num_pic_dict[index] = total_num_img
            total_num_img = 0
        return num_pic_dict

    
window = Window(True, True)