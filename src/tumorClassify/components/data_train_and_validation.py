import cv2
import numpy as np
import tensorflow as tf
from vit_keras import vit
import os
from sklearn.model_selection import train_test_split
from tumorClassify.utils.metrics import recall_c0, recall_c1, recall_c2, precision_c0, precision_c1, precision_c2
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import mlflow
import mlflow.tensorflow
import dagshub
from urllib.parse import urlparse
from pathlib import Path
from dotenv import load_dotenv
from tensorflow.keras.mixed_precision import set_global_policy

load_dotenv()  # This loads the environment variables from the .env file

class TrainValEval:
    def __init__(self, config, param):
    def __init__(self, config, param):
        self.config = config
        self.param = param

        self.path = self.config.data_training_validation.unzip_dir
        self.size = self.param.image_params.SIZE
        self.depth = self.param.image_params.DEPTH
        self.model = self.param.model_params.MODEL

        #print(self.config)

    def set_gpu(self):
        # Enable GPU memory growth to avoid OOM errors
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"Error enabling memory growth: {e}")

        # Enable mixed precision training
        #set_global_policy('mixed_float16')

    def load_image(self, path, upload=False):
        if upload:
            # Convert the file to an opencv image.
            image = np.asarray(bytearray(path.read()), dtype=np.uint8)
            image = cv2.imdecode(image, 1)
            # image = PIL.Image.open(path)
            # image = np.array(path)
            # print(type(image))
            # print(image.shape)
        else:
            image = cv2.imread(path)
        #image = cv2.imread(path)
        image = cv2.resize(image, (self.size, self.size))
        if self.model=='ResNet152V2':
          image = tf.keras.applications.resnet_v2.preprocess_input(image)
        elif self.model=='VGG19':
          image = tf.keras.applications.vgg19.preprocess_input(image)
        elif self.model=='EfficientNetB7':
          image = tf.keras.applications.efficientnet.preprocess_input(image)
        elif self.model=='EfficientNetV2S':
          image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
        elif self.model=='ConvNeXtBase':
          pass
        elif self.model=='vit_b16':
          image = vit.preprocess_inputs(image)
        if self.depth==1:
          image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #image = image/255.
        return image

    def create_label_array(self, state, length):
        label_map = {
            'normal': [1, 0, 0],
            'benign': [0, 1, 0],
            'malignant': [0, 0, 1]
        }
        return np.repeat([label_map[state]], length, axis=0)

    def load_data(self, state):
        images = []
        masks  = []

        imagesmask_path = os.listdir(os.path.join(self.path,state))
        images_path = [i for i in imagesmask_path if 'mask' not in i]
        mask_path   = [i for i in imagesmask_path if 'mask' in i]
        #print(images_path,mask_path)
        for path in images_path:
            #print(path)
            file = os.path.basename(path).replace('.png','')
            #print(file)
            mfiles=[]
            for mfile in mask_path:
                if file in mfile:
                    mfiles.append(mfile)
            #if '100' in file:print(mfiles)
            if len(mfiles)==0:
                print(mfiles)
                continue
            elif len(mfiles)==1:
                images.append(self.load_image(os.path.join(self.path, state, path)))
                masks.append(self.load_image(os.path.join(self.path, state, mfiles[0])))
                #if '248' in file:print(load_image(os.path.join(config.DATA_DIR,state,mfiles[0]), size)[150])
            elif len(mfiles)>1:
                img = 0
                for mfile in mfiles:
                    img += self.load_image(os.path.join(self.path, state, mfile))
                    #if '25' in file:print(img[60],'--'*10)
                    #img+=img
                    #if '100' in file:print('summed',img[60],'--'*10)

                    #print(mfile)
                #if '25' in file:print(img[60]/len(mfiles))
                img = np.array(img>0.5, dtype='float64')
                images.append(self.load_image(os.path.join(self.path, state, path)))
                #masks.append(img/len(mfiles))
                masks.append(np.array(img>0.5, dtype='float64'))

        y_state_label = self.create_label_array(state, len(np.array(images,dtype=np.float32)))

        #create_label_array = lambda n, label: np.repeat([label], n, axis=0)

        #y_state_label = create_label_array(len(np.array(images,dtype=np.float32)), [1, 0, 0])
        #y_benign_label = create_label_array(len(y_benign), [0, 1, 0])
        #y_malignant_label = create_label_array(len(y_malignant), [0, 0, 1])

        if self.depth == 1:
            return np.expand_dims(np.array(images,dtype=np.float32), -1), np.expand_dims(np.array(masks,dtype=np.float32), -1), y_state_label
        else:
            return np.array(images,dtype=np.float32), np.array(masks,dtype=np.float32), y_state_label

    def batch_dataset(self, x, y, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        print(batch_size, self.param.batch_params.AUTOTUNE)
        dataset = dataset.batch(batch_size).cache()
                   #.prefetch(buffer_size=self.param.batch_params.AUTOTUNE))
        return dataset

    def prepare_dataset(self, x_benign, x_malignant, x_normal, y_benign_label, y_malignant_label, y_normal_label):

        x_total_label = np.concatenate((x_benign, x_malignant, x_normal), axis=0)
        y_total_label = np.concatenate((y_benign_label, y_malignant_label, y_normal_label), axis=0)
        y_total_label = np.expand_dims(y_total_label, -1)
        random_state = 24

        X_train_total_label, X_test_total_label, y_train_total_label, y_test_total_label = train_test_split(
            x_total_label,
            y_total_label,
            test_size=self.param.batch_params.TEST_SIZE,
            shuffle=self.param.batch_params.SHUFFLE,
            random_state=self.param.batch_params.SEED)

        # y_malignant = np.expand_dims(y_malignant, -1)
        y_train_total_label = np.reshape(y_train_total_label, (-1, 3))
        y_test_total_label = np.reshape(y_test_total_label, (-1, 3))

        ny = len(y_benign_label) + len(y_malignant_label) + len(y_normal_label)
        n_classes = 3
        self.class_weight = {0: ny / (n_classes * len(y_normal_label)),
                        1: ny / (n_classes * len(y_benign_label)),
                        2: ny / (n_classes * len(y_malignant_label))}

        #ds_train_label = tf.data.Dataset.from_tensor_slices((X_train_total_label, y_train_total_label))
        #ds_test_label = tf.data.Dataset.from_tensor_slices((X_test_total_label, y_test_total_label))

        self.dataset_train_label = self.batch_dataset(X_train_total_label, y_train_total_label, self.param.batch_params.BATCH_SIZE)
        # dataset_val = prepare_dataset(val_df_scaled,62)
        self.dataset_test_label = self.batch_dataset(X_test_total_label, y_test_total_label, 1)

        return self.dataset_train_label, self.dataset_test_label

    def dense_block(self, x, units, dropout_rates, kernel_reg=0.01, bias_reg=0.01):
        for units_i, dropout_rate in zip(units, dropout_rates):
            x = tf.keras.layers.Dense(units_i, activation='relu',
                                      kernel_regularizer=tf.keras.regularizers.l2(kernel_reg),
                                      bias_regularizer=tf.keras.regularizers.l2(bias_reg))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        return x

    def prepare_model(self):

        POOLING = self.param.model_params.POOLING

        # Input layer
        input_layer = tf.keras.layers.Input(shape=(self.size, self.size, self.depth))

        if self.model=='ResNet152V2':
            self.pretrained_model = tf.keras.applications.ResNet152V2(include_top=False,
                                                                      input_shape=(self.size, self.size, self.depth),
                                                                      weights="imagenet",
                                                                      pooling=POOLING,
                                                                      )(input_layer)
        elif self.model=='VGG19':
            self.pretrained_model = tf.keras.applications.VGG19(
                                                            include_top=False,
                                                            input_shape=(self.size, self.size, self.depth),
                                                            pooling=POOLING,
                                                            weights='imagenet'
                                                            )(input_layer)
        elif self.model=='EfficientNetB7':
            self.pretrained_model = tf.keras.applications.EfficientNetB7(include_top=False,
                                                                    input_shape=(self.size, self.size, self.depth),
                                                                    weights="imagenet",
                                                                    pooling=POOLING,
                                                                    )(input_layer)
        elif self.model=='EfficientNetV2S':
            self.pretrained_model = tf.keras.applications.EfficientNetV2S(
            include_top=False,
            input_shape=(self.size, self.size, self.depth),
            weights="imagenet",
            pooling=POOLING,
            )(input_layer)
        elif self.model=='ConvNeXtBase':
            self.pretrained_model = tf.keras.applications.ConvNeXtBase(
            model_name="convnext_base",
            include_top=False,
            include_preprocessing=True,
            weights="imagenet",
            pooling=POOLING,
            )(input_layer)
        elif self.model=='vit_b16':
            self.pretrained_model = vit.vit_b16(
            #image_size=image_size,
            #activation='sigmoid',
            pretrained = True,
            include_top = False,
            pretrained_top = False,
            )(input_layer)

        # Flatten layer
        flattened = tf.keras.layers.Flatten()(self.pretrained_model)

        # Dense layers
        #dense1 = tf.keras.layers.Dense(256, activation='relu',
                      #kernel_regularizer=regularizers.l2(0.01),  # L2 regularization
                      #bias_regularizer=regularizers.l2(0.01))(flattened)
        #batch_norm1 = tf.keras.layers.BatchNormalization()(dense1)
        #dropout1 = tf.keras.layers.Dropout(0.5)(batch_norm1)
        #dense1 = tf.keras.layers.Dense(256, activation='relu',
        #              kernel_regularizer=regularizers.l2(0.01),  # L2 regularization
        #              bias_regularizer=regularizers.l2(0.01))(flattened)
        #batch_norm1 = tf.keras.layers.BatchNormalization()(dense1)
        #dropout1 = tf.keras.layers.Dropout(0.3)(batch_norm1)
        #dense2 = tf.keras.layers.Dense(64, activation='relu',
        #              kernel_regularizer=regularizers.l2(0.01),  # L2 regularization
        #              bias_regularizer=regularizers.l2(0.01),)(dropout1)
        #batch_norm2 = tf.keras.layers.BatchNormalization()(dense2)
        #dropout2 = tf.keras.layers.Dropout(0.2)(batch_norm2)
        #output_layer = tf.keras.layers.Dense(3, activation='softmax')(dropout2)

        # Define the number of units and dropout rates for each dense layer
        units = self.param.model_params.LINEAR_UNITS  # Example: three dense layers with 256, 64, and 32 units
        dropout_rates = self.param.model_params.DPOUT_UNITS  # Example: dropout rates corresponding to each dense layer

        # Create dense layers
        dense_output = self.dense_block(flattened, units, dropout_rates)

        # Output layer
        output_layer = tf.keras.layers.Dense(3, activation='softmax')(dense_output)

        # Functional model
        self.fine_tune_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        self.fine_tune_model.layers[1].trainable=False
        #print(self.fine_tune_model.summary())
        #self.pretrained_model.trainable=False

    def train_model(self):
        # Initialize MLflow
        mlflow.tensorflow.autolog()

        # Define and compile your model
        self.fine_tune_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[recall_c0, recall_c1, recall_c2, precision_c0, precision_c1, precision_c2]
        )

        # Define the directory and file name using Path
        checkpoint_dir = Path("artifacts") / "models"
        model_checkpoint_path = checkpoint_dir / "model.h5"

        # Create the directory if it doesn't exist
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Define callbacks
        checkpoint = ModelCheckpoint(
            #self.model + 'weights.hdf5',
            str(model_checkpoint_path),  # Convert Path object to string
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min',
            #save_weights_only=True,
            save_freq='epoch',
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=15,
            min_lr=0.00005
        )

        callbacks_list = [checkpoint, reduce_lr]

        ##dagshub.init(repo_owner='krunalgedia', repo_name='Breast_Tumor_Classification', mlflow=True)
        #mlflow.set_registry_uri('https://dagshub.com/krunalgedia/Breast_Tumor_Classification.mlflow')
        #tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        mlflow.set_tracking_uri(self.config.data_training_validation.mlflow_dir)  # Local directory for MLflow runs

        # Start an MLflow run
        with mlflow.start_run(run_name=self.model) as run:
            # Log parameters
            mlflow.log_params(self.param.batch_params)
            mlflow.log_params(self.param.model_params)
            mlflow.log_params(self.param.image_params)
            #mlflow.log_param("model_name", self.model)
            #mlflow.log_param("learning_rate", 0.0002)
            #mlflow.log_param("epochs", self.param.model_params.NUM_EPOCHS)
            #mlflow.log_param("batch_size", self.param.batch_params.BATCH_SIZE)

            # Fit model
            history = self.fine_tune_model.fit(
                self.dataset_train_label,
                validation_data=self.dataset_test_label,
                callbacks=callbacks_list,
                epochs=self.param.model_params.NUM_EPOCHS,
                class_weight=self.class_weight
            )
            #self.fine_tune_model.save('Detection_model', save_format='h5')
            #print(history.history)
            # Optionally, you can log additional metrics manually
            # Log metrics for each epoch
            #metrics = history.history.keys()
            #for epoch in range(len(history.history[list(metrics)[0]])):
            #    for metric in metrics:
            #        mlflow.log_metric(metric, history.history[metric][epoch], step=epoch)

            # Log metrics (manually handle EagerTensor conversion)
            metrics = history.history.keys()
            for epoch in range(len(history.history[list(metrics)[0]])):
                for metric in metrics:
                    value = history.history[metric][epoch]
                    if isinstance(value, tf.Tensor):
                        value = value.numpy()  # Convert to Python float
                    mlflow.log_metric(metric, value, step=epoch)

            print("Inspecting model attributes for serialization issues:")
            for attr in dir(self.fine_tune_model):
                try:
                    value = getattr(self.fine_tune_model, attr)
                    if isinstance(value, tf.Tensor):
                        print(f"Attribute '{attr}' contains an EagerTensor with value: {value.numpy()}")
                    elif isinstance(value, list):
                        if any(isinstance(item, tf.Tensor) for item in value):
                            print(f"Attribute '{attr}' contains a list with EagerTensor.")
                except Exception as e:
                    print(f"Error inspecting attribute '{attr}': {e}")

            if hasattr(self.fine_tune_model, 'problematic_attribute'):
                value = getattr(self.fine_tune_model, 'problematic_attribute')
                if isinstance(value, tf.Tensor):
                    setattr(self.fine_tune_model, 'problematic_attribute', value.numpy().tolist())

            # Register the model
            run_id = run.info.run_id  # Access the run ID

            #mlflow.tensorflow.log_model(self.fine_tune_model, "model")

            # Model registry does not work with file store
            #print(tracking_url_type_store)

            #if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                #mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            #else:
                #mlflow.keras.log_model()
                #mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")

            mlflow.register_model(f"runs:/{run_id}/model", self.model)

            # Optionally, set a model version stage (e.g., "Staging", "Production")
            #client = mlflow.tracking.MlflowClient()
            #client.transition_model_version_stage(
            #    name=self.model,
            #    version=0,
            #    stage="Staging"
            #)

            # End the MLflow run
            mlflow.end_run()

