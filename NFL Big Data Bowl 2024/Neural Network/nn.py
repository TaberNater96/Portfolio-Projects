#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Re-encode 'position_numeric' to have continuous values starting from 0 as we filtered the main_df for defensive players
model_df['position_numeric'], _ = pd.factorize(model_df['position_numeric'])

# Update the number of unique positions
num_positions = model_df['position_numeric'].nunique()
positions = model_df['position_numeric']

warnings.filterwarnings("ignore") 

# Set a random seed value for reproducability
seed_value= 6
os.environ['PYTHONHASHSEED']=str(seed_value) # ensures the hash operation is reproducible for consistent behavior
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# We need to tell the neural network what is a categorical variable and what is a continuous value
positions = model_df['position_numeric'].astype(int)
formations = model_df['offenseFormationNumeric'].astype(int)
defenders = model_df['defendersInTheBox'].astype(int)

# Standardize the numerical features for training
numerical_features = model_df.drop(['tackle', 'position_numeric', 'offenseFormationNumeric', 'defendersInTheBox'], axis=1)
scaler = StandardScaler()
numerical_features = scaler.fit_transform(numerical_features)

X_train_num, X_test_num, positions_train, positions_test, formations_train, formations_test, defenders_train, defenders_test, y_train, y_test = train_test_split(
    numerical_features, positions, formations, defenders, model_df['tackle'], test_size=0.2, random_state=6, stratify=model_df['tackle'])

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Define the size of the input for the embedding layers
num_positions = positions.nunique()
num_formations = formations.nunique()

# Define the size of the embeddings based on position and formation (categoricals)
embedding_size_position = 6  
embedding_size_formation = 3  

# Input layers
position_input = Input(shape=(1,), name='position_input')
offense_formation_input = Input(shape=(1,), name='offense_formation_input')
defenders_input = Input(shape=(1,), name='defenders_input')
numerical_input = Input(shape=(X_train_num.shape[1],), name='numerical_input')

# Embedding layers
position_embedding = Embedding(num_positions, embedding_size_position, input_length=1)(position_input)
offense_formation_embedding = Embedding(num_formations, embedding_size_formation, input_length=1)(offense_formation_input)

# Flatten the embeddings
position_embedding_flat = Flatten()(position_embedding)
offense_formation_embedding_flat = Flatten()(offense_formation_embedding)

# Concatenate embeddings 
concatenated_features = Concatenate()([
    position_embedding_flat, 
    offense_formation_embedding_flat, 
    defenders_input,  
    numerical_input
])

# Hypermodel definition
def build_model(hp):
    # Input layers
    position_input = Input(shape=(1,), name='position_input')
    offense_formation_input = Input(shape=(1,), name='offense_formation_input')
    defenders_input = Input(shape=(1,), name='defenders_input')
    numerical_input = Input(shape=(X_train_num.shape[1],), name='numerical_input')

    # Embedding layers with tunable sizes
    position_embedding = Embedding(num_positions, hp.Int('embedding_size_position', min_value=3, max_value=10, step=1), input_length=1)(position_input)
    offense_formation_embedding = Embedding(num_formations, hp.Int('embedding_size_formation', min_value=3, max_value=10, step=1), input_length=1)(offense_formation_input)

    # Flatten embeddings
    position_embedding_flat = Flatten()(position_embedding)
    offense_formation_embedding_flat = Flatten()(offense_formation_embedding)

    # Concatenate features
    concatenated_features = Concatenate()([position_embedding_flat, 
                                           offense_formation_embedding_flat, 
                                           defenders_input, 
                                           numerical_input])
    
    """
    Layers: this is one of the most important parts of the deep learning process and where the layers will be refined over 
    and over until a desired output is achieved, many combinations must addressed in order to find the optimal combination, 
    this is the heart of the neural network hyperparameter tuning process, and by far the most time consuming. This 
    hyperparameter tuning process can be more efficiently calibrated through the use of the Keras Tuner which is designed
    to find the optimal hyperparameters for the given intervals.
    """
    
    hidden_layer = concatenated_features

    # Tunable number of layers and units
    for i in range(hp.Int('num_layers', 1, 5)):
        hidden_layer = Dense(hp.Int('units_' + str(i), min_value=32, max_value=256, step=32), activation='relu')(concatenated_features if i == 0 else hidden_layer)
        dropout_rate = hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1)
        hidden_layer = tf.keras.layers.Dropout(dropout_rate)(hidden_layer)

    # Output layer
    output_layer = Dense(1, activation='sigmoid')(hidden_layer) # output a binary value

    # Model assembly
    model = Model(inputs=[position_input, 
                          offense_formation_input, 
                          defenders_input,  
                          numerical_input], outputs=output_layer)

    # Compile the model with tunable learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

# Tuner configuration, this is the default tuner to use but is very time consuming and cost a lot of resources
# This tuner was used first but has taken too long, so we'll use a tuner that sets specific boundries
# tuner = kt.Hyperband(build_model, 
                     # objective='val_loss', # this is the main focus of the model due to the dataset's nature
                     # max_epochs=30, 
                     # factor=3,
                     # directory='NFL_ANN_Models', # save the tuner to the cwd for callbacks   
                     # project_name='Hyperparameter_Tuning_Run2') # update number before each run

# Since the tuner has gone on for over 4 days and there isn't much improvement, we will use this to speed up the process
tuner = RandomSearch(build_model, 
                     objective='val_loss', # this is the main focus of the model due to the dataset's nature
                     max_trials=25, 
                     executions_per_trial=1,
                     directory='NFL_ANN_Models', # save the tuner to the cwd for callbacks   
                     project_name='Hyperparameter_Tuning_Run4') # update number before each run

# Create a log directory with a timestamp to have a unique directory for each run
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

"""
Callbacks: This is the most importanct section for evaluating the performace of the model. Here we are creating;

Early Stopping Callback: This is used to stop the iterations when the model declines or plateaus
Model Checkpoint Callback: Used to monitor the current-most-optimal epoch based on validation loss and save it
TensorBoard Callback: Creates a separate dashboard with detailed performance analytics
"""

# Early Stopping Callback
early_stopping_callback = EarlyStopping(monitor='val_loss', 
                                        patience=10, 
                                        verbose=1)
# Model Checkpoint Callback
model_checkpoint_callback = ModelCheckpoint('best_model.h5', 
                                            monitor='val_loss', 
                                            save_best_only=True, 
                                            verbose=1)
# TensorBoard Callback with datetime format
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, 
                                   histogram_freq=1)
# Combine all callbacks into a single list
ANN_callbacks = [early_stopping_callback, 
                 model_checkpoint_callback, 
                 tensorboard_callback]

# Start the hyperparameter search
tuner.search([positions_train, formations_train, defenders_train, X_train_num], y_train,
             epochs=15, # reduce the epoch size to speed up the tuner
             validation_split=0.1,
             callbacks=[early_stopping_callback])

# Identify the optimal hyperparameters to use in model training
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build and train the model with the best hyperparameters
model = build_model(best_hps)
history = model.fit(
    [positions_train, formations_train, defenders_train, X_train_num], y_train,
    epochs=15, # ensure this matches the tuner search parameter
    validation_split=0.1, # split 10 percent of the data to validate the model
    callbacks=ANN_callbacks, # insert callbacks
    class_weight=class_weights_dict, # insert class weights
    verbose=1
)

# Evaluate the model
evaluation = model.evaluate([positions_test, formations_test, defenders_test, X_test_num], y_test)
print(f'Test loss: {evaluation[0]}') 
print(f'Test accuracy: {evaluation[1]}')

# Run tensorboard --logdir=logs/fit in the working directory command prompt to gain access to the full dashboard
# Then navigate to http://localhost:6006 to see performances

best_model = load_model('best_model.h5')

# Standardize the numerical features separately for training and testing sets
scaler = StandardScaler()
numerical_features_train = scaler.fit_transform(X_train_num)
numerical_features_test = scaler.transform(X_test_num)

# Evaluate the best model on the test set to see how it performs on unseen data
test_loss, test_accuracy = best_model.evaluate([positions_test, formations_test, defenders_test, numerical_features_test], y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Make predictions based on the optimal performance
#y_pred = best_model.predict([positions_test, formations_test, numerical_features_test])

