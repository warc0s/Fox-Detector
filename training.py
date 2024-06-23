import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Cargar el dataset
df = pd.read_csv('dataset.csv')
df['label'] = df['label'].astype(str)

# Balancear el dataset mediante sobremuestreo
majority_class = df[df['label'] == '0']
minority_class = df[df['label'] == '1']
minority_class_upsampled = minority_class.sample(len(majority_class), replace=True)
df_balanced = pd.concat([majority_class, minority_class_upsampled])

# Dividir el dataset en entrenamiento y validación
train_df, val_df = train_test_split(df_balanced, test_size=0.2, random_state=42)

# Aumento de datos (sin recortes ni zoom)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # Ajuste de brillo
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Generadores de datos de entrenamiento y validación
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='images_dataset/',
    x_col='filename',
    y_col='label',
    target_size=(256, 256),
    batch_size=50,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory='images_dataset/',
    x_col='filename',
    y_col='label',
    target_size=(256, 256),
    batch_size=50,
    class_mode='binary'
)

# Modelo DenseNet con regularización
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)  # Agregar capa de Dropout
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compilar el modelo con un optimizador ajustado
opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping para evitar el sobreajuste
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Reduce la tasa de aprendizaje cuando la métrica de validación no mejora
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001
)

# Entrenar el modelo
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Guardar el modelo
model.save('fox_detector_model.keras')