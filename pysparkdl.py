from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml.feature import  StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml.feature import OneHotEncoder
#from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml import Pipeline
from pyspark.sql.functions import rand
from pyspark.mllib.evaluation import MulticlassMetrics

# Keras / Deep Learning
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers, regularizers
from keras.optimizers import Adam

# Elephas for Deep Learning on Spark
from elephas.ml_model import ElephasEstimator

# Spark Session
conf = SparkConf().setAppName('Spark DL Tabular Pipeline').setMaster('local[6]')
sc = SparkContext.getOrCreate(conf=conf)
sql_context = SQLContext(sc)
sc

# Load Data to Spark Dataframe
df = sql_context.read.csv('/content/drive/MyDrive/Dataset/DDL/bank.csv',
                    header=True,
                    inferSchema=True)
df.printSchema()

# Preview Dataframe (Pandas Preview is Cleaner)
df.limit(5).toPandas()

# Drop Unnessary Features (Day and Month)
df = df.drop('day', 'month')

# Preview Dataframe
df.limit(5).toPandas()

# Helper function to select features to scale given their skew
def select_features_to_scale(df=df, lower_skew=-2, upper_skew=2, dtypes='int32', drop_cols=['']):
    
    # Empty Selected Feature List for Output
    selected_features = []
    
    # Select Features to Scale based on Inputs ('in32' type, drop 'ID' columns or others, skew bounds)
    feature_list = list(df.toPandas().select_dtypes(include=[dtypes]).columns.drop(drop_cols))
    
    # Loop through 'feature_list' to select features based on Kurtosis / Skew
    for feature in feature_list:

        if df.toPandas()[feature].kurtosis() < -2 or df.toPandas()[feature].kurtosis() > 2:
            
            selected_features.append(feature)
    
    # Return feature list to scale
    return selected_features
  
  # Spark Pipeline
cat_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
num_features = ['age','balance','duration','campaign','pdays','previous']
label = 'deposit'

# Pipeline Stages List
stages = []

# Loop for StringIndexer and OHE for Categorical Variables
for features in cat_features:
    
    # Index Categorical Features
    string_indexer = StringIndexer(inputCol=features, outputCol=features + "_index")
    
    # One Hot Encode Categorical Features
    encoder = OneHotEncoder(inputCols=[string_indexer.getOutputCol()],
                                     outputCols=[features + "_class_vec"])
    # Append Pipeline Stages
    stages += [string_indexer, encoder]
    
# Index Label Feature
label_str_index =  StringIndexer(inputCol=label, outputCol="label_index")

# Scale Feature: Select the Features to Scale using helper 'select_features_to_scale' function above and Standardize 
unscaled_features = select_features_to_scale(df=df, lower_skew=-2, upper_skew=2, dtypes='int32', drop_cols=['id'])
print("Unscaled_features",unscaled_features)
unscaled_assembler = VectorAssembler(inputCols=unscaled_features, outputCol="unscaled_features")
scaler = StandardScaler(inputCol="unscaled_features", outputCol="scaled_features")

stages += [unscaled_assembler, scaler]

# Create list of Numeric Features that Are Not Being Scaled
num_unscaled_diff_list = list(set(num_features) - set(unscaled_features))

# Assemble or Concat the Categorical Features and Numeric Features
assembler_inputs = [feature + "_class_vec" for feature in cat_features] + num_unscaled_diff_list

assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="assembled_inputs") 

stages += [label_str_index, assembler]

# Assemble Final Training Data of Scaled, Numeric, and Categorical Engineered Features
assembler_final = VectorAssembler(inputCols=["scaled_features","assembled_inputs"], outputCol="features")

stages += [assembler_final]
stages

# Set Pipeline
pipeline = Pipeline(stages=stages)

# Fit Pipeline to Data
pipeline_model = pipeline.fit(df)

# Transform Data using Fitted Pipeline
df_transform = pipeline_model.transform(df)

# Preview Newly Transformed Data
df_transform.limit(5).toPandas()

# Data Structure Type is a PySpark Dataframe
type(df_transform)

# Select only 'features' and 'label_index' for Final Dataframe
df_transform_fin = df_transform.select('features','label_index')
df_transform_fin.limit(5).toPandas()

# Shuffle Data
df_transform_fin = df_transform_fin.orderBy(rand())
# Split Data into Train / Test Sets
train_data, test_data = df_transform_fin.randomSplit([.8, .2],seed=1234)

# Number of Classes
nb_classes = train_data.select("label_index").distinct().count()

# Number of Inputs or Input Dimensions
input_dim = len(train_data.select("features").first()[0])

# Set up Deep Learning Model / Architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import tensorflow as tf
sgd = tf.keras.optimizers.SGD(learning_rate = 0.01)
model = Sequential()
model.add(Dense(256, input_shape=(input_dim,), activity_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(256, activity_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd')

# Model Summary
model.summary()

# Initialize SparkML Estimator and Get Settings
estimator = ElephasEstimator()
estimator.setFeaturesCol("features")
estimator.setLabelCol("label_index")
estimator.set_keras_model_config(model.to_yaml())
estimator.set_categorical_labels(True)
estimator.set_nb_classes(nb_classes)
estimator.set_num_workers(1)
estimator.set_epochs(25) 
estimator.set_batch_size(64)
estimator.set_verbosity(1)
estimator.set_validation_split(0.10)
estimator.set_optimizer_config(sgd)
estimator.set_mode("synchronous")
estimator.set_loss("binary_crossentropy")
estimator.set_metrics(['acc'])

# Create Deep Learning Pipeline
dl_pipeline = Pipeline(stages=[estimator])
print(dl_pipeline)

def dl_pipeline_fit_score_results(dl_pipeline=dl_pipeline,
                                  train_data=train_data,
                                  test_data=test_data,
                                  label='label_index'):
    
    fit_dl_pipeline = dl_pipeline.fit(train_data)
    pred_train = fit_dl_pipeline.transform(train_data)
    pred_test = fit_dl_pipeline.transform(test_data)
    
    pnl_train = pred_train.select(label, "prediction")
    pnl_test = pred_test.select(label, "prediction")
    
    pred_and_label_train = pnl_train.rdd.map(lambda row: (row[label], row['prediction']))
    pred_and_label_test = pnl_test.rdd.map(lambda row: (row[label], row['prediction']))
    
    metrics_train = MulticlassMetrics(pred_and_label_train)
    metrics_test = MulticlassMetrics(pred_and_label_test)
    
    print("Training Data Accuracy: {}".format(round(metrics_train.precision(1.0),4)))
    print("Training Data Confusion Matrix")
    display(pnl_train.crosstab('label_index', 'prediction').toPandas())
    
    print("\nTest Data Accuracy: {}".format(round(metrics_test.precision(1.0),4)))
    print("Test Data Confusion Matrix")
    display(pnl_test.crosstab('label_index', 'prediction').toPandas())
    
    dl_pipeline_fit_score_results(dl_pipeline=dl_pipeline,
                              train_data=train_data,
                              test_data=test_data,
                              label='label_index');
dl_pipeline_fit_score_results(dl_pipeline=dl_pipeline,
                              train_data=train_data,
                              test_data=test_data,
                              label='label_index');
dl_pipeline_fit_score_results(dl_pipeline=dl_pipeline,
                              train_data=train_data,
                              test_data=test_data,
                              label='label_index');
dl_pipeline_fit_score_results(dl_pipeline=dl_pipeline,
                              train_data=train_data,
                              test_data=test_data,
                              label='label_index');
