#Natural language processing


# Importar el dataset
dataset_original = read.delim("Restaurant_Reviews.tsv",
                     quote = '',stringsAsFactors = FALSE)


# Limpieza de textos
#install.packages("tm")
#install.packages("SnowballC")
#Stopwords
library(SnowballC)
# procesamiento de texto
library(tm)
# Tomar texto 
corpus = VCorpus(VectorSource(dataset_original$Review))
# Transformar texto a minuscula
corpus = tm_map(corpus, content_transformer(tolower))
#Consultar el primer elemento del corpus
# as.character(corpus[[1]])
# eliminar numeros
corpus = tm_map(corpus, removeNumbers)
# Eliminar signos de puntuacion
corpus = tm_map(corpus, removePunctuation)
# Eliminar palabras stopwords
corpus = tm_map(corpus, removeWords, stopwords(kind = 'en'))
# Eliminar palabras deribadas loved a love
corpus = tm_map(corpus, stemDocument)
# Eliminar espacios dobles
corpus = tm_map(corpus, stripWhitespace)

# Crear el modelo de bag words
dtm = DocumentTermMatrix(corpus)
# Eliminar las palabras poco probables
dtm = removeSparseTerms(dtm, 0.999)
# Transformar a dataframe
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Codificar la variable de clasificacion como factor
dataset$Liked = factor(dataset$Liked, levels = c(0,1))

# Dividir los datos en conjunto de entrenamiento y conjunto de test
# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.80)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Ajustar el Random Forest con el conjunto de entrenamiento.
#install.packages("randomForest")
library(randomForest)
classifier = randomForest(x = training_set[,-692],
                          y = training_set$Liked,
                          ntree = 10)

# Prediccion de los resultados con el conjunto de testing
y_pred = predict(classifier, newdata = testing_set[,-692])

# Crear la matriz de confusion
cm = table(testing_set[, 692], y_pred)
