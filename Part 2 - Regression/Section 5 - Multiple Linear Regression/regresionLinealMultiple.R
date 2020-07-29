# Regresion lienal multiple
library(caTools)

dataset = read.csv("50_Startups.csv")

# codificar las variables categoricas
dataset$State = factor(dataset$State,
                       levels = c("New York","California","Florida"),
                       labels = c(1,2,3))
set.seed(123)
# separar entrenamiento y testeo
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)


# Ajustar el modelo de regresion lineal multiple con el conjunto de entrenamiento
# variables dependientes ~ independientes, el punto ( . ) agrega todas las demas variables
regresion = lm(formula = Profit ~ . ,
               data = training_set)
#summary(regresion)

# predecir los resultados con el conjunto de testing
y_pred = predict(regresion, newdata = testing_set)

# Construri un modelo optimo con la eliminacion hacia atras
SL = 0.05
regresion = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regresion)

# se elimina el p valor mas alto state
regresion = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regresion)

# se elimina el p valor mas alto Administration
regresion = lm(formula = Profit ~ R.D.Spend  + Marketing.Spend,
               data = dataset)
summary(regresion)

# se elimina el p valor mas alto Marketing.Spend
regresion = lm(formula = Profit ~ R.D.Spend ,
               data = dataset)
summary(regresion)
# entre mas Adjusted R-squared sea 1 los datos seran mas lineales

