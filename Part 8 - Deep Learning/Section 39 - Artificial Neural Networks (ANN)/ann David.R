# Red neuronal arfiticial

dataset = read.csv("Churn_Modelling.csv")
dataset = dataset[,4:14]



# Codificar las variables categoricas
dataset$Geography = as.numeric(factor(dataset$Geography,
                         levels = c("France", "Spain", "Germany"),
                         labels = c(1, 2, 3)))

dataset$Gender = as.numeric(factor(dataset$Gender,
                           levels = c("Female", "Male"),
                           labels = c(1,2)))

# Dividir los datos de entranamiento y testeo
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Escalado de valores
training_set[,-11] = scale(training_set[,-11])
testing_set[,-11] = scale(testing_set[,-11])

# Crear la red neuronal
#install.packages("h2o")
library(h2o)
h2o.init(nthreads = -1)
# activacion rectificador, capa ocultas [6,6]
classifier = h2o.deeplearning(y = "Exited",
                              training_frame = as.h2o(training_set),
                              activation = "Rectifier",
                              hidden = c(6,6),
                              epochs = 100,
                              train_samples_per_iteration = -2)

#Prediccion de los resultados
prob_pred = h2o.predict(classifier, newdata= as.h2o(testing_set[,-11]))
# transformar probabilidad a categoria
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Crear matriz de confusion
cm = table(testing_set[,11], as.vector(y_pred))

#Cerrar la sesion de H2o
h2o.shutdown()

