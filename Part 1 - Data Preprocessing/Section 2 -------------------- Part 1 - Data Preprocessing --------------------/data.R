# Plantilla para el pre procesado de datos
# imporat el dataset

dataset = read.csv('Data.csv')

# Tratamiento de los valores NA
# si el valor es NA aplica la funcion de media si no retorna el mismo valor
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN= function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Age,FUN = function(x) mean(x, na.rm = TRUE) ),
                        dataset$Salary)

# codificar las variables categoricas
dataset$Country = factor(dataset$Country,
                         levels = c("France", "Spain", "Germany"),
                         labels = c(1,2,3))

dataset$Purchased = factor(dataset$Purchased,
                           levels = c("No","Yes"),
                           labels = c(0,1))

# Dividir los datos en conjunto de entrenamiento y conjunto de testeo
#install.packages("caTools")
library(caTools)
# crear un semilla de los elementos seleccionados del dataset
set.seed(123)
# dividir los datos para entrenar por purchased 80% entrenamiento y 20% testeo
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# tomar los datos de entramiento 80%
training_set = subset(dataset, split == TRUE)
# tomar los datos de entramiento 20%
testing_set = subset(dataset, split == FALSE)


# Escalado de valores de edad y salario (normalizar datos)
training_set[,2:3] = scale(training_set[,2:3])
testing_set[,2:3] = scale(testing_set[,2:3])










