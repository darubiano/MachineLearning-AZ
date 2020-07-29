# importar dataset
#install.packages("ggplot2")
library(caTools)
library(ggplot2)

dataset = read.csv("Salary_Data.csv")

set.seed(123)
# separar los datos de entrenamiento y testeo 2/3 para entrenar
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)


# Ajustar el modelo de regresion lineal simple con el conjunto de entrenamiento
# variables dependientes ~ independientes
regresion = lm(formula = Salary ~ YearsExperience,
               data = training_set)

# resumen del modelo de regresion
#summary(regresion)

# Predecir resultado del conjunto de test
y_pred = predict(regresion, newdata = testing_set)

# Visualizacion de los resultados en el conjunto de entrenamiento
ggplot() +
  geom_point(aes(x= training_set$YearsExperience, y = training_set$Salary),
             colour = "red")+
  geom_line(aes(x = training_set$YearsExperience,
                y = predict(regresion, newdata = training_set)),
            colour = "blue")+
  ggtitle("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)")+
  xlab("Años de Experiencia")+
  ylab("Sueldo (en $)")


# Visualizacion de los resultados en el conjunto de testing
ggplot() +
  geom_point(aes(x= testing_set$YearsExperience, y = testing_set$Salary),
             colour = "red")+
  geom_line(aes(x = training_set$YearsExperience,
                y = predict(regresion, newdata = training_set)),
            colour = "blue")+
  ggtitle("Sueldo vs Años de Experiencia (Conjunto de testeo)")+
  xlab("Años de Experiencia")+
  ylab("Sueldo (en $)")










