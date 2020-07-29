# SVR 
# install.packages("e1071")
library(e1071)
library(ggplot2)

dataset = read.csv("Position_Salaries.csv")

dataset = dataset[,2:3]

# maquinas de porte vectorial
regression = svm(formula = Salary ~ .,
                 data = dataset,
                 type = "eps-regression",
                 kernel = "radial")

y_predict = predict(regression, newdata = data.frame(Level = 6.5))

ggplot() + 
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = "red") +
  geom_line(aes(x= dataset$Level, y = predict(regression, newdata = dataset)),
            color = "blue") +
  ggtitle("Prediccion lineal del sueldo en funcion del nivel del empleado") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")

















