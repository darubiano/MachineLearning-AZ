#install.packages("randomForest")
library(randomForest)
library(ggplot2)

# Regresion con arboles aleatorios

dataset = read.csv("Position_Salaries.csv")

dataset = dataset[,2:3]
# semilla 
set.seed(1234)
regression = randomForest(x = dataset[1], y = dataset$Salary,
                          ntree = 500)

# prediccion de nuevos datos
y_pred = predict(regression, newdata = data.frame(Level=6.5))


x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)

ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") +
  geom_line(aes(x= x_grid, y = predict(regression, newdata = data.frame(Level=x_grid)))
            ,color="blue")+
  ggtitle("Predccion random forest")+
  xlab("Nivel del empleado")+
  ylab("Sueldo en $")











