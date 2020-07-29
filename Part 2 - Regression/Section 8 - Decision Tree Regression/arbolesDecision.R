#Arbol de decision para regresion
#install.packages("rpart")
library(rpart)
library(ggplot2)

dataset = read.csv("Position_Salaries.csv")

dataset = dataset[,2:3]


reg_tree = rpart(formula = Salary ~ .,
                 data = dataset,
                 control = rpart.control(minsplit = 1))


y_pred = predict(reg_tree, newdata= data.frame(Level = 6.5))


x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() + 
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = "red") +
  geom_line(aes(x= x_grid), y = predict(reg_tree, newdata = data.frame(Level = x_grid)),
            color = "blue") +
  ggtitle("Prediccion arbol de decision del sueldo en funcion del nivel del empleado") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")
















