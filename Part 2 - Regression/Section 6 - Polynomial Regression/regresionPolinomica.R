library(ggplot2)
# Regresion polinomica
dataset = read.csv("Position_Salaries.csv")
# eliminar la primera columna del dataset
dataset = dataset[,2:3]

# regresion lineal
lin_reg = lm(formula = Salary ~ .,
             data = dataset)

# regresion polinomica
# agregar otro nivel al cuadrado
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,
              data = dataset)

ggplot() + 
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = "red") +
  geom_line(aes(x= dataset$Level, y = predict(lin_reg, newdata = dataset)),
             color = "blue") +
  ggtitle("Prediccion lineal del sueldo en funcion del nivel del empleado") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")


ggplot() + 
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = "red") +
  geom_line(aes(x= dataset$Level, y = predict(poly_reg, newdata = dataset)),
            color = "blue") +
  ggtitle("Prediccion polinomial del sueldo en funcion del nivel del empleado") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")


# Visualizacion de los resultados del modelo polinomico
# crear un rango de valores de 0.1 en 0.1 de x
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)

ggplot() + 
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = "red") +
  geom_line(aes(x= x_grid, y = predict(poly_reg, newdata = data.frame(Level = x_grid,
                                                                      Level2 = x_grid^2,
                                                                      Level3 = x_grid^3,
                                                                      Level4 = x_grid^4))),
            color = "blue") +
  ggtitle("Prediccion polinomial del sueldo en funcion del nivel del empleado x_grid") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")

# Prediccion de nuevos resultados con regresion linea
y_pred_lin = predict(lin_reg, newdata = data.frame(Level = 6.5))


# prediccion de nuevos resultados con regresion polinomica
y_pred_poly = predict(poly_reg, newdata = data.frame(Level = 6.5,
                                                     Level2 = 6.5^2,
                                                     Level3 = 6.5^3,
                                                     Level4 = 6.5^4))


