
#Apriori

# Preprocesado de datos
dataset = read.csv("Market_Basket_Optimisation.csv", header = FALSE)

#install.packages("arules")
#library(arules)
# crear 
dataset = read.transactions("Market_Basket_Optimisation.csv",
                            sep = ",", rm.duplicates = TRUE)
summary(dataset)
# Graficar los datos mas frecuentes
itemFrequencyPlot(dataset, topN=10)

# 3 * 7 / 7500, 3 ventas diarias en 7 dias sobre el total de datos soporte minimo
# Entrenar el algoritmo apriori con el dataset
rules = apriori(data = dataset,
                parameter = list(support= 0.003, confidence= 0.2))


#Visualizacion de los resultados
inspect(sort(rules, by = 'lift')[1:10])


#install.packages("arulesViz")
#library(arulesViz)
#Grafica
plot(rules, method = "graph", engine = "htmlwidget")

