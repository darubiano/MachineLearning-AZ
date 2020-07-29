#Clustering jerarquico

# importar los datos
dataset = read.csv("Mall_Customers.csv")
x = dataset[,4:5]

# Utilizar el dendrograma para encontrar el numero optimo de clusters
# matriz de distancias dist(x, method = "euclidean")
dendrograma = hclust(dist(x, method = "euclidean"), method = "ward.D") 

plot(dendrograma, main = "Dendrograma", xlab = "Clientes del centro comercial",
     ylab = "Distancia euclidea")

# Ajustar el clustering jerarquico a nuestro dataset
hc = hclust(dist(x, method = "euclidean"), method = "ward.D")

# Cortar el arbol (dendrograma) numero optimo 5
y_hc = cutree(hc, k=5)


#Visualizacion de los cluesters
#install.packages("cluster")
#library(cluster)
clusplot(x, y_hc, lines = 0, shade = TRUE,color = TRUE,
         labels = 4, plotchar = FALSE, span = TRUE, main = "CLustering de clientes",
         xlab ="Ingresos anuales", ylab = "Puntuacion (1-100)")






