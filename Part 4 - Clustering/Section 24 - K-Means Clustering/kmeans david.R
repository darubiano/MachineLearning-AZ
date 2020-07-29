# Clustering con k-means

# Importar los datos
dataset = read.csv("Mall_Customers.csv")
x = dataset[,4:5]

# Metodo del codo
#set.seed(6)
wcss = vector()

for (i in 1:10) {
  wcss[i] = sum(kmeans(x, i, iter.max = 300)$withinss)
}
plot(1:10, wcss, type = 'b', main = "Metodo del codo",
     xlab = "Numero de clusters (k)", ylab = "WCSS(k)")


#Aplicar el algoritmo de k-means con k optimo
#set.seed(29)
kmeans = kmeans(x,5, iter.max = 300)

#Visualizacion de los cluesters
#install.packages("cluster")
#library(cluster)
clusplot(x, kmeans$cluster, lines = 0, shade = TRUE,color = TRUE,
         labels = 4, plotchar = FALSE, span = TRUE, main = "CLustering de clientes",
         xlab ="Ingresos anuales", ylab = "Puntuacion (1-100)")




