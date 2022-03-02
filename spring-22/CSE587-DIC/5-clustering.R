library(data.table)
library(ggplot2)
library(cluster)

data = fread('data/market.csv')
data = data[, 4:5]
set.seed(7)

wcss = vector()
for (i in 1:10) {
  wcss[i] = sum(kmeans(x=data, centers = i)$withinss)
}

plot(
  x=1:10,
  y=wcss,
  type='b',
  main='Elbow graph',
  xlab='Clusters',
  ylab='WCSS',
  )

set.seed(13)

cust_cluster = kmeans(
  x = data, 
  centers = 5,
  iter.max = 500,
  nstart = 20,
)

clusplot(
  x = data,
  clus = cust_cluster$cluster,
  lines = 0,
  shade = TRUE,
  color = TRUE,
  labels = 2,
  plotchar = FALSE,
  span = TRUE,
  main = 'Clusters',
)

