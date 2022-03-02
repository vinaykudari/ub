library(kernlab)
library(ggplot2)
library(lattice)
library(caret)

data("spam")
data = spam[
  ,
  c(
    'charExclamation',
    'remove',
    'hp',
    'charDollar',
    'capitalAve',
    'type'
  )
]
set.seed(123)
train_idx = createDataPartition(data$type, p=0.75, list=FALSE)
train_set = data[train_idx, ]
test_set = data[-train_idx, ]

knn = train(
  data=train_set,
  type~.,
  method='knn',
  preProcess= c('center', 'scale'),
  tuneLength=20,
)
knn

