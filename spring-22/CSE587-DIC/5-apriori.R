library(arules)
library(arulesViz)

data('Groceries')
summary(Groceries)

inspect(head(Groceries, 5))
itemFrequencyPlot(x=Groceries, topN=15)
rules = apriori(
  Groceries,
  parameter = list(
    support = 0.01, 
    confidence = 0.4
    )
  )
inspect(head(sort(rules, by='lift'), 10))
plot(rules, method = 'two-key plot')
plot(rules, method = 'grouped')        
plot(rules, method = 'graph')

subrules = head(rules, 10, by='lift')
plot(subrules, method='graph')
