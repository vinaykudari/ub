library(data.table)

data = fread('data/50_Startups.csv')
data$'R&D Spend'[c(3, 21, 46)] = NA
data$Administration[c(27)] = NA
data$`Marketing Spend`[c(31)] = NA

write.csv(data, 'data/strat_50_startups.csv', row.names = FALSE)
startups = fread('data/strat_50_startups.csv')

summary(startups)

sub_1 = startups[, 2]
sub_2 = startups[, c(1, 5)]
sub_3 = startups[, 'Administration']
sub_4 = startups[, c('R&D Spend', 'Profit')]

r_sub_1 = startups[5:10, ]
r_sub_2 = startups[c(2:3, 4:9, 22:29, 33, 45:50)]

c_sub_1 = startups[c(2:3, 4:9, 22:29, 33, 45:50), c('R&D Spend', 'Profit')]
c_sub_2 = startups[seq(1, 50, 4), c('R&D Spend', 'Profit')]
c_sub_3 = startups[startups$Profit >= 140000, c('R&D Spend', 'Profit')]
c_sub_4 = startups[startups$Administration >= 140000 & startups$State=='Florida']
c_sub_4

sort(startups$Profit)
ordered = startups[order(State, Profit)]
ordered

startups_new = startups
startups_new$'new col' = rnorm(50, mean=0, sd=1)
startups_new

x1 = data.frame(var1=rnorm(3, mean=0, sd=1), var2=rnorm(3, mean=5, sd=3))
x2 = data.frame(var1=rnorm(3, mean=0, sd=1), var2=rnorm(3, mean=5, sd=3))
x = cbind(x1, x2)
x_r = rbind(x1, x2)

colnames(x) = c('v1', 'v2', 'v3', 'v4')
u_states = unique(startups$State)
startups$State = factor(startups$State, levels=u_states, labels=c(1, 2, 3))
summary(startups)

# @TODO melt, dcast, merge
