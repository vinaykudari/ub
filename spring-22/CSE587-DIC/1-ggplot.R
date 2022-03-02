library('data.table')
library('ggplot2')

insurance = fread('data/insurance.csv')
summary(insurance)

ggplot(data=insurance,
       aes(x=age, y=charges, color=smoker))+
  geom_point()+
  geom_smooth()

ggplot(
  data=insurance,
  aes(x=charges, fill=smoker)
)+ geom_histogram(
  binwidth = 3000,
  color='white',
  alpha=0.70,
  )

ggplot(
  data=insurance,
  aes(x=region, y=charges, fill=region)
)+ geom_boxplot(
  alpha=0.70,
)

ggplot(
  data=insurance,
  aes(x=as.factor(children), fill=as.factor(children))
)+ geom_bar()+ 
  xlab('N Children')+
  ylab('Cases')+
  ggtitle('Data')

ggplot(
  data=insurance,
  aes(x=bmi, fill=sex)
)+ geom_density(
  alpha=0.60,
)

ggplot(
  data=insurance,
  aes(x=bmi, y=charges)
)+ geom_point()+
  facet_grid(sex~region)
