library(tidyverse)
library(gamlr)

mydata<-minwage1617[,-c(2:6,8:11)]#delete identifier variables
t = mydata$YEAR-2016
mydata$t <- t
mydata$minwage_change <- ifelse(mydata$STATEFIP == 4|mydata$STATEFIP == 8|mydata$STATEFIP == 23|mydata$STATEFIP == 44|mydata$STATEFIP == 53, 1,0)
#choose 5 states that had minimum wage change only in 2017(=1), others have no change in both 2016 and 2017(=0)
mydata <- mydata[!(mydata$AGE<18),]
mydata <- mydata[!(mydata$AGE>60),]#restric our dataset btw age 18 and 60
mydata$educ <- ifelse(mydata$EDUCD<13, 0,  
              ifelse(mydata$EDUCD==14, 1,
              ifelse(mydata$EDUCD==15,2,
              ifelse(mydata$EDUCD==16,3,
              ifelse(mydata$EDUCD==17,4, 
              ifelse(mydata$EDUCD==22,5,
              ifelse(mydata$EDUCD==23, 6,
              ifelse(mydata$EDUCD==25, 7,
              ifelse(mydata$EDUCD==26, 8,
              ifelse(mydata$EDUCD==30, 9,
              ifelse(mydata$EDUCD==40, 10,
              ifelse(mydata$EDUCD==50, 11, 12))))))))))))
#create "education" variable and assign years from 0 to 12          
mydata$employed <- ifelse(mydata$EMPSTAT == 1, 1, 0)#=1 if employed
mydata$laborforce <- ifelse(mydata$EMPSTAT == 1|mydata$EMPSTAT==2, 1, 0)#=1 if in laborforce          
# I tried to make occupation dummy variables but there are more than 20 categories so I gave it up
mydata$wwly <- ifelse(mydata$WKSWORK2>0 & mydata$CLASSWKR==2 & mydata$CLASSWKRD<29, 1, 0)
# =1 if wage worker last year, which means not self-employed, not an unpaid worker, worked at least 1 week last year
mydata$midpoint <- ifelse(mydata$WKSWORK2==0, 0,
                  ifelse(mydata$WKSWORK2==1, 7,
                  ifelse(mydata$WKSWORK2==2, 20,
                  ifelse(mydata$WKSWORK2==3, 33,
                  ifelse(mydata$WKSWORK2==4, 43.5,
                  ifelse(mydata$WKSWORK2==5, 48.5, 51))))))
#midpoint variables is the midpoint values of the interval for weeks worked last year
#the data doesn't have continuous value but it only gives us categorical values such as '=1 if 1-13weeks'
mydata$annualhw = mydata$midpoint*mydata$UHRSWORK
#generate "annual hours worked" variable = midpoint of weeks worked * hours worked per week
mydata$nonlaborinc = mydata$FTOTINC - mydata$INCWAGE
#create "nonlabor income" which is "total family income - own wage income for one year"
mydata$hourlywage = mydata$INCWAGE/mydata$annualhw
#generate "hourly wage" = "wage income"/"annual hours worked"

mydata$logannualhw = log(mydata$annualhw)
mydata$lognonlaborinc = log(mydata$nonlaborinc)
mydata$loghourlywage = log(mydata$hourlywage)
#generate log values

mydata$agesq= (mydata$AGE)^2
mydata <- mydata[,-c(1,9,11:15, 17:23,30:33)]#delete variables we won't use
workdata <- subset(mydata, wwly==1)
workdata <- subset(workdata, !loghourlywage==-Inf)
workdata <- subset(workdata, !lognonlaborinc==-Inf)
workdata = workdata %>%
  mutate(STATEFIP = as_factor(STATEFIP)) %>%
  mutate(SEX = as_factor(SEX)) %>%
  mutate(RACE = as_factor(RACE)) %>%
  mutate(MARST = as_factor(MARST)) %>% 
  mutate(HISPAN = as_factor(HISPAN)) %>% 
  mutate(OCC = as_factor(OCC))

controls = data.frame(workdata[,c(2:9,12,17,19)])
y = workdata$logannualhw
d = workdata$minwage_change
t = workdata$t

orig = glm(y~ d+t+., data=controls)
summary(orig)$coef
summary(orig)$coef['d',] %>% round(3)

interact = glm(y ~ d + (.^2)*t, data=controls)
# unable to use interactions because the data is too huge

x = sparse.model.matrix(~ .*t, data=controls)[,-1]
dim(x)

naive = gamlr(cbind(d,x),y)
coef(naive)["d",]

treat = gamlr(x,d,lambda.min.ratio=1e-4)
plot(treat)#a useless plot for binary d
coef(treat)#also useless I guess

dhat = predict(treat, x, type="response") %>% drop
plot(dhat,d,bty="n",pch=21,bg=8) 
cor(drop(dhat),d)^2

# simple linear model: significance to dres
dres = drop(d - dhat)
causal = lm(y ~ dres + dhat)
summary(causal)

causal2 = gamlr(cbind(d, dhat, x),y,free=2,lmr=1e-4)
coef(causal2)["d",]
