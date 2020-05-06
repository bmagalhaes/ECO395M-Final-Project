mydata<-minwage1617[,-c(2:6,8:11)]#delete identifier variables
data_az = subset(mydata, STATEFIP != 53 & STATEFIP != 8 & STATEFIP != 23
                 & STATEFIP != 44)

# creating post variable, which is equivalent to interacting treated state with year of treatment
data_az$post = ifelse(data_az$STATEFIP == 4 & data_az$YEAR == 2017, 1, 0)
data_az <- data_az[!(data_az$AGE<18),]
data_az <- data_az[!(data_az$AGE>60),]
data_az$educ <- ifelse(data_az$EDUCD<13, 0,  
                       ifelse(data_az$EDUCD==14, 1,
                       ifelse(data_az$EDUCD==15,2,
                       ifelse(data_az$EDUCD==16,3,
                       ifelse(data_az$EDUCD==17,4, 
                       ifelse(data_az$EDUCD==22,5,
                       ifelse(data_az$EDUCD==23, 6,
                       ifelse(data_az$EDUCD==25, 7,
                       ifelse(data_az$EDUCD==26, 8,
                       ifelse(data_az$EDUCD==30, 9,
                       ifelse(data_az$EDUCD==40, 10,
                       ifelse(data_az$EDUCD==50, 11, 12))))))))))))

data_az$employed <- ifelse(data_az$EMPSTAT == 1, 1, 0)#=1 if employed
data_az$laborforce <- ifelse(data_az$EMPSTAT == 1|data_az$EMPSTAT==2, 1, 0)#=1 if in laborforce

data_az$wwly <- ifelse(data_az$WKSWORK2>0 & data_az$CLASSWKR==2 & data_az$CLASSWKRD<29, 1, 0)
# =1 if wage worker last year, which means not self-employed, not an unpaid worker, worked at least 1 week last year
data_az$midpoint <- ifelse(data_az$WKSWORK2==0, 0,
                        ifelse(data_az$WKSWORK2==1, 7,
                        ifelse(data_az$WKSWORK2==2, 20,
                        ifelse(data_az$WKSWORK2==3, 33,
                        ifelse(data_az$WKSWORK2==4, 43.5,
                        ifelse(data_az$WKSWORK2==5, 48.5, 51))))))
#midpoint variables is the midpoint values of the interval for weeks worked last year
#the data doesn't have continuous value but it only gives us categorical values such as '=1 if 1-13weeks'
data_az$annualhw = data_az$midpoint*data_az$UHRSWORK
#generate "annual hours worked" variable = midpoint of weeks worked * hours worked per week
data_az$nonlaborinc = data_az$FTOTINC - data_az$INCWAGE
#create "nonlabor income" which is "total family income - own wage income for one year"
data_az$hourlywage = data_az$INCWAGE/data_az$annualhw
#generate "hourly wage" = "wage income"/"annual hours worked"

data_az$logannualhw = log(data_az$annualhw)
data_az$lognonlaborinc = log(data_az$nonlaborinc)
data_az$loghourlywage = log(data_az$hourlywage)
#generate log values

data_az$agesq= (data_az$AGE)^2
workdata <- subset(data_az, wwly==1)
workdata <- subset(workdata, !loghourlywage==-Inf)

#Treating categoricals as factor in order to avoid creating dummies
workdata = workdata %>%
  mutate(YEAR = as_factor(YEAR)) %>%
  mutate(STATEFIP = as_factor(STATEFIP)) %>%
  mutate(SEX = as_factor(SEX)) %>%
  mutate(RACE = as_factor(RACE)) %>%
  mutate(MARST = as_factor(MARST))

fe_lm1 = felm(loghourlywage ~ post + SEX + AGE + agesq + RACE + MARST + NCHILD
              + NCHLT5 | YEAR + STATEFIP | 0 | STATEFIP, data=workdata)
fe_lm2 = felm(logannualhw ~ post + SEX + AGE + agesq + RACE + MARST + NCHILD
              + NCHLT5 | YEAR + STATEFIP | 0 | STATEFIP, data=workdata)

stargazer(fe_lm1, fe_lm2, type = 'text',
          omit = c("SEX", "AGE", "agesq", "RACE", "MARST", "NCHILD", "NCHLT5"),
          covariate.labels = "Minimum Wage Increase",
          dep.var.labels = c("(Log) Hourly Wage", "(Log) Annual Hours Worked"),
          omit.stat = c("ser","rsq", "f"),
          add.lines = list(c("State Fixed effects", "Yes", "Yes"),
                           c("Year Fixed effects", "Yes", "Yes"),
                           c("Controls", "Yes", "Yes")),
          notes = "Standard errors clustered by State")
