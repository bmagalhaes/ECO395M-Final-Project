library(tidyverse)
library(haven)
library(stargazer)
library(lmtest)
library(lfe)
library(xtable)
library(tidytable)
library(grid)
library(car)

texas = read.csv("https://raw.githubusercontent.com/bmagalhaes/ECO395M-Final-Project/master/texas.csv")

texas$post = ifelse(texas$statefip == 48 & texas$year >= 1993, 1, 0)

texas = texas %>%
  mutate(year_fixed = as_factor(year)) %>%
  mutate(statefip = as_factor(statefip))

fe_lm = felm(bmprison ~ post + alcohol + aidscapita + income + ur + poverty + 
               black + perc1519 | year_fixed + statefip | 0 | 0, data=texas)

stargazer(fe_lm, type = 'text',
          covariate.labels = "Prison capacity expansion",
          omit = c("alcohol", "aidscapita", "income", "ur", "poverty",
                   "black", "perc1519"),
          dep.var.labels = "Black Male Prisoners",
            add.lines = list(c("State Fixed effects", "Yes"),
                           c("Year Fixed effects", "Yes")),
          omit.stat = "ser")



texas = texas %>%
  mutate(time_til = ifelse(statefip == 48, year - 1993, 0),
    lead1 = case_when(time_til == -1 ~ 1, TRUE ~ 0),
    lead2 = case_when(time_til == -2 ~ 1, TRUE ~ 0),
    lead3 = case_when(time_til == -3 ~ 1, TRUE ~ 0),
    lead4 = case_when(time_til == -4 ~ 1, TRUE ~ 0),
    lead5 = case_when(time_til == -5 ~ 1, TRUE ~ 0),
    lead6 = case_when(time_til == -6 ~ 1, TRUE ~ 0),
    lead7 = case_when(time_til == -7 ~ 1, TRUE ~ 0),
    lead8 = case_when(time_til == -8 ~ 1, TRUE ~ 0),
    lag0 = case_when(time_til == 0 ~ 1, TRUE ~ 0),
    lag1 = case_when(time_til == 1 ~ 1, TRUE ~ 0),
    lag2 = case_when(time_til == 2 ~ 1, TRUE ~ 0),
    lag3 = case_when(time_til == 3 ~ 1, TRUE ~ 0),
    lag4 = case_when(time_til == 4 ~ 1, TRUE ~ 0),
    lag5 = case_when(time_til == 5 ~ 1, TRUE ~ 0),
    lag6 = case_when(time_til == 6 ~ 1, TRUE ~ 0),
    lag7 = case_when(time_til == 7 ~ 1, TRUE ~ 0)
  )

fe_leads = felm(bmprison ~ lead1 + lead2 + lead3 + lead4 + lead5 + lead6 + lead7
                + lead8 + lag1 + lag2 + lag3 + lag4 + lag5 + lag6 + lag7
                + alcohol + aidscapita + income + ur + poverty + black + perc1519
                | year_fixed + statefip | 0 | 0, data=texas)

plot_order <- c("lead8", "lead7", "lead6", "lead5", "lead4", "lead3", "lead2",
                "lead1", "lag1", "lag2", "lag3", "lag4", "lag5", 
                "lag6", "lag7")

leadslags_bmprison <- tibble(
  sd = c(fe_leads$se[plot_order], 0),
  mean = c(coef(fe_leads)[plot_order], 0),
  label = c(-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 0)
)

round(fe_lm[["coefficients"]], 2)
round(fe_lm[["se"]], 2)
text_asmrh = "DD Coefficient = 28454.82 (se = 1235.93)"

plot_1 = leadslags_bmprison %>%
  ggplot(aes(x = label, y = mean, ymin = mean-1.96*sd, ymax = mean+1.96*sd)) +
  geom_point(color = "dark blue") +
  geom_line(aes(y = mean+1.96*sd, x=label), colour = 'dark blue', linetype = "dashed") +
  geom_line(aes(y = mean-1.96*sd, x=label), colour = 'dark blue', linetype = "dashed")+
  geom_line(color = "dark blue") +
  geom_ribbon(alpha = 0.4, fill = "light blue") +
  theme_bw() +
  xlab("Years Relative to Prison Expansion") +
  ylab("Black Male Prison") +
  geom_hline(yintercept = 0, color = "dark grey", size = 0.8) +
  geom_vline(xintercept = 0, color = "dark grey", size = 0.8) +
  geom_hline(yintercept = 28454.8160032, color = "red", size = 1.2) +
  scale_x_continuous(breaks= c(-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7)) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank()) +
  annotation_custom(grid.text(text_asmrh, x=0.30,  y=0.88,
                              gp=gpar(col="black", fontsize=8, fontface="bold")))
plot_1

f_test = linearHypothesis(fe_leads, c("lead1 = lead2",
                                                "lead2 = lead3",
                                                "lead3 = lead4",
                                                "lead4 = lead5",
                                                "lead5 = lead6",
                                                "lead6 = lead7",
                                                "lead7 = 0"))

stargazer(f_test, flip = TRUE, type = 'text', summary.stat = c("mean"))

install.packages("dummies")
library(dummies)

texas_dummies <- dummy.data.frame(texas, names = c("state") , sep = ".")

texas$trend = texas$year - 1985

