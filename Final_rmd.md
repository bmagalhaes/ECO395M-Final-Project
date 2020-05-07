ECO 395M: Final Project
=======================

Bernardo Arreal Magalhaes - UTEID ba25727

Adhish Luitel - UTEID al49674

Ji Heon Shim - UTEID js93996

Introduction
------------

Empirical economists are very often interested in estimating the impact
of certain events or policies on a particular outcome. Wooldridge (2013)
describes the effectiveness applications of Differences-in-Differences
methodology when the data arise from a natural experiment. This kind of
experiment occurs when an exogenous event changes the environment in
which individuals opperate, and require observations of both treatment
and control group before and after the change.

This methodology is particularly powerfull since it neutralizes
unobserved, but fixed, omitted variables (Angrist and Pischke, 2018).
Nonetheless, it relies on a quite strong – and unfortunately not
testable – assumption that the outcome in the multiple
individuals/firms/states included in the analysis share the same trend
over time, which is called parallel trends assumption.

The table below illustrates a simple versionf of the
Differences-in-Differences method and why this assumption is required.
By decreasing the outcome after treatment from the outcome before
treatment for a treated state, the difference (D1) is going to be the
effect caused by the treatment (E) plus a trend (T). This step
neutralizes unobserved factors of a particular state. For a state that
wasn't treated, the difference (D1) before and after treatment is the
trend (T) only. So, if we assume that T is the same for both states, we
can decrease T, that was measured from the control state, from T + E in
order to isolate the causal effect E.

![](https://raw.githubusercontent.com/bmagalhaes/ECO395M-Final-Project/master/4.0-table1.png)

This assumption is not testable because we don’t know what would’ve
happened to the treatment state had it not been treated. What we can do
is to test whether the trends were parallel before the treatment and
argue that, if the assumption holds for the pre-treatment period, there
might be no reason for it not hold in the post-treatment too.

But what if we could predict what would have happened to the treated
state in this alternative world where it wasn’t treated without having
to rely on the parallel trends assumption? In this particular study, we
are going to compare the application of different predictive methods
with the Differences-in-Differences estimator of a research question.

Research topic brief summary
----------------------------

During the 1980s, the state of Texas lost a civil action lawsuit where a
prisoner argued that the state Department of Corrections was engagin in
unconstitutional practices regarding prisoners conditions. The court
ruled in favor of the prisoner, and forced the state to pursue a series
of settlements. Among other orders, the counter placed constraints on
the number of inmates allowed per cells. Given this constraint, state
legislators approved a billion dollar prison construction project that
ended up doubling the state’s capacity within 3 years.

The nature of this expansion allow us to use it as a natural experiment
to estimate the effect of prison expansion on incarceration. The
baseline model we will be using is bmprison ~ alcohol + aidscapita +
income + ur + poverty + black + perc1519 + year + state +
year\_after1993\*state\_texas (analogus to the original analysis)
