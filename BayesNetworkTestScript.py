#!/usr/bin/env python3

from BayesianNetworks import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#############################
## Example Tests from Bishop Pattern recognition textbook on page 377
#############################
BatteryState = readFactorTable(['battery'], [0.9, 0.1], [[1, 0]])
FuelState = readFactorTable(['fuel'], [0.9, 0.1], [[1, 0]])
GaugeBF = readFactorTable(['gauge', 'battery', 'fuel'], [0.8, 0.2, 0.2, 0.1, 0.2, 0.8, 0.8, 0.9], [[1, 0], [1, 0], [1, 0]])

carNet = [BatteryState, FuelState, GaugeBF] # carNet is a list of factors 
## Notice that different order of operations give the same answer
## (rows/columns may be permuted)
joinFactors(joinFactors(BatteryState, FuelState), GaugeBF)
joinFactors(joinFactors(GaugeBF, FuelState), BatteryState)

marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'gauge')
joinFactors(marginalizeFactor(GaugeBF, 'gauge'), BatteryState)

joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState)
marginalizeFactor(joinFactors(joinFactors(GaugeBF, FuelState), BatteryState), 'battery')

marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'gauge')
marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'fuel')

evidenceUpdateNet(carNet, 'fuel', '1')
evidenceUpdateNet(carNet, ['fuel', 'battery'], ['1', '0'])

## Marginalize must first combine all factors involving the variable to
## marginalize. Again, this operation may lead to factors that aren't
## probabilities.
marginalizeNetworkVariables(carNet, 'battery') ## this returns back a list
marginalizeNetworkVariables(carNet, 'fuel') ## this returns back a list
marginalizeNetworkVariables(carNet, ['battery', 'fuel'])

# inference
print("inference starts")
print(inference(carNet, ['battery', 'fuel'], [], []) )        ## chapter 8 equation (8.30)
print(inference(carNet, ['battery'], ['fuel'], [0]))           ## chapter 8 equation (8.31)
print(inference(carNet, ['battery'], ['gauge'], [0]))          ##chapter 8 equation  (8.32)
print(inference(carNet, [], ['gauge', 'battery'], [0, 0]))    ## chapter 8 equation (8.33)
print("inference ends")
###########################################################################
#RiskFactor Data Tests
###########################################################################
riskFactorNet = pd.read_csv('RiskFactorsData.csv')

# Create factors

income      = readFactorTablefromData(riskFactorNet, ['income'])
smoke       = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
exercise    = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
long_sit    = readFactorTablefromData(riskFactorNet, ['long_sit', 'income'])
stay_up     = readFactorTablefromData(riskFactorNet, ['stay_up', 'income'])
bmi         = readFactorTablefromData(riskFactorNet, ['bmi', 'income'])
diabetes    = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi'])

## you need to create more factor tables

risk_net = [income, smoke, long_sit, stay_up, exercise, bmi, diabetes]
print("income dataframe is ")
print(income)
factors = riskFactorNet.columns

# example test p(diabetes|smoke=1,exercise=2,long_sit=1)

margVars = list(set(factors) - {'diabetes', 'smoke', 'exercise','long_sit'})
obsVars  = ['smoke', 'exercise','long_sit']
obsVals  = [1, 2, 1]

p = inference(risk_net, margVars, obsVars, obsVals)
print(p)


### Please write your own test scrip similar to  the previous example 
###########################################################################
#HW4 test scripts start from here
###########################################################################

riskFactorNet = pd.read_csv('RiskFactorsData.csv')

factors = riskFactorNet.columns

income = readFactorTablefromData(riskFactorNet, ['income'])
exercise = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
long_sit = readFactorTablefromData(riskFactorNet, ['long_sit', 'income'])
stay_up = readFactorTablefromData(riskFactorNet, ['stay_up', 'income'])
smoke = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
bmi = readFactorTablefromData(riskFactorNet, ['bmi', 'income', 'exercise', 'long_sit'])
bp = readFactorTablefromData(riskFactorNet, ['bp', 'exercise', 'long_sit', 'income', 'stay_up', 'smoke'])
cholesterol = readFactorTablefromData(riskFactorNet, ['cholesterol', 'exercise', 'stay_up', 'income', 'smoke'])
diabetes = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi'])
stroke = readFactorTablefromData(riskFactorNet, ['stroke', 'bmi', 'bp', 'cholesterol'])
attack = readFactorTablefromData(riskFactorNet, ['attack', 'bmi', 'bp', 'cholesterol'])
angina = readFactorTablefromData(riskFactorNet, ['angina', 'bmi', 'bp', 'cholesterol'])

risk_net = [income, exercise, long_sit, stay_up, smoke, bmi, bp, cholesterol, diabetes, stroke, attack, angina]

# Question1 ------------------------------------------------------------------------------------------------------

print('Question1 -------------------------------------------')

size = 0
for factor in risk_net:
    size += len(factor)
print('size of the network is: %d' % (size))

# Question2 ------------------------------------------------------------------------------------------------------

print('Question2 -------------------------------------------')

healthoutcomes = ['diabetes', 'stroke', 'attack', 'angina']

# bad habits
# smoke = 1, exercise = 2, long_sit = 1, stay_up = 1

for health in healthoutcomes:
    margVars = list(set(factors) - {health, 'smoke', 'exercise', 'long_sit', 'stay_up'})
    obsVars  = ['smoke', 'exercise', 'long_sit', 'stay_up']
    obsVals  = [1, 2, 1, 1]
    p = inference(risk_net, margVars, obsVars, obsVals)
    print('The probability of %s if I have bad habits is: \n' % (health), p, '\n')

# good habits
# smoke = 2, exercise = 1, long_sit = 2, stay_up = 2

for health in healthoutcomes:
    margVars = list(set(factors) - {health, 'smoke', 'exercise', 'long_sit', 'stay_up'})
    obsVars  = ['smoke', 'exercise', 'long_sit', 'stay_up']
    obsVals  = [2, 1, 2, 2]
    p = inference(risk_net, margVars, obsVars, obsVals)
    print('The probability of %s if I have good habits is: \n' % (health), p, '\n')

# poor health
# bp = 1, cholesterol = 1, bmi = 3

for health in healthoutcomes:
    margVars = list(set(factors) - {health, 'bp', 'cholesterol', 'bmi'})
    obsVars  = ['bp', 'cholesterol', 'bmi']
    obsVals  = [1, 1, 3]
    p = inference(risk_net, margVars, obsVars, obsVals)
    print('The probability of %s if I have poor health is: \n' % (health), p, '\n')

# good health
# bp = 3, cholesterol = 2, bmi = 2

for health in healthoutcomes:
    margVars = list(set(factors) - {health, 'bp', 'cholesterol', 'bmi'})
    obsVars  = ['bp', 'cholesterol', 'bmi']
    obsVals  = [3, 2, 2]
    p = inference(risk_net, margVars, obsVars, obsVals)
    print('The probability of %s if I have good health is: \n' % (health), p, '\n')

# Question3 ------------------------------------------------------------------------------------------------------

print('Question3 -------------------------------------------')

healthoutcomes = ['diabetes', 'stroke', 'attack', 'angina']
probs = {}

for health in healthoutcomes:
    result = {}
    for level in range(1, 9):
        obsVars = ['income']
        obsVals = str(level)
        
        margVars = list(set(factors) - {health, 'income'})
        p = inference(risk_net, margVars, obsVars, obsVals)
        print('The probability of %s with income level %d is: \n' % (health, level), p)
        result[level] = float(p[p[health]==1]['probs'])
        
    probs[health] = result

# plot probability of health outcome given income levelS

for health, prob in probs.items():
    print('probability of %s given income status is saved. ' % (health))
    x, y = [], []
    for level, probability in prob.items():
        x.append(level)
        y.append(probability)
    plt.clf()
    plt.plot(x, y)
    plt.ylabel('probability') 
    plt.xlabel('income level')
    plt.savefig('%s_given_income.png' % (health))


# Question4 ------------------------------------------------------------------------------------------------------

print('Question4 -------------------------------------------')

income = readFactorTablefromData(riskFactorNet, ['income'])
exercise = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
long_sit = readFactorTablefromData(riskFactorNet, ['long_sit', 'income'])
stay_up = readFactorTablefromData(riskFactorNet, ['stay_up', 'income'])
smoke = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
bmi = readFactorTablefromData(riskFactorNet, ['bmi', 'income', 'exercise', 'long_sit'])
bp = readFactorTablefromData(riskFactorNet, ['bp', 'exercise', 'long_sit', 'income', 'stay_up', 'smoke'])
cholesterol = readFactorTablefromData(riskFactorNet, ['cholesterol', 'exercise', 'stay_up', 'income', 'smoke'])

# add edges from smoke to each outcome and edges from exercise to each outcome
diabetes_with_smoke_exercise = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi', 'smoke', 'exercise'])
stroke_with_smoke_exercise = readFactorTablefromData(riskFactorNet, ['stroke', 'bmi', 'bp', 'cholesterol', 'smoke', 'exercise'])
attack_with_smoke_exercise = readFactorTablefromData(riskFactorNet, ['attack', 'bmi', 'bp', 'cholesterol', 'smoke', 'exercise'])
angina_with_smoke_exercise = readFactorTablefromData(riskFactorNet, ['angina', 'bmi', 'bp', 'cholesterol', 'smoke', 'exercise'])

second_risk_net = [income, exercise, long_sit, stay_up, smoke, bmi, bp, cholesterol, diabetes_with_smoke_exercise, stroke_with_smoke_exercise, attack_with_smoke_exercise, angina_with_smoke_exercise]

healthoutcomes = ['diabetes', 'stroke', 'attack', 'angina']

# bad habits
# smoke = 1, exercise = 2, long_sit = 1, stay_up = 1

for health in healthoutcomes:
    margVars = list(set(factors) - {health, 'smoke', 'exercise', 'long_sit', 'stay_up'})
    obsVars  = ['smoke', 'exercise', 'long_sit', 'stay_up']
    obsVals  = [1, 2, 1, 1]
    p = inference(second_risk_net, margVars, obsVars, obsVals)
    print('The probability of %s if I have bad habits is: \n' % (health), p, '\n')

# good habits
# smoke = 2, exercise = 1, long_sit = 2, stay_up = 2

for health in healthoutcomes:
    margVars = list(set(factors) - {health, 'smoke', 'exercise', 'long_sit', 'stay_up'})
    obsVars  = ['smoke', 'exercise', 'long_sit', 'stay_up']
    obsVals  = [2, 1, 2, 2]
    p = inference(second_risk_net, margVars, obsVars, obsVals)
    print('The probability of %s if I have good habits is: \n' % (health), p, '\n')

# poor health
# bp = 1, cholesterol = 1, bmi = 3

for health in healthoutcomes:
    margVars = list(set(factors) - {health, 'bp', 'cholesterol', 'bmi'})
    obsVars  = ['bp', 'cholesterol', 'bmi']
    obsVals  = [1, 1, 3]
    p = inference(second_risk_net, margVars, obsVars, obsVals)
    print('The probability of %s if I have poor health is: \n' % (health), p, '\n')

# good health
# bp = 3, cholesterol = 2, bmi = 2

for health in healthoutcomes:
    margVars = list(set(factors) - {health, 'bp', 'cholesterol', 'bmi'})
    obsVars  = ['bp', 'cholesterol', 'bmi']
    obsVals  = [3, 2, 2]
    p = inference(second_risk_net, margVars, obsVars, obsVals)
    print('The probability of %s if I have good health is: \n' % (health), p, '\n')


# Question5 ------------------------------------------------------------------------------------------------------

print('Question5 -------------------------------------------')

# add edge from diabetes to stroke
stroke_with_diabetes = readFactorTablefromData(riskFactorNet, ['stroke', 'bmi', 'bp', 'cholesterol', 'smoke', 'exercise', 'diabetes'])
third_risk_net = [income, exercise, long_sit, stay_up, smoke, bmi, bp, cholesterol, diabetes_with_smoke_exercise, stroke_with_diabetes, attack_with_smoke_exercise, angina_with_smoke_exercise]

obsVars = ['diabetes']
margVars = list(set(factors) - {'stroke', 'diabetes'})

# second network
print('second network: ')

for obsVals in ['1', '3']:
    p = inference(second_risk_net, margVars, obsVars, obsVals)
    probability = float(p[p['stroke']==1]['probs'])
    print('probability of stroke level 1 given diabetes level %s is %f' % (obsVals, probability))
    
# third network: Adding an edge from diabetes to stroke
print('third network: Adding an edge from diabetes to stroke')

for obsVals in ['1', '3']:
    p = inference(third_risk_net, margVars, obsVars, obsVals)
    probability = float(p[p['stroke']==1]['probs'])
    print('probability of stroke level 1 given diabetes level %s is %f' % (obsVals, probability))
