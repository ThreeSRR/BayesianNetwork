import numpy as np
import pandas as pd
from functools import reduce

## Function to create a conditional probability table
## Conditional probability is of the form p(x1 | x2, ..., xk)
## varnames: vector of variable names (strings) first variable listed 
##           will be x_i, remainder will be parents of x_i, p1, ..., pk
## probs: vector of probabilities for the flattened probability table
## outcomesList: a list containing a vector of outcomes for each variable
## factorTable is in the type of pandas dataframe
## See the test file for examples of how this function works
def readFactorTable(varnames, probs, outcomesList):
    factorTable = pd.DataFrame({'probs': probs})

    totalfactorTableLength = len(probs)
    numVars = len(varnames)

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(totalfactorTableLength / (k * numLevs))
        k = k * numLevs

    return factorTable

## Build a factorTable from a data frame using frequencies
## from a data frame of data to generate the probabilities.
## data: data frame read using pandas read_csv
## varnames: specify what variables you want to read from the table
## factorTable is in the type of pandas dataframe
def readFactorTablefromData(data, varnames):
    numVars = len(varnames)
    outcomesList = []

    for i in range(0, numVars):
        name = varnames[i]
        outcomesList = outcomesList + [list(set(data[name]))]

    lengths = list(map(lambda x: len(x), outcomesList))
    m = reduce(lambda x, y: x * y, lengths)
   
    factorTable = pd.DataFrame({'probs': np.zeros(m)})

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(m / (k * numLevs))
        k = k * numLevs

    numLevels = len(outcomesList[0])

    # creates the vector called fact to index probabilities 
    # using matrix multiplication with the data frame
    fact = np.zeros(data.shape[1])
    lastfact = 1
    for i in range(len(varnames) - 1, -1, -1):
        fact = np.where(np.isin(list(data), varnames[i]), lastfact, fact)
        lastfact = lastfact * len(outcomesList[i])

    # Compute unnormalized counts of subjects that satisfy all conditions
    a = (data - 1).dot(fact) + 1
    for i in range(0, m):
        factorTable.at[i,'probs'] = sum(a == (i+1))

    # normalize the conditional probabilities
    skip = int(m / numLevels)
    for i in range(0, skip):
        normalizeZ = 0
        for j in range(i, m, skip):
            normalizeZ = normalizeZ + factorTable['probs'][j]
        for j in range(i, m, skip):
            if normalizeZ != 0:
                factorTable.at[j,'probs'] = factorTable['probs'][j] / normalizeZ

    return factorTable


## Join of two factors
## factor1, factor2: two factor tables
##
## Should return a factor table that is the join of factor 1 and 2.
## You can assume that the join of two factors is a valid operation.
## Hint: You can look up pd.merge for mergin two factors
def joinFactors(factor1, factor2):
    
    f1 = pd.DataFrame.copy(factor1)
    f2 = pd.DataFrame.copy(factor2)
    f1['commom'] = 1
    f2['commom'] = 1

    intersection = list((f1.columns).intersection(f2.columns))
    intersection.remove('probs')

    joinFactor = pd.merge(f1, f2, on=intersection, how='outer')
    joinFactor['probs_x'] *= joinFactor['probs_y']
    joinFactor = joinFactor.drop(columns=['probs_y','commom'], axis=1)
    joinFactor = joinFactor.rename(columns={'probs_x' : 'probs'})

    return joinFactor

## Marginalize a variable from a factor
## table: a factor table in dataframe
## hiddenVar: a string of the hidden variable name to be marginalized
##
## Should return a factor table that marginalizes margVar out of it.
## Assume that hiddenVar is on the left side of the conditional.
## Hint: you can look can pd.groupby
def marginalizeFactor(factorTable, hiddenVar):
    
    factor = pd.DataFrame.copy(factorTable)

    if hiddenVar not in list(factor.columns):
        return factor

    factor = factor.drop(columns=hiddenVar, axis=1)
    var = list(factor.columns)
    var.remove('probs')

    if not var:
        return factor
    else:
        factor = factor.groupby(var, as_index=False).sum()

    return factor

## Marginalize a list of variables 
## bayesnet: a list of factor tables and each table in dataframe type
## hiddenVar: a string of the variable name to be marginalized
##
## Should return a Bayesian network containing a list of factor tables that results
## when the list of variables in hiddenVar is marginalized out of bayesnet.
def marginalizeNetworkVariables(bayesNet, hiddenVar):
  
    if isinstance(hiddenVar, str):
        hiddenVar = [hiddenVar]

    if not bayesNet or not hiddenVar:
        return bayesNet

    marginalizeBayesNet = bayesNet.copy()

    for var in hiddenVar:
        tmp = []
        tmpfactor = None
        for factor in marginalizeBayesNet:
            if var in factor.columns:
                tmpfactor = factor if tmpfactor is None else joinFactors(factor, tmpfactor)
            else:
                tmp.append(factor)
        if tmpfactor is not None:
            tmp.append(marginalizeFactor(tmpfactor, var))
        marginalizeBayesNet = tmp.copy()

    return marginalizeBayesNet

## Update BayesNet for a set of evidence variables
## bayesNet: a list of factor and factor tables in dataframe format
## evidenceVars: a vector of variable names in the evidence list
## evidenceVals: a vector of values for corresponding variables (in the same order)
##
## Set the values of the evidence variables. Other values for the variables
## should be removed from the tables. You do not need to normalize the factors
def evidenceUpdateNet(bayesNet, evidenceVars, evidenceVals):

    if isinstance(evidenceVars, str):
        evidenceVars = [evidenceVars]
    if isinstance(evidenceVals, str):
        evidenceVals = [evidenceVals]

    updatedBayesNet = bayesNet.copy()
    for idx in range(len(evidenceVars)):
        variable = evidenceVars[idx]
        value = int(evidenceVals[idx])
        tmpnet = updatedBayesNet.copy()
        updatedBayesNet = []

        for factorTable in tmpnet:
            if variable in factorTable.columns:
                factorTable = factorTable[factorTable[variable]==value]
            updatedBayesNet.append(factorTable)

    return updatedBayesNet


## Run inference on a Bayesian network
## bayesNet: a list of factor tables and each table iin dataframe type
## hiddenVar: a string of the variable name to be marginalized
## evidenceVars: a vector of variable names in the evidence list
## evidenceVals: a vector of values for corresponding variables (in the same order)
##
## This function should run variable elimination algorithm by using 
## join and marginalization of the sets of variables. 
## The order of the elimiation can follow hiddenVar ordering
## It should return a single joint probability table. The
## variables that are hidden should not appear in the table. The variables
## that are evidence variable should appear in the table, but only with the single
## evidence value. The variables that are not marginalized or evidence should
## appear in the table with all of their possible values. The probabilities
## should be normalized to sum to one.
def inference(bayesNet, hiddenVar, evidenceVars, evidenceVals): 

    if not bayesNet:
        return bayesNet

    inferenceNet = bayesNet.copy()

    inferenceNet = evidenceUpdateNet(inferenceNet, evidenceVars, evidenceVals)
    inferenceNet = marginalizeNetworkVariables(inferenceNet, hiddenVar=hiddenVar)

    length = len(inferenceNet)
    if length == 1:
        factor = inferenceNet[0]
    else:
        factor = inferenceNet[0]
        for idx in range(1, length):
            factor = joinFactors(factor, inferenceNet[idx])

    norm = sum(list(factor['probs']))
    factor['probs'] /= norm

    return factor




