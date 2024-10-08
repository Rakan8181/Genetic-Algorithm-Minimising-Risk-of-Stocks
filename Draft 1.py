import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import random


stocks = ["META","AMZN","TSLA","AAPL","GOOG","NFLX","MSFT","NVDA","V","UNH","XOM",
          "LLY","JNJ","WMT","JPM","MA","PG","AVGO","CVX","HD"]
std = []
#print(df.loc[0]) row
#print(df["Open"]) column
def calcReturns(df):
    returns = []
    lst = df["Close"]
    for c in range(len(lst)-1):
        if c != 0:
            ret = (lst[c+1] / lst[c]) - 1
            returns.append(ret)

    return returns

def getReturns(stock):


    file = pd.read_csv("C:\\Users\\44734\\source\\repos\\Toller\\Stocks\\%s.csv"%(stock))
    lst = calcReturns(file)
    return lst


def getStd(stock):
    global stocks
    for c,stock in enumerate(stocks):
        file = pd.read_csv("C:\\Users\\44734\\source\\repos\\Toller\\Stocks\\%s.csv"%(stock))
        lst = calcReturns(file)
    std = np.std(lst)
    return std
    
    


def calcCovariance(stock1,stock2):
    l1 = getReturns(stock1)
    l2 = getReturns(stock2)
    l1.pop(-1)
    l2.pop(-1)
  #  print(*l1)
  #  print()
  #  print(*l2)

    mean1 = sum(l1) / len(l1) #all lengths are 124 anyway
    mean2 = sum(l2) / len(l2)
    
    total = 0
    for c in range(len(l1)): #length is 124
        total += ((l1[c] - mean1) * (l2[c] - mean2))

    
    covariance = total / (len(l1))
    #sample size so divided by sample size - 1
    #intuitively: smaller sample size, means very unlikely to achieve extremes (both high and low) so variance would be smaller, so divide by a smaller number to achieve a larger variance. Proof for why 1 too long.
                  
    return covariance



#finding correlation of aapl and amzn. Example
#std = statistics.stdev(getReturns("AAPL")) * statistics.stdev(getReturns("AAPL"))
#correlation = calcCovariance("AAPL","AAPL") / std
#print(correlation) SHOULD BE 1 but is 0.99931
covarianceMatrix = [[0 for i in range(20)] for j in range(20)]
correlationMatrix = [[0 for i in range(20)] for j in range(20)]

for c1,stock1 in enumerate(stocks):
    for c2,stock2 in enumerate(stocks):
        covariance = round(calcCovariance(stock1,stock2),2) #unround when doing algorithm
        std = statistics.stdev(getReturns(stock1)) * statistics.stdev(getReturns(stock2))
        correlation = round(calcCovariance(stock1,stock2) / std,2)
        covarianceMatrix[c1][c2] = covariance
        correlationMatrix[c1][c2] = correlation
#DISPLAY: correlation / covariance matrix
#print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in covarianceMatrix]))
for i in stocks:
    additionalSpace = 7 - len(i)
    print(i,additionalSpace*" ",end="")
print()
print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in correlationMatrix]))

def calcFitness(lst):#calculate fitness, given a list of 5 stocks e.g [1,2,3,9,10]
    global stocks
    t = 0
    for stockNum in lst:
        for n in range(5):
            stock1 = stocks[stockNum]
         #   print("this",n,lst[n])
            stock2 = stocks[lst[n]]
            t += calcCovariance(stock1,stock2)
    return t


randomSample = [] #generate random 100 combinations of 5 stocks to start genetic algorithm
for i in range(100):
    n = random.sample(range(0,19),5)
    n = [int(i) for i in n]
    n.sort()
    randomSample.append(n)


#dic = {}
fitnessValues = []
combinations = []
for c,lst in enumerate(randomSample):
    fitness = calcFitness(lst)
    fitnessValues.append(fitness)
    combinations.append(lst)


#worstPerformer = dic[max(dic.keys())] #TSLA AAPL GOOG NFLX NVDA 

def merge_replace(lstFitness,lstCombinations): #2 combinations are picked at random then merged; child replaces worst performer
    check = False #randomise two of the 100. cannot be the same
    while check == False:
        n1 = random.randint(0,98)
        n2 = random.randint(0,98)
        if n1 != n2:
            check = True
    l1 = [1 if i in lstCombinations[n1] else 0 for i in range(20)]
    l2 = [1 if i in lstCombinations[n2] else 0 for i in range(20)]
    merged = [0 for i in range(20)] #may be inefficient. Append instead? WS

    check = False
    while check == False: #merge l1 and l2. repeat merging until merged list has 5 1s
        for b in range(20):
            if l1[b] == l2[b]:
                merged[b] = l1[b]
            else:
                n = random.randint(0,1)
                merged[b] = n

        if merged.count(1) == 5:
            new = []
    #        merged = mutate(merged)  #with MUTATION
            for c,bit in enumerate(merged):
                if bit == 1:
                    new.append(c)
            lstCombinations.append(new)
            fitness = calcFitness(new)
            lstFitness.append(fitness)
            check = True
    combined = zip(lstFitness, lstCombinations)
    sorted_combined = sorted(combined, key=lambda x: x[0])
    sortedFitness, sortedCombinations = zip(*sorted_combined)
    sortedFitness = list(sortedFitness)
    sortedCombinations = list(sortedCombinations)
    sortedFitness.pop()
    sortedCombinations.pop()
    return [sortedFitness,sortedCombinations]

    #removing the highest fitness function (highest covariance so highest risk)
    #and its corresponding combination

def mutate(genome): #genome: [1,0,0,1,1,...]
    check = False
    while check == False:
        n1 = random.randint(0,19)
        n2 = random.randint(0,19)
        if genome[n1] != genome[n2]:
            check = True
    genome[n1] = abs(genome[n1] - 1)
    genome[n2] = abs(genome[n2] - 1)
    print(genome.count(1))
    
    return genome

#MAIN

for i in range(999):
   print(i)
   lst = merge_replace(fitnessValues,combinations)
   fitnessValues = lst[0]
   combinations = lst[1]
   if i % 100 == 0:
       x = np.arange(1,101)
       y = np.array(fitnessValues)   
       plt.scatter(x,y,color="red")
       plt.show()

#print(fitnessValues[0])
#print(combinations[0])

#explain genetic algorithm, linking to chromosomes
#20 C 5, too large to iterate
#pick 100 at random combinations, 1011100000...0
#pick 2 and merge, probability involved

# calcuate fitness function for each set, now 101, using covariance
#can calculate correlation, standardized, more intuitive to understand
#can explain statistics, and intuition on why dividing by (sample size - 1)
# for sample size
# remove the worst performing set of 5 stocks.
#keep repeating, and system should evolve to one gene



# 1. calculate covariance matrix 20 by 20: covariance between 2 stocks:
# multiply the returns for each data point and take the average of 123 multiplications
#covariance if both stocks are the same: should be std squared = variance
#can do manually, may easier to use numpy
#x = np.array([[0, 3, 4], [1, 2, 4], [3, 4, 5]])
#print("Shape of array:\n", np.shape(x))
#print("Covariance matrix of x:\n", np.cov(x))#



#2. to find correlation divide correlation by (std of firm A * std of firm B)
#
#
#

#PRESENTATION
#explain n v np problem. 50 C 10 is too big
#explain genetic algorithm
#explain problem i want to solve
#explain covariance and correlation (of returns of stocks)
#show covariance then correlation matrix: diagonal are 1s
#
#
#


