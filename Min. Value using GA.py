# this Code Made by Putri Apriyanti Windya (1301174169) IF-41-12 Informatics Engineering, Telkom University

import numpy as np
import random


# generate the possible population in this case i use Binary genotype
def populationGenerator(nKrom,nIndv):
	
	genotypeL = []

	for i in range(nIndv):
		gen = []
		for j in range(nKrom):
			rand_ = random.randint(0,1)
			gen.append(rand_)
		genotypeL.append(gen)

	return genotypeL


# decode the chromosome that has been generated
def decodeFunc(minx1, maxx1, minx2, maxx2,individu):
	x1 = 	minx1 + (maxx1 - minx1)/(2**-1 + 2**-2 + 2**-3 + 2**-4) * (individu[0]*2**-1 + individu[1]*2**-2 + individu[2]*2**-3 + individu[3]*2**-4)
	x2 = 	minx2 + (maxx2 - minx2)/(2**-1 + 2**-2 + 2**-3 + 2**-4) * (individu[4]*2**-1 + individu[5]*2**-2 + individu[6]*2**-3 + individu[7]*2**-4)
	return x1,x2

# generate kromosomX1 and kromosomX2 list 
def decodeFuncResult(population,minx1, maxx1, minx2, maxx2):
	# initiate list of chromosome x1 dan chromosome x2
	kromosomX1 = []
	kromosomX2 = []

	# decode the chromosome using decodeFunc
	for individu in population:
		x1,x2 = decodeFunc(minx1,maxx1,minx2,maxx2,individu)
		# print(x1," ",x2)
		kromosomX1.append(x1)
		kromosomX2.append(x2)

	return kromosomX1, kromosomX2

# fitness value of a chromosome
def fitnessFunc(x1,x2):
	nh = (4-2*(x1**2) + (x1**4/3))*x1**2 + x1*x2 + (-4 + 4*x2**2)*x2**2
	a  = 0.001
	return 1 / (nh + a)

# compute all fitness of each chromosome
def computeFitness(nIdividu, kromosomX1, kromosomX2):
	# initiate list of Fitness
	fitnessL = []

	# compute fitness of each chromosome using fitnessFunc
	for i in range(nIndividu):
		fitn= fitnessFunc(kromosomX1[i],kromosomX2[i])
		fitnessL.append(fitn)

	return fitnessL

# compute total fitness
def totalFitness(fitness):
	return np.sum(fitness)


# compute each fitness probability
def fitProbability(fitness):
	total = np.sum(fitness)
	fitP = []
	for i in range(len(fitness)):
			fit = fitness[i]/total
			fitP.append(fit)
			# print(i,"XZ",fit)
	return fitP

# compute all of fitness probability
def fitProbCumulative(fitness):
	fitProb = fitProbability(fitness)
	cumP = []
	cum = 0
	for i in range(len(fitness)):
		cum += fitProb[i]
		cumP.append(cum) 
		# print("YZ",cum)
	return cumP

# random
def rand_(fitness):
	rnd = []
	for i in range(len(fitness)):
		ran_ = random.random()
		rnd.append(ran_)
	return rnd

# select the parent that will be cross using roulette wheel selection
def parentSelection(fitness, population, randm):
	probFit = fitProbability(fitness)
	probKum = fitProbCumulative(fitness)

	random.seed(random.randint(0,1000))
	# rand_ = random.uniform(0,1)
	for i in range(len(population)):
		# rand_ = random.uniform(0,1)
		# print("renn",randm)
		# if rand_ > probKum[i-1] and rand_ < probKum[i]:
		# 	parent = population[i]
		if i > 0 and randm > probKum[i-1] and randm <probKum[i]:
			parents = population[i]
			# print(i+1,parents)
			break
		elif randm < probKum[0]:
			parents = population[i]
			# print(i+1,parents)
			break

	return parents

# create the child using crossover
def crossOverC(parentL):
  childL = []
  j = 1
  i = 0
 
  while i < len(parentL):
    # pointC = 2
    pointC = random.randint(1,nKromosom-1)
    # print("X", pointC)
    c1 = parentL[i]
    c2 = parentL[j]

    childL.append(c1[:pointC] + c2[pointC:])
    childL.append(c2[:pointC] + c1[pointC:])
    i += 2
    j += 2

  return childL

# Mutate the child from child list
def mutationChild(childL,pMutation):
  mutL = childL
  for i in range(len(mutL)):
    rand_ = random.random()
    if rand_ < pMutation:
      randP = random.randint(1,nKromosom-1)
      # print(i," " ,randP)
      if mutL[i][randP] == 1:
              mutL[i][randP] = 0
      else:
              mutL[i][randP] = 1

  return mutL 

# changes Generation
def generationRep(old,new, fitness, fitnessN):
	minimO = np.min(fitness)
	minimN = np.min(fitnessN)
	newGen = []
	# maximum = fitness[0]
	for i in range(len(old)):
		if fitness[i] < fitnessN[i]:
			newGen.append(fitnessN[i])
		else:
			newGen.append(fitness[i])

	return new


# initiation step
nKromosom = 8
nIndividu = 10
numOfGeneration = 100
mProbability = 0.10
cProbability = 0.80
minx1 = -3 
maxx1 = 3
minx2 = -2
maxx2 = 2

# call function populationGenerator to generate possible population 
populationA = populationGenerator(nKromosom, nIndividu)
# print(population)
	


x1, x2 = decodeFuncResult(populationA,minx1, maxx1, minx2, maxx2);

# # compute fitness of each chromosome using computeFitness
# fitnessL = computeFitness(nIndividu, x1, x2)
# totalFitness = 0
# totalFitness = np.sum(fitnessL)

fitnessL = []

# looping for all generation in GA
for i in range(numOfGeneration):
	# make list for next gen
	nextGenL = []
	selParentL = []
	fitnessNextL = []

	population = populationA
	# compute fitness of each chromosome using computeFitness
	fitnessL = computeFitness(nIndividu, x1, x2)
	totalFitness = 0
	totalFitness = np.sum(fitnessL)
	# print(fitnessL)

	rand = rand_(fitnessL)
	for j in range(nIndividu):
		selParentL.append(parentSelection(fitnessL, population,rand[j]))
	# print(selParentL)
	childL = crossOverC(selParentL)
	# print(childL)
	mutationL = mutationChild(childL, mProbability)
	# print(mutationL)
	x1, x2 = decodeFuncResult(population,minx1, maxx1, minx2, maxx2);
	fitnessNextL = computeFitness(len(mutationL),x1,x2)
	nextGenL = generationRep(population,mutationL, fitnessL, fitnessNextL)
	population = nextGenL
	x1, x2 = decodeFuncResult(population,minx1, maxx1, minx2, maxx2);
	fitnessL = computeFitness(len(population),x1,x2)

idx = 0
maximum= fitnessL[0]
# print(len(fitnessL))
# print(len(population))
for k in range(len(fitnessL)):
	if (fitnessL[k] > maximum):
		maximum = fitnessL[k]
		# print(idx)
		idx = k
# Generation end

# find the minimum value of last generation 
# minFitness = np.min(fitnessL)
print("====== Result =====")
print("Kromosom = ",population[idx])
print("Fitness = ",fitnessL[idx])
print("X1 = ",x1[idx])
print("X2 = ", x2[idx])
	
