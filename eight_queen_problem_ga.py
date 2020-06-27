import math
import random
import sys

class Chromosome:
    def __init__(self, maxLength):
        self.mMaxLength = maxLength
        self.mFitness = 0.0
        self.mSelected = False
        self.mSelectionProbability = 0.0
        self.mConflicts = 0

        self.mData = [0] * maxLength
        for i in range(self.mMaxLength):
            self.mData[i] = i
        return

    def calculate_conflicts(self):
        x = 0
        y = 0
        tempx = 0
        tempy = 0
        board = []
        conflicts = 0
        dx = [-1, 1, -1, 1]
        dy = [-1, 1, 1, -1]
        done = False

        for i in range(self.mMaxLength):
            board.append([""] * self.mMaxLength)
            board[i][self.mData[i]] = "Q"


        # Walk through each of the Queens and compute the number of conflicts.
        for i in range(self.mMaxLength):
            x = i
            y = self.mData[i]

            # Check diagonals.
            for j in range(4):
                tempx = x
                tempy = y
                done = False
                while not done:
                    tempx += dx[j]
                    tempy += dy[j]
                    if (tempx < 0 or tempx >= self.mMaxLength) or (tempy < 0 or tempy >= self.mMaxLength):
                        done = True
                    else:
                        if board[tempx][tempy] == "Q":
                            conflicts += 1

        self.mConflicts = conflicts
        return

    def get_conflicts(self):
        return self.mConflicts

    def set_selection_probability(self, probability):
        self.mSelectionProbability = probability
        return

    def get_selection_probability(self):
        return self.mSelectionProbability

    def set_selected(self, isSelected):
        self.mSelected = isSelected
        return

    def get_selected(self):
        return self.mSelected

    def set_fitness(self, score):
        self.mFitness = score
        return

    def get_fitness(self):
        return self.mFitness

    def set_data(self, index, value):
        self.mData[index] = value
        return

    def get_data(self, index):
        return self.mData[index]


class NQueenGA:
    def __init__(self, startSize, maxEpochs, matingProb, mutationRate, minSelect, maxSelect, generation, minShuffles,
                 maxShuffles, pbcMax, maxLength):
        self.mStartSize = startSize
        self.mEpochs = maxEpochs
        self.mMatingProbability = matingProb
        self.mMutationRate = mutationRate
        self.mMinSelect = minSelect
        self.mMaxSelect = maxSelect
        self.mOffspringPerGeneration = generation
        self.mMinimumShuffles = minShuffles
        self.mMaximumShuffles = maxShuffles
        self.mPBCMax = pbcMax
        self.mMaxLength = maxLength

        self.epoch = 0
        self.childCount = 0
        self.nextMutation = 0  # For scheduling mutations.
        self.mutations = 0
        self.population = []
        return

    def get_exclusive_random_integer(self, high, numberA):
        done = False
        numberB = 0

        while not done:
            numberB = random.randrange(0, high)
            if numberB != numberA:
                done = True

        return numberB

    def get_exclusive_random_integer_by_array(self, low, high, arrayA):
        done = False
        getRand = 0

        if high != low:
            while not done:
                done = True
                getRand = random.randrange(low, high)
                for i in range(len(arrayA)):
                    if getRand == arrayA[i]:
                        done = False
        else:
            getRand = high

        return getRand

    def math_round(self, inValue):
        outValue = 0
        if math.modf(inValue)[0] >= 0.5:
            outValue = math.ceil(inValue)
        else:
            outValue = math.floor(inValue)
        return outValue

    def get_maximum(self):
        # Returns an array index.
        popSize = 0;
        thisChromo = Chromosome(self.mMaxLength)
        thatChromo = Chromosome(self.mMaxLength)
        maximum = 0
        foundNewMaximum = False
        done = False

        while not done:
            foundNewMaximum = False
            popSize = len(self.population)
            for i in range(popSize):
                if i != maximum:
                    thisChromo = self.population[i]
                    thatChromo = self.population[maximum]
                    # The maximum has to be in relation to the Target.
                    if math.fabs(thisChromo.get_conflicts() > thatChromo.get_conflicts()):
                        maximum = i
                        foundNewMaximum = True

            if foundNewMaximum == False:
                done = True

        return maximum

    def get_minimum(self):
        # Returns an array index.
        popSize = 0;
        thisChromo = Chromosome(self.mMaxLength)
        thatChromo = Chromosome(self.mMaxLength)
        minimum = 0
        foundNewMinimum = False
        done = False

        while not done:
            foundNewMinimum = False
            popSize = len(self.population)
            for i in range(popSize):
                if i != minimum:
                    thisChromo = self.population[i]
                    thatChromo = self.population[minimum]
                    # The minimum has to be in relation to the Target.
                    if math.fabs(thisChromo.get_conflicts() < thatChromo.get_conflicts()):
                        minimum = i
                        foundNewMinimum = True

            if foundNewMinimum == False:
                done = True

        return minimum

    def exchange_mutation(self, index, exchanges):
        i = 0
        tempData = 0
        thisChromo = Chromosome(self.mMaxLength)
        gene1 = 0
        gene2 = 0
        done = False

        thisChromo = self.population[index]

        while not done:
            gene1 = random.randrange(0, self.mMaxLength)
            gene2 = self.get_exclusive_random_integer(self.mMaxLength, gene1)

            # Exchange the chosen genes.
            tempData = thisChromo.get_data(gene1)
            thisChromo.set_data(gene1, thisChromo.get_data(gene2))
            thisChromo.set_data(gene2, tempData)

            if i == exchanges:
                done = True

            i += 1

        self.mutations += 1
        return

    def initialize_chromosomes(self):
        for i in range(self.mStartSize):
            newChromo = Chromosome(self.mMaxLength)
            self.population.append(newChromo)
            chromoIndex = len(self.population) - 1

            # Randomly choose the number of shuffles to perform.
            shuffles = random.randrange(self.mMinimumShuffles, self.mMaximumShuffles)

            self.exchange_mutation(chromoIndex, shuffles)

            newChromo = self.population[chromoIndex]
            newChromo.calculate_conflicts()

        return

    def get_fitness(self):
        # Lowest errors = 100%, Highest errors = 0%
        popSize = len(self.population)
        thisChromo = Chromosome(self.mMaxLength)
        bestScore = 0
        worstScore = 0

        # The worst score would be the one with the highest energy, best would be lowest.
        thisChromo = self.population[self.get_maximum()]
        worstScore = thisChromo.get_conflicts()

        # Convert to a weighted percentage.
        thisChromo = self.population[self.get_minimum()]
        bestScore = worstScore - thisChromo.get_conflicts()

        for i in range(popSize):
            thisChromo = self.population[i]
            thisChromo.set_fitness((worstScore - thisChromo.get_conflicts()) * 100.0 / bestScore)

        return

    def roulette_selection(self):
        j = 0
        popSize = 0
        genTotal = 0.0
        selTotal = 0.0
        rouletteSpin = 0.0
        thisChromo = Chromosome(self.mMaxLength)
        thatChromo = Chromosome(self.mMaxLength)
        done = False

        popSize = len(self.population)
        for i in range(popSize):
            thisChromo = self.population[i]
            genTotal += thisChromo.get_fitness()

        genTotal *= 0.01

        for i in range(popSize):
            thisChromo = self.population[i]
            thisChromo.set_selection_probability(thisChromo.get_fitness() / genTotal)

        for i in range(self.mOffspringPerGeneration):
            rouletteSpin = random.randrange(0, 99)
            j = 0
            selTotal = 0
            done = False
            while not done:
                thisChromo = self.population[j]
                selTotal += thisChromo.get_selection_probability()
                if selTotal >= rouletteSpin:
                    if j == 0:
                        thatChromo = self.population[j]
                    elif j >= popSize - 1:
                        thatChromo = self.population[popSize - 1]
                    else:
                        thatChromo = self.population[j - 1]

                    thatChromo.set_selected(True)
                    done = True
                else:
                    j += 1

        return

    def choose_first_parent(self):
        parent = 0
        thisChromo = Chromosome(self.mMaxLength)
        done = False

        while not done:
            # Randomly choose an eligible parent.
            parent = random.randrange(0, len(self.population) - 1)
            thisChromo = self.population[parent]
            if thisChromo.get_selected() == True:
                done = True

        return parent

    def choose_second_parent(self, parentA):
        parentB = 0
        thisChromo = Chromosome(self.mMaxLength)
        done = False

        while not done:
            # Randomly choose an eligible parent.
            parentB = random.randrange(0, len(self.population) - 1)
            if parentB != parentA:
                thisChromo = self.population[parentB]
                if thisChromo.get_selected() == True:
                    done = True

        return parentB

    def partially_mapped_crossover(self, chromA, chromB, child1, child2):
        thisChromo = Chromosome(self.mMaxLength)
        thisChromo = self.population[chromA]
        thatChromo = Chromosome(self.mMaxLength)
        thatChromo = self.population[chromB]
        newChromo1 = Chromosome(self.mMaxLength)
        newChromo1 = self.population[child1]
        newChromo2 = Chromosome(self.mMaxLength)
        newChromo2 = self.population[child2]

        crossPoint1 = random.randrange(0, self.mMaxLength)
        crossPoint2 = self.get_exclusive_random_integer(self.mMaxLength, crossPoint1)
        if crossPoint2 < crossPoint1:
            j = crossPoint1
            crossPoint1 = crossPoint2
            crossPoint2 = j

        # Copy Parent genes to offspring.
        for i in range(self.mMaxLength):
            newChromo1.set_data(i, thisChromo.get_data(i))
            newChromo2.set_data(i, thatChromo.get_data(i))

        for i in range(crossPoint1, crossPoint2 + 1):
            # // Get the two items to swap.
            item1 = thisChromo.get_data(i)
            item2 = thatChromo.get_data(i)
            pos1 = 0
            pos2 = 0

            # Get the items' positions in the offspring.
            for j in range(self.mMaxLength):
                if newChromo1.get_data(j) == item1:
                    pos1 = j
                elif newChromo1.get_data(j) == item2:
                    pos2 = j

            # Swap them.
            if item1 != item2:
                newChromo1.set_data(pos1, item2)
                newChromo1.set_data(pos2, item1)

            # Get the items'  positions in the offspring.
            for j in range(self.mMaxLength):
                if newChromo2.get_data(j) == item2:
                    pos1 = j
                elif newChromo2.get_data(j) == item1:
                    pos2 = j

            # Swap them.
            if item1 != item2:
                newChromo2.set_data(pos1, item1)
                newChromo2.set_data(pos2, item2)

        return

    def position_based_crossover(self, chromA, chromB, child1, child2):
        k = 0
        numPoints = 0
        tempArray1 = [0] * self.mMaxLength
        tempArray2 = [0] * self.mMaxLength
        matchFound = False
        thisChromo = Chromosome(self.mMaxLength)
        thisChromo = self.population[chromA]
        thatChromo = Chromosome(self.mMaxLength)
        thatChromo = self.population[chromB]
        newChromo1 = Chromosome(self.mMaxLength)
        newChromo1 = self.population[child1]
        newChromo2 = Chromosome(self.mMaxLength)
        newChromo2 = self.population[child2]

        # Choose and sort the crosspoints.
        numPoints = random.randrange(0, self.mPBCMax)
        crossPoints = [0] * numPoints
        for i in range(numPoints):
            crossPoints[i] = self.get_exclusive_random_integer_by_array(0, self.mMaxLength - 1, crossPoints)

        # Get non-chosens from parent 2
        k = 0
        for i in range(self.mMaxLength):
            matchFound = False
            for j in range(numPoints):
                if thatChromo.get_data(i) == thisChromo.get_data(crossPoints[j]):
                    matchFound = True

            if matchFound == False:
                tempArray1[k] = thatChromo.get_data(i)
                k += 1

        # Insert chosens into child 1.
        for i in range(numPoints):
            newChromo1.set_data(crossPoints[i], thisChromo.get_data(crossPoints[i]))

        # Fill in non-chosens to child 1.
        k = 0
        for i in range(self.mMaxLength):
            matchFound = False
            for j in range(numPoints):
                if i == crossPoints[j]:
                    matchFound = True

            if matchFound == False:
                newChromo1.set_data(i, tempArray1[k])
                k += 1

        # Get non-chosens from parent 1
        k = 0
        for i in range(self.mMaxLength):
            matchFound = False
            for j in range(numPoints):
                if thisChromo.get_data(i) == thatChromo.get_data(crossPoints[j]):
                    matchFound = True

            if matchFound == False:
                tempArray2[k] = thisChromo.get_data(i)
                k += 1

        # Insert chosens into child 2.
        for i in range(numPoints):
            newChromo2.set_data(crossPoints[i], thatChromo.get_data(crossPoints[i]))

        # Fill in non-chosens to child 2.
        k = 0
        for i in range(self.mMaxLength):
            matchFound = False
            for j in range(numPoints):
                if i == crossPoints[j]:
                    matchFound = True

            if matchFound == False:
                newChromo2.set_data(i, tempArray2[k])
                k += 1

        return

    def displacement_mutation(self, index):
        j = 0
        point1 = 0
        length = 0
        point2 = 0
        tempArray1 = [0] * self.mMaxLength
        tempArray2 = [0] * self.mMaxLength
        thisChromo = Chromosome(self.mMaxLength)
        thisChromo = self.population[index]

        # Randomly choose a section to be displaced.
        point1 = random.randrange(0, self.mMaxLength)
        # sys.stdout.write(str(point1))
        # sys.stdout.write(str(self.mMaxLength))

        # Generate re-insertion point.
        candidate = self.mMaxLength - (point1 + 2)
        if candidate <= 0:
            candidate = 1
        point2 = self.get_exclusive_random_integer(candidate, point1)

        j = 0
        for i in range(self.mMaxLength):  # Get non-chosen
            if i < point1 or i > point1 + length:
                tempArray1[j] = thisChromo.get_data(i)
                j += 1

        j = 0
        for i in range(point1, point1 + length + 1):  # Get chosen
            tempArray2[j] = thisChromo.get_data(i)
            j += 1

        j = 0
        for i in range(point2, point2 + length + 1):  # Place chosen
            thisChromo.set_data(i, tempArray2[j])
            j += 1

        j = 0
        for i in range(i, self.mMaxLength):  # Place non-chosen
            if i < point2 or i > point2 + length:
                thisChromo.set_data(i, tempArray1[j])
                j += 1

        self.mutations += 1
        return

    def do_mating(self):
        getRand = 0
        parentA = 0
        parentB = 0
        newChildIndex1 = 0
        newChildIndex2 = 0
        newChromo1 = Chromosome(self.mMaxLength)
        newChromo2 = Chromosome(self.mMaxLength)

        for i in range(self.mOffspringPerGeneration):
            parentA = self.choose_first_parent()
            # Test probability of mating.
            getRand = random.randrange(0, 100)

            if getRand <= self.mMatingProbability * 100:
                parentB = self.choose_second_parent(parentA)
                newChromo1 = Chromosome(self.mMaxLength)
                newChromo2 = Chromosome(self.mMaxLength)
                self.population.append(newChromo1)
                newIndex1 = len(self.population) - 1
                self.population.append(newChromo2)
                newIndex2 = len(self.population) - 1

                self.partially_mapped_crossover(parentA, parentB, newIndex1, newIndex2)
                # self.position_based_crossover(parentA, parentB, newIndex1, newIndex2)

                if self.childCount - 1 == self.nextMutation:
                    self.exchange_mutation(newIndex1, 1)
                    # self.displacement_mutation(newIndex1)
                elif self.childCount == self.nextMutation:
                    self.exchange_mutation(newIndex2, 1)
                    # self.displacement_mutation(newIndex2)

                newChromo1 = self.population[newIndex1]
                newChromo1.calculate_conflicts()
                newChromo2 = self.population[newIndex2]
                newChromo2.calculate_conflicts()

                self.childCount += 2

                # Schedule next mutation.
                if math.fmod(self.childCount, self.math_round(1.0 / self.mMutationRate)) == 0:
                    self.nextMutation = self.childCount + random.randrange(0, self.math_round(1.0 / self.mMutationRate))

        return

    def prep_next_epoch(self):
        popSize = 0;
        thisChromo = Chromosome(self.mMaxLength)

        # Reset flags for selected individuals.
        popSize = len(self.population)
        for i in range(popSize):
            thisChromo = self.population[i]
            thisChromo.set_selected(False)

        return

    def print_best_solution(self, bestSolution=Chromosome(8)):
        board = []
        for i in range(self.mMaxLength):
            board.append([""] * self.mMaxLength)
            board[i][bestSolution.get_data(i)] = "Q"
        rows_results = []
        # Display the sequence.

        for j in range(self.mMaxLength):
            for i in range(self.mMaxLength):
                if board[i][j] == "Q":
                    val_res = (j,i)
                    rows_results.append(val_res)
        sequence_list = list(list((zip(*(sorted(rows_results,key = lambda x : x[1] )))))[0])

        sequence_list_str = ' '.join([str(sequence_val) for sequence_val in sequence_list])
        print(sequence_list_str)
        return

    def n_queen_sequence_generator(self):
        popSize = 0
        thisChromo = Chromosome(self.mMaxLength)
        done = False

        self.mutations = 0
        self.nextMutation = random.randrange(0, self.math_round(1.0 / self.mMutationRate))

        while not done:
            popSize = len(self.population)
            for i in range(popSize):
                thisChromo = self.population[i]
                if thisChromo.get_conflicts() == 0 or self.epoch == self.mEpochs:
                    done = True

            self.get_fitness()

            self.roulette_selection()

            self.do_mating()

            self.prep_next_epoch()

            self.epoch += 1
            # runtime epoch status.
            #print("Epoch: " + str(self.epoch) + "\n")

        if self.epoch != self.mEpochs:
            popSize = len(self.population)
            for i in range(popSize):
                thisChromo = self.population[i]
                if thisChromo.get_conflicts() == 0:
                    self.print_best_solution(thisChromo)
        return


if __name__ == '__main__':

    nqueen = NQueenGA(75, 1000, 0.7, 0.001, 10, 50,20, 8, 20, 4, 8)
    nqueen.initialize_chromosomes()
    nqueen.n_queen_sequence_generator()