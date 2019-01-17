from deap import creator, tools, algorithms, gp, base
from deap.gp import *
import operator
from scipy.io import arff
from io import StringIO
from scoop import futures
import math
import os
import numpy
import re
import warnings
warnings.filterwarnings("ignore")

# open the arff file
directory = os.getcwd().split('src')[0]
file_name = "\\desharnais.arff"
raw_data = open(directory + file_name).read()

# read arff
f = StringIO(raw_data)
data, meta = arff.loadarff(f)

# Get attribute names and indices of values
attributes = {}
for i in range(len(meta.names())):
    attributes[meta.names()[i]] = i
attributes.pop("Project")
attributes.pop("YearEnd")
attributes.pop("PointsNonAdjust")
print(attributes)
# convert data to a dictionary
final_data = []

for i in range(len(data)):
    current = data[i]
    test_dict = {}
    for key in attributes.keys():
        value_expression = current[attributes[key]]
        if key == 'Language':
            value_expression = str(value_expression)
            value_expression = int(re.sub("\D", "", value_expression))

            test_dict[key] = value_expression
        else:
            test_dict[key] = int(value_expression)

    final_data.append(test_dict)

# split data into training and testData
trainData = final_data[:int(len(data) * 0.8)]
testData = final_data[int(len(data) * 0.8):]

# recommended in deap docs
def protectedDiv(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return 1

def protectedSqrt(x):
    if(x >= 0):
        return x**(0.5)
    else:
        # how else to handle?
        return  abs(x)**(0.5)

def protectedLog10(x):
    if(x <= 0.0):
        return 0
    else:
       return math.log10(x)

def protectedLog2(x):
    if(x <= 0.0):
        return 0
    else:
       return math.log2(x)

def distance(x, y):
    if x >= y:
        result = x - y
    else:
        result = y - x
    return result

# create primitve set and add operators
primitive_set = PrimitiveSet("main", 8)
primitive_set.addPrimitive(operator.add, 2)
primitive_set.addPrimitive(operator.mul, 2)
primitive_set.addPrimitive(protectedSqrt, 1)
primitive_set.addPrimitive(protectedLog2, 1)
primitive_set.addPrimitive(protectedLog10, 1)
primitive_set.addEphemeralConstant("ran%d" % random.randrange(10,1000), lambda: random.randrange(-1, 1))
primitive_set.addPrimitive(protectedDiv, 2)
primitive_set.addPrimitive(operator.sub, 2)
primitive_set.addPrimitive(math.sin, 1)
primitive_set.addPrimitive(math.cos, 1)

# rename the arguments
count = 0
for key in attributes.keys():
    argName = 'ARG%d' % count
    primitive_set.renameArguments(**{argName : key})
    count += 1

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

def main(popSize, mutation, cx, nGens, tournSize):
    minTreeSize, maxTreeSize = 4, 14
    gp_toolbox = base.Toolbox()
    gp_toolbox.register("map", futures.map)
    gp_toolbox.register("expr", gp.genHalfAndHalf, pset=primitive_set, min_= minTreeSize, max_= maxTreeSize)
    gp_toolbox.register("individual", tools.initIterate, creator.Individual, gp_toolbox.expr)
    gp_toolbox.register("population", tools.initRepeat, list, gp_toolbox.individual)
    gp_toolbox.register("compile", gp.compile, pset=primitive_set)

    # https://deap.readthedocs.io/en/master/examples/gp_symbreg.html
    def evaluate(individual, trainData):
        func = gp_toolbox.compile(individual)
        difference = 0
        for i in range(len(trainData)):
            try:
                currentValue = func(trainData[i]['TeamExp'],
                                trainData[i]['ManagerExp'],
                                trainData[i]['Length'],
                                trainData[i]['Transactions'],
                                trainData[i]['Entities'],
                                trainData[i]['Adjustment'],
                                trainData[i]['PointsAjust'],
                                trainData[i]['Language'])
            except:
                print("integer too large!")
                currentValue = 2,147,483,647

            difference += distance(trainData[i]['Effort'], currentValue)

        mae = difference / len(trainData)

        return mae,

    gp_toolbox.register("evaluate", evaluate, trainData=trainData)
    gp_toolbox.register("mate", gp.cxOnePoint)
    gp_toolbox.register("select", tools.selTournament, tournsize=tournSize)
    gp_toolbox.register("expr_mut", gp.genHalfAndHalf, min_=minTreeSize, max_=maxTreeSize)
    gp_toolbox.register("mutate", gp.mutUniform, expr=gp_toolbox.expr_mut, pset=primitive_set)
    # https://deap.readthedocs.io/en/master/examples/gp_symbreg.html
    # limit overall tree height
    gp_toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)) # static limit of 17 recomended in deap docs
    gp_toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    # register stats for fitness and size of each individual (tree)
    mstats = tools.Statistics(lambda individual: individual.fitness.values)

    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    log = tools.Logbook()

    pop = gp_toolbox.population(n=popSize)
    hof = tools.HallOfFame(popSize)
    print("Starting GA")
    pop, log = algorithms.eaSimple(pop, gp_toolbox, cx, mutation, nGens, mstats, hof, True)

    print("GA Complete after %d gens, tournament selection between %d, mutation rate of %f, crossover rate of %f" % (
    nGens, tournSize, mutation, cx))

    # Results Time
    # Calculate Correlation Coefficient
    coefficients = []
    maes = []
    guesses = []
    answers = []
    final_function = None

    final_train_mae = 0
    final_train_rmse = 0

    for i in range(1):
        hof_func = gp_toolbox.compile(hof[i])
        current_cc = 0
        current_mae = 0
        guesses.clear()
        answers.clear()

        for j in range(len(trainData)):
            guess = hof_func(trainData[j]['TeamExp'],
                             trainData[j]['ManagerExp'],
                             trainData[j]['Length'],
                             trainData[j]['Transactions'],
                             trainData[j]['Entities'],
                             trainData[j]['Adjustment'],
                             trainData[j]['PointsAjust'],
                             trainData[j]['Language'])
            guesses.append(guess)
            answers.append(trainData[j]['Effort'])

        diff = 0
        diffSquared = 0
        for j in range(len(guesses)):
            difference = distance(guesses[j], answers[j])
            diff += difference
            diffSquared += pow(difference, 2)

        MAE = diff / len(trainData)
        RMSE = protectedSqrt(diffSquared / len(trainData))

        final_train_mae = MAE
        final_train_rmse = RMSE

        current_cc = numpy.corrcoef(guesses, answers)[0, 1]
        coefficients.append(current_cc)
        print("\nCoefficient for Best Individual on training set = %f" % (current_cc))
        print("MAE for Best Individual on training set = %f\n" % (MAE))
        print("RMSE for Best Individual on training set = %f\n" % (RMSE))
        final_function = hof_func

    final_answers = []
    final_guesses = []

    for i in range(len(testData)):
        currentDataPoint = testData[i]
        answer = currentDataPoint['Effort']
        guess = final_function(currentDataPoint['TeamExp'],
                               currentDataPoint['ManagerExp'],
                               currentDataPoint['Length'],
                               currentDataPoint['Transactions'],
                               currentDataPoint['Entities'],
                               currentDataPoint['Adjustment'],
                               currentDataPoint['PointsAjust'],
                               currentDataPoint['Language'])
        final_answers.append(answer)
        final_guesses.append(guess)

    diff = 0
    diffSquared = 0

    for i in range(len(final_guesses)):
        absoluteError = distance(final_guesses[i], final_answers[i])
        diff += absoluteError
        diffSquared += pow(absoluteError, 2)

    final_mae = diff / len(final_guesses)
    final_mae_diff = distance(final_mae, final_train_mae)
    final_rmse = protectedSqrt(diffSquared / len(testData))
    final_rmse_diff = distance(final_rmse, final_train_rmse)
    final_cc = numpy.corrcoef(final_guesses, final_answers)[0, 1]

    print("\nCoefficient for Best Individual on test set = %f" % (final_cc))
    print("MAE for Best Individual on test set = %f\n" % (final_mae))
    print("RMSE for Best Individual on test set = %f\n" % (final_rmse))

    return final_mae, final_mae_diff, final_rmse, final_rmse_diff, final_cc, hof[0]
if __name__ == "__main__":
    main()
