
from cProfile import label

from collections import UserList
from dbm import ndbm
from gc import callbacks
from turtle import pos, width
# from syslog import LOG_DEBUG
from unicodedata import name
from urllib.parse import uses_params

import dearpygui.dearpygui as dpg

import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians

from platformdirs import user_data_dir

# tsp functions made in tsp_code


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        R = 6373.0

        lat1 = radians(self.x)
        lon1 = radians(self.y)
        lat2 = radians(city.x)
        lon2 = radians(city.y)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


# In[5]:


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child



def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children


def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual



def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


# In[15]:


def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


# In[16]:


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    # dpg.add_text("Initial Distance :")
    a = str(1 / rankRoutes(pop)[0][1])
    # init_dist = a
    # dpg.add_text(a)

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    # dpg.add_text("Final Distance ")
    val = str(1 / rankRoutes(pop)[0][1])
    # final_dist = val
    # dpg.add_text(val)
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute, a, val


def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


# selected_cities = []


# tsp_code ends

lucknow_areas = {'Alambag': [26.82512, 80.9138], 'Charbag': [26.82626, 80.91482], 'Aliganj': [26.896099, 80.95153], 'Jankipuram': [26.923059, 80.940399], 'Aminabad': [26.8428, 80.928902], 'Gomti nagar': [26.862337, 81.019958], 'Indra nagar': [26.84626, 80.949], 'Bakshi ka talab': [26.98654, 80.921051], 'Husain ganj': [26.86777, 80.94816], 'Hzartganj': [26.68188, 80.98273], 'Badshah nagar': [26.87003, 80.96029], 'Krishna nagar': [26.79685, 80.88471], 'Transport nagar': [26.77992, 80.88399],  'Pandeyganj': [26.85068, 80.9168], 'Shanti nagar': [26.87055, 80.97177], 'Alamnagar': [26.84812, 80.871918], 'Kalyanpur': [26.94735, 80.989929]}


cities = {'Delhi': [28.66, 77.23], 'Mumbai': [18.9667, 72.8333], 'Kolkāta': [22.5411, 88.3378], 'Bangalore': [12.9699, 77.598], 'Chennai': [13.0825, 80.275], 'Hyderābād': [17.3667, 78.4667], 'Pune': [18.5196, 73.8553], 'Ahmedabad': [23.03, 72.58], 'Sūrat': [21.17, 72.83], 'Lucknow': [26.847, 80.947], 'Jaipur': [26.9167, 75.8667], 'Cawnpore': [26.4725, 80.3311], 'Mirzāpur': [25.15, 82.58], 'Nāgpur': [21.1539, 79.0831], 'Ghāziābād': [28.6667, 77.4167], 'Indore': [22.7206, 75.8472], 'Vadodara': [22.3, 73.2], 'Vishākhapatnam': [17.7333, 83.3167], 'Bhopāl': [23.25, 77.4167], 'Chinchvad': [18.6278, 73.8131], 'Patna': [25.61, 85.1414], 'Ludhiāna': [30.9083, 75.8486], 'Āgra': [27.18, 78.02], 'Kalyān': [19.2502, 73.1602], 'Madurai': [9.9197, 78.1194], 'Jamshedpur': [22.8, 86.1833], 'Nāsik': [20.0, 73.7833], 'Farīdābād': [28.4333, 77.3167], 'Aurangābād': [24.7704, 84.38], 'Rājkot': [22.2969, 70.7984], 'Meerut': [28.99, 77.7], 'Jabalpur': [23.1667, 79.9333], 'Thāne': [19.18, 72.9633], 'Dhanbād': [23.7928, 86.435], 'Allahābād': [25.455, 81.84], 'Vārānasi': [25.3189, 83.0128], 'Srīnagar': [34.0911, 74.8061], 'Amritsar': [31.6167, 74.85], 'Alīgarh': [27.88, 78.08], 'Bhiwandi': [19.3, 73.0667], 'Gwalior': [26.215, 78.1931], 'Bhilai': [21.2167, 81.4333], 'Hāora': [22.59, 88.31], 'Rānchi': [23.3556, 85.3347], 'Bezwāda': [16.5167, 80.6167], 'Chandīgarh': [30.7353, 76.7911], 'Mysore': [12.3086, 76.6531], 'Raipur': [21.2379, 81.6337], 'Kota': [25.18, 75.83], 'Bareilly': [28.364, 79.415], 'Jodhpur': [26.2918, 73.0168], 'Coimbatore': [11.0, 76.9667], 'Dispur': [26.15, 91.77], 'Guwāhāti': [26.1667, 91.7667], 'Solāpur': [17.6833, 75.9167], 'Trichinopoly': [10.8269, 78.6928], 'Hubli': [15.36, 75.125], 'Jalandhar': [31.3256, 75.5792], 'Bhubaneshwar': [20.2644, 85.8281], 'Bhayandar': [19.3, 72.85], 'Morādābād': [28.8418, 78.7568], 'Kolhāpur': [16.7, 74.2333], 'Thiruvananthapuram': [8.5, 76.8997], 'Sahāranpur': [29.964, 77.546], 'Warangal': [17.9756, 79.6011], 'Salem': [11.65, 78.1667], 'Mālegaon': [20.55, 74.55], 'Kochi': [9.9667, 76.2833], 'Gorakhpur': [26.7611, 83.3667], 'Shimoga': [13.9304, 75.56], 'Tiruppūr': [11.1075, 77.3398], 'Guntūr': [16.3, 80.45], 'Raurkela': [22.2492, 84.8828], 'Mangalore': [12.8703, 74.8806], 'Nānded': [19.15, 77.3333], 'Cuttack': [20.45, 85.8667], 'Chānda': [19.95, 79.3], 'Dehra Dūn': [30.318, 78.029], 'Durgāpur': [23.55, 87.32], 'Āsansol': [23.6833, 86.9667], 'Bhāvnagar': [21.765, 72.1369], 'Amrāvati': [20.9333, 77.75], 'Nellore': [14.4333, 79.9667], 'Ajmer': [26.468, 74.639], 'Tinnevelly': [8.7289, 77.7081], 'Bīkaner': [28.0181, 73.3169], 'Agartala': [23.8333, 91.2667], 'Ujjain': [23.1828, 75.7772], 'Jhānsi': [25.4486, 78.5696], 'Ulhāsnagar': [19.2167, 73.15], 'Davangere': [14.4667, 75.9167], 'Jammu': [32.7333, 74.85], 'Belgaum': [15.8667, 74.5], 'Gulbarga': [17.3333, 76.8333], 'Jāmnagar': [22.47, 70.07], 'Dhūlia': [20.9, 74.7833], 'Gaya': [24.75, 85.0167], 'Jalgaon': [21.0167, 75.5667], 'Kurnool': [15.8222, 78.035], 'Udaipur': [24.5833, 73.6833], 'Bellary': [15.15, 76.915], 'Sāngli': [16.8667, 74.5667], 'Tuticorin': [8.7833, 78.1333], 'Calicut': [11.25, 75.7667], 'Akola': [20.7333, 77.0], 'Bhāgalpur': [25.25, 87.0167], 'Sīkar': [27.6119, 75.1397], 'Tumkūr': [13.33, 77.1], 'Quilon': [8.8853, 76.5864], 'Muzaffarnagar': [29.4708, 77.7033], 'Bhīlwāra': [25.35, 74.6333], 'Nizāmābād': [18.6704, 78.1], 'Bhātpāra': [22.8667, 88.4167], 'Kākināda': [16.9333, 82.2167], 'Parbhani': [19.2704, 76.76], 'Pānihāti': [22.69, 88.37], 'Lātūr': [18.4004, 76.57], 'Rohtak': [28.9, 76.5667], 'Rājapālaiyam': [9.4204, 77.58], 'Ahmadnagar': [19.0833, 74.7333], 'Cuddapah': [14.4667, 78.8167], 'Rājahmundry': [16.9833, 81.7833], 'Alwar': [27.5667, 76.6167], 'Muzaffarpur': [26.12, 85.3833], 'Bilāspur': [22.15, 82.0167], 'Mathura': [27.4833, 77.6833], 'Kāmārhāti': [22.67, 88.37], 'Patiāla': [30.3204, 76.385], 'Saugor': [23.8504, 78.75], 'Bijāpur': [16.8244, 75.7154], 'Brahmapur': [19.32, 84.8], 'Shāhjānpur': [27.8804, 79.905], 'Trichūr': [10.52, 76.21], 'Barddhamān': [23.25, 87.85], 'Kulti': [23.73, 86.85], 'Sambalpur': [21.4704, 83.9701], 'Purnea': [25.78, 87.47], 'Hisar': [29.1489, 75.7367], 'Fīrozābād': [27.15, 78.3949], 'Bīdar': [17.9229, 77.5175], 'Rāmpur': [28.8154, 79.025], 'Shiliguri': [26.72, 88.42], 'Bāli': [22.65, 88.34], 'Pānīpat': [29.4004, 76.97], 'Karīmnagar': [18.4333, 79.15], 'Bhuj': [23.2504, 69.81], 'Ichalkaranji': [16.7, 74.47], 'Tirupati': [13.65, 79.42], 'Hospet': [15.2667, 76.4], 'Āīzawl': [23.7104, 92.72], 'Sannai': [24.16, 80.83], 'Bārāsat': [22.2333, 88.45], 'Ratlām': [23.3167, 75.0667], 'Handwāra': [34.4, 74.28], 'Drug': [21.19, 81.28], 'Imphāl': [24.82, 93.95], 'Anantapur': [14.6833, 77.6], 'Etāwah': [26.7855, 79.015], 'Rāichūr': [16.2104, 77.355], 'Ongole': [15.5, 80.05], 'Bharatpur': [27.2172, 77.49], 'Begusarai': [25.42, 86.13], 'Sonīpat': [28.9958, 77.0114], 'Rāmgundam': [18.8, 79.45], 'Hāpur': [28.7437, 77.7628], 'Uluberiya': [22.47, 88.11], 'Porbandar': [21.6425, 69.6047], 'Pāli': [25.7725, 73.3233], 'Vizianagaram': [18.1167, 83.4167], 'Puducherry': [11.93, 79.83], 'Karnāl': [29.6804, 76.97], 'Nāgercoil': [8.17, 77.43], 'Tanjore': [10.8, 79.15], 'Sambhal': [28.58, 78.55], 'Naihāti': [22.9, 88.42], 'Secunderābād': [17.45, 78.5], 'Kharagpur': [22.3302, 87.3237], 'Dindigul': [10.35, 77.95], 'Shimla': [31.1033, 77.1722], 'Ingrāj Bāzār': [25.0, 88.15], 'Ellore': [16.7, 81.1], 'Puri': [19.8, 85.8167], 'Haldia': [22.0257, 88.0583], 'Nandyāl': [15.48, 78.48], 'Bulandshahr': [28.4104, 77.8484], 'Chakradharpur': [22.7, 85.63], 'Bhiwāni': [28.7833, 76.1333], 'Gurgaon': [28.45, 77.02], 'Burhānpur': [21.3004, 76.13], 'Khammam': [17.25, 80.15], 'Madhyamgram': [22.7, 88.45], 'Ghāndīnagar': [23.22, 72.68], 'Baharampur': [24.1, 88.25], 'Mahbūbnagar': [16.7333, 77.9833], 'Mahesāna': [23.6, 72.4], 'Ādoni': [15.63, 77.28], 'Rāiganj': [25.6167, 88.1167], 'Bhusāval': [21.02, 75.83], 'Bahraigh': [27.6204, 81.6699], 'Shrīrāmpur': [22.75, 88.34], 'Tonk': [26.1505, 75.79], 'Sirsa': [29.4904, 75.03], 'Jaunpur': [25.7333, 82.6833], 'Madanapalle': [
    13.55, 78.5], 'Hugli': [22.9, 88.39], 'Vellore': [12.9204, 79.15], 'Alleppey': [9.5004, 76.37], 'Cuddalore': [11.75, 79.75], 'Deo': [24.6561, 84.4356], 'Chīrāla': [15.82, 80.35], 'Machilīpatnam': [16.1667, 81.1333], 'Medinīpur': [22.4333, 87.3333], 'Bāramūla': [34.2, 74.34], 'Chandannagar': [22.8667, 88.3833], 'Fatehpur': [25.8804, 80.8], 'Udipi': [13.3322, 74.7461], 'Tenāli': [16.243, 80.64], 'Sitalpur': [27.63, 80.75], 'Conjeeveram': [12.8308, 79.7078], 'Proddatūr': [14.73, 78.55], 'Navsāri': [20.8504, 72.92], 'Godhra': [22.7755, 73.6149], 'Budaun': [28.03, 79.09], 'Chittoor': [13.2, 79.1167], 'Harīpur': [31.52, 75.98], 'Saharsa': [25.88, 86.6], 'Vidisha': [23.5239, 77.8061], 'Pathānkot': [32.2689, 75.6497], 'Nalgonda': [17.05, 79.27], 'Dibrugarh': [27.4833, 95.0], 'Bālurghāt': [25.2167, 88.7667], 'Krishnanagar': [23.4, 88.5], 'Fyzābād': [26.7504, 82.17], 'Silchar': [24.7904, 92.79], 'Shāntipur': [23.25, 88.43], 'Hindupur': [13.83, 77.49], 'Erode': [11.3408, 77.7172], 'Jāmuria': [23.7, 87.08], 'Hābra': [22.83, 88.63], 'Ambāla': [30.3786, 76.7725], 'Mauli': [30.7194, 76.7181], 'Kolār': [13.1333, 78.1333], 'Shillong': [25.5744, 91.8789], 'Bhīmavaram': [16.5333, 81.5333], 'New Delhi': [28.7, 77.2], 'Mandsaur': [24.03, 75.08], 'Kumbakonam': [10.9805, 79.4], 'Tiruvannāmalai': [12.2604, 79.1], 'Chicacole': [18.3, 83.9], 'Bānkura': [23.25, 87.0667], 'Mandya': [12.5242, 76.8958], 'Hassan': [13.005, 76.1028], 'Yavatmāl': [20.4, 78.1333], 'Pīlibhīt': [28.64, 79.81], 'Pālghāt': [10.7792, 76.6547], 'Abohar': [30.1204, 74.29], 'Pālakollu': [16.5333, 81.7333], 'Kānchrāpāra': [22.97, 88.43], 'Port Blair': [11.6667, 92.75], 'Alīpur Duār': [26.4837, 89.5667], 'Hāthras': [27.6, 78.05], 'Guntakal': [15.17, 77.38], 'Navadwīp': [23.4088, 88.3657], 'Basīrhat': [22.6572, 88.8942], 'Hālīsahar': [22.95, 88.42], 'Rishra': [22.71, 88.35], 'Dharmavaram': [14.4142, 77.715], 'Baidyabāti': [22.79, 88.32], 'Darjeeling': [27.0417, 88.2631], 'Sopur': [34.3, 74.47], 'Gudivāda': [16.43, 80.99], 'Adilābād': [19.6667, 78.5333], 'Titāgarh': [22.74, 88.37], 'Chittaurgarh': [24.8894, 74.6239], 'Narasaraopet': [16.236, 80.054], 'Dam Dam': [22.62, 88.42], 'Vālpārai': [10.3204, 76.97], 'Osmānābād': [18.1667, 76.05], 'Champdani': [22.8, 88.37], 'Bangaon': [23.07, 88.82], 'Khardah': [22.72, 88.38], 'Tādpatri': [14.92, 78.02], 'Jalpāiguri': [26.5167, 88.7333], 'Suriāpet': [17.15, 79.6167], 'Tādepallegūdem': [16.8333, 81.5], 'Bānsbāria': [22.97, 88.4], 'Negapatam': [10.7667, 79.8333], 'Bhadreswar': [22.82, 88.35], 'Chilakalūrupet': [16.0892, 80.1672], 'Kalyani': [22.975, 88.4344], 'Gangtok': [27.33, 88.62], 'Kohīma': [25.6667, 94.1194], 'Khambhāt': [22.3131, 72.6194], 'Emmiganūr': [15.7333, 77.4833], 'Rāyachoti': [14.05, 78.75], 'Kāvali': [14.9123, 79.9944], 'Mancherāl': [18.8679, 79.4639], 'Kadiri': [14.12, 78.17], 'Ootacamund': [11.4086, 76.6939], 'Anakāpalle': [17.68, 83.02], 'Sirsilla': [18.38, 78.83], 'Kāmāreddipet': [18.3167, 78.35], 'Pāloncha': [17.5815, 80.6765], 'Kottagūdem': [17.55, 80.63], 'Tanuku': [16.75, 81.7], 'Bodhan': [18.67, 77.9], 'Karūr': [10.9504, 78.0833], 'Mangalagiri': [16.43, 80.55], 'Kairāna': [29.4, 77.2], 'Mārkāpur': [15.735, 79.27], 'Malaut': [30.19, 74.499], 'Bāpatla': [15.8889, 80.47], 'Badvel': [14.75, 79.05], 'Jorhāt': [26.75, 94.2167], 'Koratla': [18.82, 78.72], 'Pulivendla': [14.4167, 78.2333], 'Jaisalmer': [26.9167, 70.9167], 'Tādepalle': [16.4667, 80.6], 'Armūr': [18.79, 78.29], 'Jatani': [20.17, 85.7], 'Gadwāl': [16.23, 77.8], 'Nagari': [13.33, 79.58], 'Wanparti': [16.361, 78.0627], 'Ponnūru': [16.0667, 80.5667], 'Vinukonda': [16.05, 79.75], 'Itānagar': [27.1, 93.62], 'Tezpur': [26.6338, 92.8], 'Narasapur': [16.4333, 81.6833], 'Kothāpet': [19.3333, 79.4833], 'Mācherla': [16.48, 79.43], 'Kandukūr': [15.2165, 79.9042], 'Sāmalkot': [17.0531, 82.1695], 'Bobbili': [18.5667, 83.4167], 'Sattenapalle': [16.3962, 80.1497], 'Vrindāvan': [27.5806, 77.7006], 'Mandapeta': [16.87, 81.93], 'Belampalli': [19.0558, 79.4931], 'Bhīmunipatnam': [17.8864, 83.4471], 'Nāndod': [21.8704, 73.5026], 'Pithāpuram': [17.1167, 82.2667], 'Punganūru': [13.3667, 78.5833], 'Puttūr': [13.45, 79.55], 'Jalor': [25.35, 72.6167], 'Palmaner': [13.2, 78.75], 'Dholka': [22.72, 72.47], 'Jaggayyapeta': [16.892, 80.0976], 'Tuni': [17.35, 82.55], 'Amalāpuram': [16.5833, 82.0167], 'Jagtiāl': [18.8, 78.93], 'Vikārābād': [17.33, 77.9], 'Venkatagiri': [13.9667, 79.5833], 'Sihor': [21.7, 71.97], 'Jangaon': [17.72, 79.18], 'Mandamāri': [18.9822, 79.4811], 'Metpalli': [18.8297, 78.5878], 'Repalle': [16.02, 80.85], 'Bhainsa': [19.1, 77.9667], 'Jasdan': [22.03, 71.2], 'Jammalamadugu': [14.85, 78.38], 'Rāmeswaram': [9.28, 79.3], 'Addanki': [15.811, 79.9738], 'Nidadavole': [16.92, 81.67], 'Bodupāl': [17.4139, 78.5783], 'Rājgīr': [25.03, 85.42], 'Rajaori': [33.38, 74.3], 'Naini Tal': [29.3919, 79.4542], 'Channarāyapatna': [12.9, 76.39], 'Maihar': [24.27, 80.75], 'Panaji': [15.48, 73.83], 'Junnar': [19.2, 73.88], 'Amudālavalasa': [18.4167, 83.9], 'Damān': [20.417, 72.85], 'Kovvūr': [17.0167, 81.7333], 'Solan': [30.92, 77.12], 'Dwārka': [22.2403, 68.9686], 'Pathanāmthitta': [9.2647, 76.7872], 'Kodaikānal': [10.23, 77.48], 'Udhampur': [32.93, 75.13], 'Giddalūr': [15.3784, 78.9265], 'Yellandu': [17.6, 80.33], 'Shrīrangapattana': [12.4181, 76.6947], 'Angamāli': [10.196, 76.386], 'Umaria': [23.5245, 80.8365], 'Fatehpur Sīkri': [27.0911, 77.6611], 'Mangūr': [17.9373, 80.8185], 'Pedana': [16.2667, 81.1667], 'Uran': [18.89, 72.95], 'Chimākurti': [15.5819, 79.868], 'Devarkonda': [16.7, 78.9333], 'Bandipura': [34.4225, 74.6375], 'Silvassa': [20.2666, 73.0166], 'Pāmidi': [14.95, 77.5833], 'Narasannapeta': [18.4151, 84.0447], 'Jaynagar-Majilpur': [22.1772, 88.4258], 'Khed Brahma': [24.0299, 73.0463], 'Khajurāho': [24.85, 79.9333], 'Koilkuntla': [15.2333, 78.3167], 'Diu': [20.7197, 70.9904], 'Kulgam': [33.64, 75.02], 'Gauripur': [26.08, 89.97], 'Abu': [24.5925, 72.7083], 'Curchorem': [15.25, 74.1], 'Kavaratti': [10.5626, 72.6369], 'Panchkula': [30.6915, 76.8537], 'Kagaznāgār': [19.3316, 79.4661]}



def addlist(sender, app_data, user_data):
    # print(f"sender is: {sender}")
    # print(f"user_data is: {user_data}")

    if dpg.get_value(sender) == True:
        # print({Sender}, " added")
        selected_cities.append(user_data)
        print(selected_cities)

    else:
        # print(Sender, " is removed")
        selected_cities.remove(user_data)
        print(selected_cities)


def graph():
    cityList = []
# cit
# City = City(x,y)
    for i in selected_cities:
        cityList.append(City(x=cities[i][0], y=cities[i][1]))
    geneticAlgorithmPlot(population=cityList, popSize=100,
                         eliteSize=20, mutationRate=0.01, generations=500)
def graph_lucknow():
    cityList = []
# cit
# City = City(x,y)
    for i in selected_cities:
        cityList.append(City(x=lucknow_areas[i][0], y=lucknow_areas[i][1]))
    geneticAlgorithmPlot(population=cityList, popSize=100,
                         eliteSize=20, mutationRate=0.01, generations=500)



def calculate_city():
    # print("gg guysss")
    # print(cities["Delhi"][0])
    cityList = []
# cit
# City = City(x,y)
    for i in selected_cities:
        cityList.append(City(x=lucknow_areas[i][0], y=lucknow_areas[i][1]))
    # init_dist = 0
    # final_dist = 0

    cities_coordinates, init_dist, final_dist = geneticAlgorithm(population=cityList, popSize=100,
                                                                 eliteSize=20, mutationRate=0.01, generations=500)
    # print(cities_coordinates)

    city_routes = []
    # print(len(cities_coordinates))
    # print(cities["Delhi"][0], cities_coordinates[0].x)
    for i in range(len(cities_coordinates)):
        for city in lucknow_areas:

            if lucknow_areas[city][0] == cities_coordinates[i].x and lucknow_areas[city][1] == cities_coordinates[i].y:
                # print("added")
                city_routes.append(city)
    print("serial wise list of cities to be connected")
    print(city_routes)
    # with dpg.window

    # with dpg.window(label="TSP path Calculator"):
    with dpg.window(label="Solution", pos=(500, 0), width=600, height=700):
        dpg.add_text("Initial distance :")
        dpg.add_text(init_dist)
        dpg.add_text("Final Distance :")
        dpg.add_text(final_dist)
        dpg.add_text("Serial wise list of Areas to be connected -")
        for i in range(len(city_routes)):
            dpg.add_text(city_routes[i])


def calculate():
    # print("gg guysss")
    # print(cities["Delhi"][0])
    cityList = []
# cit
# City = City(x,y)
    for i in selected_cities:
        cityList.append(City(x=cities[i][0], y=cities[i][1]))
    # init_dist = 0
    # final_dist = 0

    cities_coordinates, init_dist, final_dist = geneticAlgorithm(population=cityList, popSize=100,
                                                                 eliteSize=20, mutationRate=0.01, generations=500)
    # print(cities_coordinates)

    city_routes = []
    # print(len(cities_coordinates))
    # print(cities["Delhi"][0], cities_coordinates[0].x)
    for i in range(len(cities_coordinates)):
        for city in cities:

            if cities[city][0] == cities_coordinates[i].x and cities[city][1] == cities_coordinates[i].y:
                # print("added")
                city_routes.append(city)
    print("serial wise list of cities to be connected")
    print(city_routes)
    # with dpg.window

    # with dpg.window(label="TSP path Calculator"):
    with dpg.window(label="Solution", pos=(500, 0), width=600, height=700):
        dpg.add_text("Initial distance :")
        dpg.add_text(init_dist)
        dpg.add_text("Final Distance :")
        dpg.add_text(final_dist)
        dpg.add_text("Serial wise list of cities to be connected -")
        for i in range(len(city_routes)):
            dpg.add_text(city_routes[i])


dpg.create_context()
selected_cities = []

with dpg.window(label="TSP path Calculator for Lucknow City",pos=(0,40),  width=500, height=650):

    dpg.add_text("Select the Areas you want to connect")
    dpg.add_checkbox(label='Alambag', callback=addlist, user_data='Alambag'),dpg.add_checkbox(label='Charbag', callback=addlist, user_data='Charbag'),dpg.add_checkbox(label='Aliganj', callback=addlist, user_data='Aliganj'),dpg.add_checkbox(label='Jankipuram', callback=addlist, user_data='Jankipuram'),dpg.add_checkbox(label='Aminabad', callback=addlist, user_data='Aminabad'),dpg.add_checkbox(label='Gomti nagar', callback=addlist, user_data='Gomti nagar'),dpg.add_checkbox(label='Indra nagar', callback=addlist, user_data='Indra nagar'),dpg.add_checkbox(label='Bakshi ka talab', callback=addlist, user_data='Bakshi ka talab'),dpg.add_checkbox(label='Husain ganj', callback=addlist, user_data='Husain ganj'),dpg.add_checkbox(label='Hzartganj', callback=addlist, user_data='Hzartganj'),dpg.add_checkbox(label='Badshah nagar', callback=addlist, user_data='Badshah nagar'),dpg.add_checkbox(label='Krishna nagar', callback=addlist, user_data='Krishna nagar'),dpg.add_checkbox(label='Transport nagar', callback=addlist, user_data='Transport nagar'),
    # dpg.add_checkbox(label='Surfarajgaj', callback=addlist, user_data='Surfarajgaj'),
    dpg.add_checkbox(label='Pandeyganj', callback=addlist, user_data='Pandeyganj'),dpg.add_checkbox(label='Shanti nagar', callback=addlist, user_data='Shanti nagar'),dpg.add_checkbox(label='Alamnagar', callback=addlist, user_data='Alamnagar'),dpg.add_checkbox(label='Kalyanpur', callback=addlist, user_data='Kalyanpur')

    dpg.add_button(label="Find the best Path",
                   callback=calculate_city)
    dpg.add_button(label="Show The Graph", callback=graph_lucknow)




with dpg.window(label="TSP path Calculator for India",  width=500, height=650):

    dpg.add_text("Select the cities you want to connect")

    dpg.add_checkbox(label='Delhi', callback=addlist, user_data='Delhi'), dpg.add_checkbox(label='Mumbai', callback=addlist, user_data='Mumbai'), dpg.add_checkbox(label='Kolkāta', callback=addlist, user_data='Kolkāta'), dpg.add_checkbox(label='Bangalore', callback=addlist, user_data='Bangalore'), dpg.add_checkbox(label='Chennai', callback=addlist, user_data='Chennai'), dpg.add_checkbox(label='Hyderābād', callback=addlist, user_data='Hyderābād'), dpg.add_checkbox(label='Pune', callback=addlist, user_data='Pune'), dpg.add_checkbox(label='Ahmedabad', callback=addlist, user_data='Ahmedabad'), dpg.add_checkbox(label='Sūrat', callback=addlist, user_data='Sūrat'), dpg.add_checkbox(label='Lucknow', callback=addlist, user_data='Lucknow'), dpg.add_checkbox(label='Jaipur', callback=addlist, user_data='Jaipur'), dpg.add_checkbox(label='Cawnpore', callback=addlist, user_data='Cawnpore'), dpg.add_checkbox(label='Mirzāpur', callback=addlist, user_data='Mirzāpur'), dpg.add_checkbox(label='Nāgpur', callback=addlist, user_data='Nāgpur'), dpg.add_checkbox(label='Ghāziābād', callback=addlist, user_data='Ghāziābād'), dpg.add_checkbox(label='Indore', callback=addlist, user_data='Indore'), dpg.add_checkbox(label='Vadodara', callback=addlist, user_data='Vadodara'), dpg.add_checkbox(label='Vishākhapatnam', callback=addlist, user_data='Vishākhapatnam'), dpg.add_checkbox(label='Bhopāl', callback=addlist, user_data='Bhopāl'), dpg.add_checkbox(label='Chinchvad', callback=addlist, user_data='Chinchvad'), dpg.add_checkbox(label='Patna', callback=addlist, user_data='Patna'), dpg.add_checkbox(label='Ludhiāna', callback=addlist, user_data='Ludhiāna'), dpg.add_checkbox(label='Āgra', callback=addlist, user_data='Āgra'), dpg.add_checkbox(label='Kalyān', callback=addlist, user_data='Kalyān'), dpg.add_checkbox(label='Madurai', callback=addlist, user_data='Madurai'), dpg.add_checkbox(label='Jamshedpur', callback=addlist, user_data='Jamshedpur'), dpg.add_checkbox(label='Nāsik', callback=addlist, user_data='Nāsik'), dpg.add_checkbox(label='Farīdābād', callback=addlist, user_data='Farīdābād'), dpg.add_checkbox(label='Aurangābād', callback=addlist, user_data='Aurangābād'), dpg.add_checkbox(label='Rājkot', callback=addlist, user_data='Rājkot'), dpg.add_checkbox(label='Meerut', callback=addlist, user_data='Meerut'), dpg.add_checkbox(label='Jabalpur', callback=addlist, user_data='Jabalpur'), dpg.add_checkbox(label='Thāne', callback=addlist, user_data='Thāne'), dpg.add_checkbox(label='Dhanbād', callback=addlist, user_data='Dhanbād'), dpg.add_checkbox(label='Allahābād', callback=addlist, user_data='Allahābād'), dpg.add_checkbox(label='Vārānasi', callback=addlist, user_data='Vārānasi'), dpg.add_checkbox(label='Srīnagar', callback=addlist, user_data='Srīnagar'), dpg.add_checkbox(label='Amritsar', callback=addlist, user_data='Amritsar'), dpg.add_checkbox(label='Alīgarh', callback=addlist, user_data='Alīgarh'), dpg.add_checkbox(label='Bhiwandi', callback=addlist, user_data='Bhiwandi'), dpg.add_checkbox(label='Gwalior', callback=addlist, user_data='Gwalior'), dpg.add_checkbox(label='Bhilai', callback=addlist, user_data='Bhilai'), dpg.add_checkbox(label='Hāora', callback=addlist, user_data='Hāora'), dpg.add_checkbox(label='Rānchi', callback=addlist, user_data='Rānchi'), dpg.add_checkbox(label='Bezwāda', callback=addlist, user_data='Bezwāda'), dpg.add_checkbox(label='Chandīgarh', callback=addlist, user_data='Chandīgarh'), dpg.add_checkbox(label='Mysore', callback=addlist, user_data='Mysore'), dpg.add_checkbox(label='Raipur', callback=addlist, user_data='Raipur'), dpg.add_checkbox(label='Kota', callback=addlist, user_data='Kota'), dpg.add_checkbox(label='Bareilly', callback=addlist, user_data='Bareilly'), dpg.add_checkbox(label='Jodhpur', callback=addlist, user_data='Jodhpur'), dpg.add_checkbox(label='Coimbatore', callback=addlist, user_data='Coimbatore'), dpg.add_checkbox(label='Dispur', callback=addlist, user_data='Dispur'), dpg.add_checkbox(label='Guwāhāti', callback=addlist, user_data='Guwāhāti'), dpg.add_checkbox(label='Solāpur', callback=addlist, user_data='Solāpur'), dpg.add_checkbox(label='Trichinopoly', callback=addlist, user_data='Trichinopoly'), dpg.add_checkbox(label='Hubli', callback=addlist, user_data='Hubli'), dpg.add_checkbox(label='Jalandhar', callback=addlist, user_data='Jalandhar'), dpg.add_checkbox(label='Bhubaneshwar', callback=addlist, user_data='Bhubaneshwar'), dpg.add_checkbox(label='Bhayandar', callback=addlist, user_data='Bhayandar'), dpg.add_checkbox(label='Morādābād', callback=addlist, user_data='Morādābād'), dpg.add_checkbox(label='Kolhāpur', callback=addlist, user_data='Kolhāpur'), dpg.add_checkbox(label='Thiruvananthapuram', callback=addlist, user_data='Thiruvananthapuram'), dpg.add_checkbox(label='Sahāranpur', callback=addlist, user_data='Sahāranpur'), dpg.add_checkbox(label='Warangal', callback=addlist, user_data='Warangal'), dpg.add_checkbox(label='Salem', callback=addlist, user_data='Salem'), dpg.add_checkbox(label='Mālegaon', callback=addlist, user_data='Mālegaon'), dpg.add_checkbox(label='Kochi', callback=addlist, user_data='Kochi'), dpg.add_checkbox(label='Gorakhpur', callback=addlist, user_data='Gorakhpur'), dpg.add_checkbox(label='Shimoga', callback=addlist, user_data='Shimoga'), dpg.add_checkbox(label='Tiruppūr', callback=addlist, user_data='Tiruppūr'), dpg.add_checkbox(label='Guntūr', callback=addlist, user_data='Guntūr'), dpg.add_checkbox(label='Raurkela', callback=addlist, user_data='Raurkela'), dpg.add_checkbox(label='Mangalore', callback=addlist, user_data='Mangalore'), dpg.add_checkbox(label='Nānded', callback=addlist, user_data='Nānded'), dpg.add_checkbox(label='Cuttack', callback=addlist, user_data='Cuttack'), dpg.add_checkbox(label='Chānda', callback=addlist, user_data='Chānda'), dpg.add_checkbox(label='Dehra Dūn', callback=addlist, user_data='Dehra Dūn'), dpg.add_checkbox(label='Durgāpur', callback=addlist, user_data='Durgāpur'), dpg.add_checkbox(label='Āsansol', callback=addlist, user_data='Āsansol'), dpg.add_checkbox(label='Bhāvnagar', callback=addlist, user_data='Bhāvnagar'), dpg.add_checkbox(label='Amrāvati', callback=addlist, user_data='Amrāvati'), dpg.add_checkbox(label='Nellore', callback=addlist, user_data='Nellore'), dpg.add_checkbox(label='Ajmer', callback=addlist, user_data='Ajmer'), dpg.add_checkbox(label='Tinnevelly', callback=addlist, user_data='Tinnevelly'), dpg.add_checkbox(label='Bīkaner', callback=addlist, user_data='Bīkaner'), dpg.add_checkbox(label='Agartala', callback=addlist, user_data='Agartala'), dpg.add_checkbox(label='Ujjain', callback=addlist, user_data='Ujjain'), dpg.add_checkbox(label='Jhānsi', callback=addlist, user_data='Jhānsi'), dpg.add_checkbox(label='Ulhāsnagar', callback=addlist, user_data='Ulhāsnagar'), dpg.add_checkbox(label='Davangere', callback=addlist, user_data='Davangere'), dpg.add_checkbox(label='Jammu', callback=addlist, user_data='Jammu'), dpg.add_checkbox(label='Belgaum', callback=addlist, user_data='Belgaum'), dpg.add_checkbox(label='Gulbarga', callback=addlist, user_data='Gulbarga'), dpg.add_checkbox(label='Jāmnagar', callback=addlist, user_data='Jāmnagar'), dpg.add_checkbox(label='Dhūlia', callback=addlist, user_data='Dhūlia'), dpg.add_checkbox(label='Gaya', callback=addlist, user_data='Gaya'), dpg.add_checkbox(label='Jalgaon', callback=addlist, user_data='Jalgaon'), dpg.add_checkbox(label='Kurnool', callback=addlist, user_data='Kurnool'), dpg.add_checkbox(label='Udaipur', callback=addlist, user_data='Udaipur'), dpg.add_checkbox(label='Bellary', callback=addlist, user_data='Bellary'), dpg.add_checkbox(label='Sāngli', callback=addlist, user_data='Sāngli'), dpg.add_checkbox(label='Tuticorin', callback=addlist, user_data='Tuticorin'), dpg.add_checkbox(label='Calicut', callback=addlist, user_data='Calicut'), dpg.add_checkbox(label='Akola', callback=addlist, user_data='Akola'), dpg.add_checkbox(label='Bhāgalpur', callback=addlist, user_data='Bhāgalpur'), dpg.add_checkbox(label='Sīkar', callback=addlist, user_data='Sīkar'), dpg.add_checkbox(label='Tumkūr', callback=addlist, user_data='Tumkūr'), dpg.add_checkbox(label='Quilon', callback=addlist, user_data='Quilon'), dpg.add_checkbox(label='Muzaffarnagar', callback=addlist, user_data='Muzaffarnagar'), dpg.add_checkbox(label='Bhīlwāra', callback=addlist, user_data='Bhīlwāra'), dpg.add_checkbox(label='Nizāmābād', callback=addlist, user_data='Nizāmābād'), dpg.add_checkbox(label='Bhātpāra', callback=addlist, user_data='Bhātpāra'), dpg.add_checkbox(label='Kākināda', callback=addlist, user_data='Kākināda'), dpg.add_checkbox(label='Parbhani', callback=addlist, user_data='Parbhani'), dpg.add_checkbox(label='Pānihāti', callback=addlist, user_data='Pānihāti'), dpg.add_checkbox(label='Lātūr', callback=addlist, user_data='Lātūr'), dpg.add_checkbox(label='Rohtak', callback=addlist, user_data='Rohtak'), dpg.add_checkbox(label='Rājapālaiyam', callback=addlist, user_data='Rājapālaiyam'), dpg.add_checkbox(label='Ahmadnagar', callback=addlist, user_data='Ahmadnagar'), dpg.add_checkbox(label='Cuddapah', callback=addlist, user_data='Cuddapah'), dpg.add_checkbox(label='Rājahmundry', callback=addlist, user_data='Rājahmundry'), dpg.add_checkbox(label='Alwar', callback=addlist, user_data='Alwar'), dpg.add_checkbox(label='Muzaffarpur', callback=addlist, user_data='Muzaffarpur'), dpg.add_checkbox(label='Bilāspur', callback=addlist, user_data='Bilāspur'), dpg.add_checkbox(label='Mathura', callback=addlist, user_data='Mathura'), dpg.add_checkbox(label='Kāmārhāti', callback=addlist, user_data='Kāmārhāti'), dpg.add_checkbox(label='Patiāla', callback=addlist, user_data='Patiāla'), dpg.add_checkbox(label='Saugor', callback=addlist, user_data='Saugor'), dpg.add_checkbox(label='Bijāpur', callback=addlist, user_data='Bijāpur'), dpg.add_checkbox(label='Brahmapur', callback=addlist, user_data='Brahmapur'), dpg.add_checkbox(label='Shāhjānpur', callback=addlist, user_data='Shāhjānpur'), dpg.add_checkbox(label='Trichūr', callback=addlist, user_data='Trichūr'), dpg.add_checkbox(label='Barddhamān', callback=addlist, user_data='Barddhamān'), dpg.add_checkbox(label='Kulti', callback=addlist, user_data='Kulti'), dpg.add_checkbox(label='Sambalpur', callback=addlist, user_data='Sambalpur'), dpg.add_checkbox(label='Purnea', callback=addlist, user_data='Purnea'), dpg.add_checkbox(label='Hisar', callback=addlist, user_data='Hisar'), dpg.add_checkbox(label='Fīrozābād', callback=addlist, user_data='Fīrozābād'), dpg.add_checkbox(label='Bīdar', callback=addlist, user_data='Bīdar'), dpg.add_checkbox(label='Rāmpur', callback=addlist, user_data='Rāmpur'), dpg.add_checkbox(label='Shiliguri', callback=addlist, user_data='Shiliguri'), dpg.add_checkbox(label='Bāli', callback=addlist, user_data='Bāli'), dpg.add_checkbox(label='Pānīpat', callback=addlist, user_data='Pānīpat'), dpg.add_checkbox(label='Karīmnagar', callback=addlist, user_data='Karīmnagar'), dpg.add_checkbox(label='Bhuj', callback=addlist, user_data='Bhuj'), dpg.add_checkbox(label='Ichalkaranji', callback=addlist, user_data='Ichalkaranji'), dpg.add_checkbox(label='Tirupati', callback=addlist, user_data='Tirupati'), dpg.add_checkbox(label='Hospet', callback=addlist, user_data='Hospet'), dpg.add_checkbox(label='Āīzawl', callback=addlist, user_data='Āīzawl'), dpg.add_checkbox(label='Sannai', callback=addlist, user_data='Sannai'), dpg.add_checkbox(label='Bārāsat', callback=addlist, user_data='Bārāsat'), dpg.add_checkbox(label='Ratlām', callback=addlist, user_data='Ratlām'), dpg.add_checkbox(label='Handwāra', callback=addlist, user_data='Handwāra'), dpg.add_checkbox(label='Drug', callback=addlist, user_data='Drug'), dpg.add_checkbox(label='Imphāl', callback=addlist, user_data='Imphāl'), dpg.add_checkbox(label='Anantapur', callback=addlist, user_data='Anantapur'), dpg.add_checkbox(label='Etāwah', callback=addlist, user_data='Etāwah'), dpg.add_checkbox(label='Rāichūr', callback=addlist, user_data='Rāichūr'), dpg.add_checkbox(label='Ongole', callback=addlist, user_data='Ongole'), dpg.add_checkbox(label='Bharatpur', callback=addlist, user_data='Bharatpur'), dpg.add_checkbox(label='Begusarai', callback=addlist, user_data='Begusarai'), dpg.add_checkbox(label='Sonīpat', callback=addlist, user_data='Sonīpat'), dpg.add_checkbox(label='Rāmgundam', callback=addlist, user_data='Rāmgundam'), dpg.add_checkbox(label='Hāpur', callback=addlist, user_data='Hāpur'), dpg.add_checkbox(label='Uluberiya', callback=addlist, user_data='Uluberiya'), dpg.add_checkbox(label='Porbandar', callback=addlist, user_data='Porbandar'), dpg.add_checkbox(label='Pāli', callback=addlist, user_data='Pāli'), dpg.add_checkbox(label='Vizianagaram', callback=addlist, user_data='Vizianagaram'), dpg.add_checkbox(label='Puducherry', callback=addlist, user_data='Puducherry'), dpg.add_checkbox(label='Karnāl', callback=addlist, user_data='Karnāl'), dpg.add_checkbox(label='Nāgercoil', callback=addlist, user_data='Nāgercoil'), dpg.add_checkbox(label='Tanjore', callback=addlist, user_data='Tanjore'), dpg.add_checkbox(label='Sambhal', callback=addlist, user_data='Sambhal'), dpg.add_checkbox(label='Naihāti', callback=addlist, user_data='Naihāti'), dpg.add_checkbox(label='Secunderābād', callback=addlist, user_data='Secunderābād'), dpg.add_checkbox(label='Kharagpur', callback=addlist, user_data='Kharagpur'), dpg.add_checkbox(label='Dindigul', callback=addlist, user_data='Dindigul'), dpg.add_checkbox(label='Shimla', callback=addlist, user_data='Shimla'), dpg.add_checkbox(label='Ingrāj Bāzār', callback=addlist, user_data='Ingrāj Bāzār'), dpg.add_checkbox(label='Ellore', callback=addlist, user_data='Ellore'), dpg.add_checkbox(label='Puri', callback=addlist, user_data='Puri'), dpg.add_checkbox(label='Haldia', callback=addlist, user_data='Haldia'), dpg.add_checkbox(label='Nandyāl', callback=addlist, user_data='Nandyāl'), dpg.add_checkbox(label='Bulandshahr', callback=addlist, user_data='Bulandshahr'), dpg.add_checkbox(label='Chakradharpur', callback=addlist, user_data='Chakradharpur'), dpg.add_checkbox(label='Bhiwāni', callback=addlist, user_data='Bhiwāni'), dpg.add_checkbox(label='Gurgaon', callback=addlist, user_data='Gurgaon'), dpg.add_checkbox(label='Burhānpur', callback=addlist, user_data='Burhānpur'), dpg.add_checkbox(label='Khammam', callback=addlist, user_data='Khammam'), dpg.add_checkbox(label='Madhyamgram', callback=addlist, user_data='Madhyamgram'), dpg.add_checkbox(label='Ghāndīnagar', callback=addlist, user_data='Ghāndīnagar'), dpg.add_checkbox(label='Baharampur', callback=addlist, user_data='Baharampur'), dpg.add_checkbox(label='Mahbūbnagar', callback=addlist, user_data='Mahbūbnagar'), dpg.add_checkbox(label='Mahesāna', callback=addlist, user_data='Mahesāna'), dpg.add_checkbox(label='Ādoni', callback=addlist, user_data='Ādoni'), dpg.add_checkbox(label='Rāiganj', callback=addlist, user_data='Rāiganj'), dpg.add_checkbox(label='Bhusāval', callback=addlist, user_data='Bhusāval'), dpg.add_checkbox(label='Bahraigh', callback=addlist, user_data='Bahraigh'), dpg.add_checkbox(label='Shrīrāmpur', callback=addlist, user_data='Shrīrāmpur'), dpg.add_checkbox(label='Tonk', callback=addlist, user_data='Tonk'), dpg.add_checkbox(label='Sirsa', callback=addlist, user_data='Sirsa'), dpg.add_checkbox(label='Jaunpur', callback=addlist, user_data='Jaunpur'), dpg.add_checkbox(label='Madanapalle', callback=addlist, user_data='Madanapalle'), dpg.add_checkbox(label='Hugli', callback=addlist,
      user_data='Hugli'), dpg.add_checkbox(label='Vellore', callback=addlist, user_data='Vellore'), dpg.add_checkbox(label='Alleppey', callback=addlist, user_data='Alleppey'), dpg.add_checkbox(label='Cuddalore', callback=addlist, user_data='Cuddalore'), dpg.add_checkbox(label='Deo', callback=addlist, user_data='Deo'), dpg.add_checkbox(label='Chīrāla', callback=addlist, user_data='Chīrāla'), dpg.add_checkbox(label='Machilīpatnam', callback=addlist, user_data='Machilīpatnam'), dpg.add_checkbox(label='Medinīpur', callback=addlist, user_data='Medinīpur'), dpg.add_checkbox(label='Bāramūla', callback=addlist, user_data='Bāramūla'), dpg.add_checkbox(label='Chandannagar', callback=addlist, user_data='Chandannagar'), dpg.add_checkbox(label='Fatehpur', callback=addlist, user_data='Fatehpur'), dpg.add_checkbox(label='Udipi', callback=addlist, user_data='Udipi'), dpg.add_checkbox(label='Tenāli', callback=addlist, user_data='Tenāli'), dpg.add_checkbox(label='Sitalpur', callback=addlist, user_data='Sitalpur'), dpg.add_checkbox(label='Conjeeveram', callback=addlist, user_data='Conjeeveram'), dpg.add_checkbox(label='Proddatūr', callback=addlist, user_data='Proddatūr'), dpg.add_checkbox(label='Navsāri', callback=addlist, user_data='Navsāri'), dpg.add_checkbox(label='Godhra', callback=addlist, user_data='Godhra'), dpg.add_checkbox(label='Budaun', callback=addlist, user_data='Budaun'), dpg.add_checkbox(label='Chittoor', callback=addlist, user_data='Chittoor'), dpg.add_checkbox(label='Harīpur', callback=addlist, user_data='Harīpur'), dpg.add_checkbox(label='Saharsa', callback=addlist, user_data='Saharsa'), dpg.add_checkbox(label='Vidisha', callback=addlist, user_data='Vidisha'), dpg.add_checkbox(label='Pathānkot', callback=addlist, user_data='Pathānkot'), dpg.add_checkbox(label='Nalgonda', callback=addlist, user_data='Nalgonda'), dpg.add_checkbox(label='Dibrugarh', callback=addlist, user_data='Dibrugarh'), dpg.add_checkbox(label='Bālurghāt', callback=addlist, user_data='Bālurghāt'), dpg.add_checkbox(label='Krishnanagar', callback=addlist, user_data='Krishnanagar'), dpg.add_checkbox(label='Fyzābād', callback=addlist, user_data='Fyzābād'), dpg.add_checkbox(label='Silchar', callback=addlist, user_data='Silchar'), dpg.add_checkbox(label='Shāntipur', callback=addlist, user_data='Shāntipur'), dpg.add_checkbox(label='Hindupur', callback=addlist, user_data='Hindupur'), dpg.add_checkbox(label='Erode', callback=addlist, user_data='Erode'), dpg.add_checkbox(label='Jāmuria', callback=addlist, user_data='Jāmuria'), dpg.add_checkbox(label='Hābra', callback=addlist, user_data='Hābra'), dpg.add_checkbox(label='Ambāla', callback=addlist, user_data='Ambāla'), dpg.add_checkbox(label='Mauli', callback=addlist, user_data='Mauli'), dpg.add_checkbox(label='Kolār', callback=addlist, user_data='Kolār'), dpg.add_checkbox(label='Shillong', callback=addlist, user_data='Shillong'), dpg.add_checkbox(label='Bhīmavaram', callback=addlist, user_data='Bhīmavaram'), dpg.add_checkbox(label='New Delhi', callback=addlist, user_data='New Delhi'), dpg.add_checkbox(label='Mandsaur', callback=addlist, user_data='Mandsaur'), dpg.add_checkbox(label='Kumbakonam', callback=addlist, user_data='Kumbakonam'), dpg.add_checkbox(label='Tiruvannāmalai', callback=addlist, user_data='Tiruvannāmalai'), dpg.add_checkbox(label='Chicacole', callback=addlist, user_data='Chicacole'), dpg.add_checkbox(label='Bānkura', callback=addlist, user_data='Bānkura'), dpg.add_checkbox(label='Mandya', callback=addlist, user_data='Mandya'), dpg.add_checkbox(label='Hassan', callback=addlist, user_data='Hassan'), dpg.add_checkbox(label='Yavatmāl', callback=addlist, user_data='Yavatmāl'), dpg.add_checkbox(label='Pīlibhīt', callback=addlist, user_data='Pīlibhīt'), dpg.add_checkbox(label='Pālghāt', callback=addlist, user_data='Pālghāt'), dpg.add_checkbox(label='Abohar', callback=addlist, user_data='Abohar'), dpg.add_checkbox(label='Pālakollu', callback=addlist, user_data='Pālakollu'), dpg.add_checkbox(label='Kānchrāpāra', callback=addlist, user_data='Kānchrāpāra'), dpg.add_checkbox(label='Port Blair', callback=addlist, user_data='Port Blair'), dpg.add_checkbox(label='Alīpur Duār', callback=addlist, user_data='Alīpur Duār'), dpg.add_checkbox(label='Hāthras', callback=addlist, user_data='Hāthras'), dpg.add_checkbox(label='Guntakal', callback=addlist, user_data='Guntakal'), dpg.add_checkbox(label='Navadwīp', callback=addlist, user_data='Navadwīp'), dpg.add_checkbox(label='Basīrhat', callback=addlist, user_data='Basīrhat'), dpg.add_checkbox(label='Hālīsahar', callback=addlist, user_data='Hālīsahar'), dpg.add_checkbox(label='Rishra', callback=addlist, user_data='Rishra'), dpg.add_checkbox(label='Dharmavaram', callback=addlist, user_data='Dharmavaram'), dpg.add_checkbox(label='Baidyabāti', callback=addlist, user_data='Baidyabāti'), dpg.add_checkbox(label='Darjeeling', callback=addlist, user_data='Darjeeling'), dpg.add_checkbox(label='Sopur', callback=addlist, user_data='Sopur'), dpg.add_checkbox(label='Gudivāda', callback=addlist, user_data='Gudivāda'), dpg.add_checkbox(label='Adilābād', callback=addlist, user_data='Adilābād'), dpg.add_checkbox(label='Titāgarh', callback=addlist, user_data='Titāgarh'), dpg.add_checkbox(label='Chittaurgarh', callback=addlist, user_data='Chittaurgarh'), dpg.add_checkbox(label='Narasaraopet', callback=addlist, user_data='Narasaraopet'), dpg.add_checkbox(label='Dam Dam', callback=addlist, user_data='Dam Dam'), dpg.add_checkbox(label='Vālpārai', callback=addlist, user_data='Vālpārai'), dpg.add_checkbox(label='Osmānābād', callback=addlist, user_data='Osmānābād'), dpg.add_checkbox(label='Champdani', callback=addlist, user_data='Champdani'), dpg.add_checkbox(label='Bangaon', callback=addlist, user_data='Bangaon'), dpg.add_checkbox(label='Khardah', callback=addlist, user_data='Khardah'), dpg.add_checkbox(label='Tādpatri', callback=addlist, user_data='Tādpatri'), dpg.add_checkbox(label='Jalpāiguri', callback=addlist, user_data='Jalpāiguri'), dpg.add_checkbox(label='Suriāpet', callback=addlist, user_data='Suriāpet'), dpg.add_checkbox(label='Tādepallegūdem', callback=addlist, user_data='Tādepallegūdem'), dpg.add_checkbox(label='Bānsbāria', callback=addlist, user_data='Bānsbāria'), dpg.add_checkbox(label='Negapatam', callback=addlist, user_data='Negapatam'), dpg.add_checkbox(label='Bhadreswar', callback=addlist, user_data='Bhadreswar'), dpg.add_checkbox(label='Chilakalūrupet', callback=addlist, user_data='Chilakalūrupet'), dpg.add_checkbox(label='Kalyani', callback=addlist, user_data='Kalyani'), dpg.add_checkbox(label='Gangtok', callback=addlist, user_data='Gangtok'), dpg.add_checkbox(label='Kohīma', callback=addlist, user_data='Kohīma'), dpg.add_checkbox(label='Khambhāt', callback=addlist, user_data='Khambhāt'), dpg.add_checkbox(label='Aurangābād', callback=addlist, user_data='Aurangābād'), dpg.add_checkbox(label='Emmiganūr', callback=addlist, user_data='Emmiganūr'), dpg.add_checkbox(label='Rāyachoti', callback=addlist, user_data='Rāyachoti'), dpg.add_checkbox(label='Kāvali', callback=addlist, user_data='Kāvali'), dpg.add_checkbox(label='Mancherāl', callback=addlist, user_data='Mancherāl'), dpg.add_checkbox(label='Kadiri', callback=addlist, user_data='Kadiri'), dpg.add_checkbox(label='Ootacamund', callback=addlist, user_data='Ootacamund'), dpg.add_checkbox(label='Anakāpalle', callback=addlist, user_data='Anakāpalle'), dpg.add_checkbox(label='Sirsilla', callback=addlist, user_data='Sirsilla'), dpg.add_checkbox(label='Kāmāreddipet', callback=addlist, user_data='Kāmāreddipet'), dpg.add_checkbox(label='Pāloncha', callback=addlist, user_data='Pāloncha'), dpg.add_checkbox(label='Kottagūdem', callback=addlist, user_data='Kottagūdem'), dpg.add_checkbox(label='Tanuku', callback=addlist, user_data='Tanuku'), dpg.add_checkbox(label='Bodhan', callback=addlist, user_data='Bodhan'), dpg.add_checkbox(label='Karūr', callback=addlist, user_data='Karūr'), dpg.add_checkbox(label='Mangalagiri', callback=addlist, user_data='Mangalagiri'), dpg.add_checkbox(label='Kairāna', callback=addlist, user_data='Kairāna'), dpg.add_checkbox(label='Mārkāpur', callback=addlist, user_data='Mārkāpur'), dpg.add_checkbox(label='Malaut', callback=addlist, user_data='Malaut'), dpg.add_checkbox(label='Bāpatla', callback=addlist, user_data='Bāpatla'), dpg.add_checkbox(label='Badvel', callback=addlist, user_data='Badvel'), dpg.add_checkbox(label='Jorhāt', callback=addlist, user_data='Jorhāt'), dpg.add_checkbox(label='Koratla', callback=addlist, user_data='Koratla'), dpg.add_checkbox(label='Pulivendla', callback=addlist, user_data='Pulivendla'), dpg.add_checkbox(label='Jaisalmer', callback=addlist, user_data='Jaisalmer'), dpg.add_checkbox(label='Tādepalle', callback=addlist, user_data='Tādepalle'), dpg.add_checkbox(label='Armūr', callback=addlist, user_data='Armūr'), dpg.add_checkbox(label='Jatani', callback=addlist, user_data='Jatani'), dpg.add_checkbox(label='Gadwāl', callback=addlist, user_data='Gadwāl'), dpg.add_checkbox(label='Nagari', callback=addlist, user_data='Nagari'), dpg.add_checkbox(label='Wanparti', callback=addlist, user_data='Wanparti'), dpg.add_checkbox(label='Ponnūru', callback=addlist, user_data='Ponnūru'), dpg.add_checkbox(label='Vinukonda', callback=addlist, user_data='Vinukonda'), dpg.add_checkbox(label='Itānagar', callback=addlist, user_data='Itānagar'), dpg.add_checkbox(label='Tezpur', callback=addlist, user_data='Tezpur'), dpg.add_checkbox(label='Narasapur', callback=addlist, user_data='Narasapur'), dpg.add_checkbox(label='Kothāpet', callback=addlist, user_data='Kothāpet'), dpg.add_checkbox(label='Mācherla', callback=addlist, user_data='Mācherla'), dpg.add_checkbox(label='Kandukūr', callback=addlist, user_data='Kandukūr'), dpg.add_checkbox(label='Sāmalkot', callback=addlist, user_data='Sāmalkot'), dpg.add_checkbox(label='Bobbili', callback=addlist, user_data='Bobbili'), dpg.add_checkbox(label='Sattenapalle', callback=addlist, user_data='Sattenapalle'), dpg.add_checkbox(label='Vrindāvan', callback=addlist, user_data='Vrindāvan'), dpg.add_checkbox(label='Mandapeta', callback=addlist, user_data='Mandapeta'), dpg.add_checkbox(label='Belampalli', callback=addlist, user_data='Belampalli'), dpg.add_checkbox(label='Bhīmunipatnam', callback=addlist, user_data='Bhīmunipatnam'), dpg.add_checkbox(label='Nāndod', callback=addlist, user_data='Nāndod'), dpg.add_checkbox(label='Pithāpuram', callback=addlist, user_data='Pithāpuram'), dpg.add_checkbox(label='Punganūru', callback=addlist, user_data='Punganūru'), dpg.add_checkbox(label='Puttūr', callback=addlist, user_data='Puttūr'), dpg.add_checkbox(label='Jalor', callback=addlist, user_data='Jalor'), dpg.add_checkbox(label='Palmaner', callback=addlist, user_data='Palmaner'), dpg.add_checkbox(label='Dholka', callback=addlist, user_data='Dholka'), dpg.add_checkbox(label='Jaggayyapeta', callback=addlist, user_data='Jaggayyapeta'), dpg.add_checkbox(label='Tuni', callback=addlist, user_data='Tuni'), dpg.add_checkbox(label='Amalāpuram', callback=addlist, user_data='Amalāpuram'), dpg.add_checkbox(label='Jagtiāl', callback=addlist, user_data='Jagtiāl'), dpg.add_checkbox(label='Vikārābād', callback=addlist, user_data='Vikārābād'), dpg.add_checkbox(label='Venkatagiri', callback=addlist, user_data='Venkatagiri'), dpg.add_checkbox(label='Sihor', callback=addlist, user_data='Sihor'), dpg.add_checkbox(label='Jangaon', callback=addlist, user_data='Jangaon'), dpg.add_checkbox(label='Mandamāri', callback=addlist, user_data='Mandamāri'), dpg.add_checkbox(label='Metpalli', callback=addlist, user_data='Metpalli'), dpg.add_checkbox(label='Repalle', callback=addlist, user_data='Repalle'), dpg.add_checkbox(label='Bhainsa', callback=addlist, user_data='Bhainsa'), dpg.add_checkbox(label='Jasdan', callback=addlist, user_data='Jasdan'), dpg.add_checkbox(label='Jammalamadugu', callback=addlist, user_data='Jammalamadugu'), dpg.add_checkbox(label='Rāmeswaram', callback=addlist, user_data='Rāmeswaram'), dpg.add_checkbox(label='Addanki', callback=addlist, user_data='Addanki'), dpg.add_checkbox(label='Nidadavole', callback=addlist, user_data='Nidadavole'), dpg.add_checkbox(label='Bodupāl', callback=addlist, user_data='Bodupāl'), dpg.add_checkbox(label='Rājgīr', callback=addlist, user_data='Rājgīr'), dpg.add_checkbox(label='Rajaori', callback=addlist, user_data='Rajaori'), dpg.add_checkbox(label='Naini Tal', callback=addlist, user_data='Naini Tal'), dpg.add_checkbox(label='Channarāyapatna', callback=addlist, user_data='Channarāyapatna'), dpg.add_checkbox(label='Maihar', callback=addlist, user_data='Maihar'), dpg.add_checkbox(label='Panaji', callback=addlist, user_data='Panaji'), dpg.add_checkbox(label='Junnar', callback=addlist, user_data='Junnar'), dpg.add_checkbox(label='Amudālavalasa', callback=addlist, user_data='Amudālavalasa'), dpg.add_checkbox(label='Damān', callback=addlist, user_data='Damān'), dpg.add_checkbox(label='Kovvūr', callback=addlist, user_data='Kovvūr'), dpg.add_checkbox(label='Solan', callback=addlist, user_data='Solan'), dpg.add_checkbox(label='Dwārka', callback=addlist, user_data='Dwārka'), dpg.add_checkbox(label='Pathanāmthitta', callback=addlist, user_data='Pathanāmthitta'), dpg.add_checkbox(label='Kodaikānal', callback=addlist, user_data='Kodaikānal'), dpg.add_checkbox(label='Udhampur', callback=addlist, user_data='Udhampur'), dpg.add_checkbox(label='Giddalūr', callback=addlist, user_data='Giddalūr'), dpg.add_checkbox(label='Yellandu', callback=addlist, user_data='Yellandu'), dpg.add_checkbox(label='Shrīrangapattana', callback=addlist, user_data='Shrīrangapattana'), dpg.add_checkbox(label='Angamāli', callback=addlist, user_data='Angamāli'), dpg.add_checkbox(label='Umaria', callback=addlist, user_data='Umaria'), dpg.add_checkbox(label='Fatehpur Sīkri', callback=addlist, user_data='Fatehpur Sīkri'), dpg.add_checkbox(label='Mangūr', callback=addlist, user_data='Mangūr'), dpg.add_checkbox(label='Pedana', callback=addlist, user_data='Pedana'), dpg.add_checkbox(label='Uran', callback=addlist, user_data='Uran'), dpg.add_checkbox(label='Chimākurti', callback=addlist, user_data='Chimākurti'), dpg.add_checkbox(label='Devarkonda', callback=addlist, user_data='Devarkonda'), dpg.add_checkbox(label='Bandipura', callback=addlist, user_data='Bandipura'), dpg.add_checkbox(label='Silvassa', callback=addlist, user_data='Silvassa'), dpg.add_checkbox(label='Pāmidi', callback=addlist, user_data='Pāmidi'), dpg.add_checkbox(label='Narasannapeta', callback=addlist, user_data='Narasannapeta'), dpg.add_checkbox(label='Jaynagar-Majilpur', callback=addlist, user_data='Jaynagar-Majilpur'), dpg.add_checkbox(label='Khed Brahma', callback=addlist, user_data='Khed Brahma'), dpg.add_checkbox(label='Khajurāho', callback=addlist, user_data='Khajurāho'), dpg.add_checkbox(label='Koilkuntla', callback=addlist, user_data='Koilkuntla'), dpg.add_checkbox(label='Diu', callback=addlist, user_data='Diu'), dpg.add_checkbox(label='Kulgam', callback=addlist, user_data='Kulgam'), dpg.add_checkbox(label='Gauripur', callback=addlist, user_data='Gauripur'), dpg.add_checkbox(label='Abu', callback=addlist, user_data='Abu'), dpg.add_checkbox(label='Curchorem', callback=addlist, user_data='Curchorem'), dpg.add_checkbox(label='Kavaratti', callback=addlist, user_data='Kavaratti'), dpg.add_checkbox(label='Panchkula', callback=addlist, user_data='Panchkula'), dpg.add_checkbox(label='Kagaznāgār', callback=addlist, user_data='Kagaznāgār')

   

    dpg.add_button(label="Find the best Path",
                   callback=calculate)
    dpg.add_button(label="Show The Graph", callback=graph)




dpg.create_viewport(
    title='Travelling Salesman Problem by NoOne', width=1000, height=700)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
