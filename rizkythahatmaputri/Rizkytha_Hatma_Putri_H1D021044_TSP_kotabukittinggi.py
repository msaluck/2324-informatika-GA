# -*- coding: utf-8 -*-
"""Rizkytha Hatma Putri_H1D021044_TSP_kotabukittinggi.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1UxRBI1-nDf1bDUI8BfqwxDxBvMRPkxzA
"""

import math
import random as rd
import numpy as np

"""# nilai fitness
menghitung nilai fitness dari suatu kromosom (kombinasi) berdasarkan total jarak yang harus ditempuh. kromosom direpresentasikan sebagai indeks node yang harus dikunjungi.
"""

#menghitung nilai fitness sebagai satu dibagi total jarak
def nilaiFitness(kombinasi, node):
    total_jarak = 0
    for i in range(1, len(kombinasi)):
        total_jarak += math.sqrt((node[kombinasi[i]][0] - node[kombinasi[i-1]][0])**2 + (node[kombinasi[i-1]][1] - node[kombinasi[i]][1])**2)
    return 1 / total_jarak

"""# seleksi parent
digunakan untuk memilih orang tua (kromosom) berdasarkan probabilitas fitness.
metode roulette wheel digunakan untuk memilih kromosom secara acak berdasarkan probabilitasnya.
"""

#menggunakan roulette wheel selection untuk memilih orang tua
def rouletteWheelSelection(populasi, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    cumulative_probs = np.cumsum(selection_probs)
    rand_val = np.random.rand()
    for i, cumulative_prob in enumerate(cumulative_probs):
        if rand_val < cumulative_prob:
            return populasi[i]

"""# generate kromosom
menghasilkan populasi awal dengan kromosom-kromosom acak. setiap kromosom berisi indeks node yang diacak.
"""

#membuat kromosom acak
def generateKromosom(jumlahPopulasi, panjangKromosom):
    populasi = []
    for _ in range(jumlahPopulasi):
        kromosom = np.arange(panjangKromosom)
        np.random.shuffle(kromosom)
        populasi.append(list(kromosom))
    return populasi

"""# crossover
melakukan crossover antara dua kromosom dengan probabilitas pC.
poin pemotongan (crossover point) dipilih secara acak.
anak kromosom terbentuk dengan menggabungkan bagian kromosom orang tua yang berbeda.
"""

#proses crossover
def crossover(kromosom1, kromosom2, pC):
    if np.random.rand() <= pC:
        point = np.random.randint(1, len(kromosom1) - 1)
        tmpKromosom1 = kromosom1[:point] + [gene for gene in kromosom2 if gene not in kromosom1[:point]]
        tmpKromosom2 = kromosom2[:point] + [gene for gene in kromosom1 if gene not in kromosom2[:point]]
        return tmpKromosom1, tmpKromosom2
    else:
        return kromosom1, kromosom2

"""# mutasi
mengubah beberapa gen dalam kromosom dengan probabilitas pM.
gen yang dipilih secara acak ditukar dengan gen lain dalam kromosom.
"""

#proses mutasi
def mutasi(kromosom, pM):
    for i in range(len(kromosom)):
        if np.random.rand() <= pM:
            j = np.random.randint(len(kromosom))
            kromosom[i], kromosom[j] = kromosom[j], kromosom[i]
    return kromosom

"""# pergantian generasi
menjalankan algoritma genetika dengan metode steady state.
populasi diperbarui melalui seleksi orang tua, crossover, dan mutasi.
hasil akhir adalah populasi dengan kromosom terbaik.
"""

#algoritma genetika steady state
def steadyState(jumlahGeneration, populasi, jumlahPopulasi, pC, pM, node):
    for _ in range(jumlahGeneration):
        fitnesses = [nilaiFitness(kromosom, node) for kromosom in populasi]
        new_population = []
        while len(new_population) < jumlahPopulasi:
            parent1 = rouletteWheelSelection(populasi, fitnesses)
            parent2 = rouletteWheelSelection(populasi, fitnesses)
            child1, child2 = crossover(parent1, parent2, pC)
            child1 = mutasi(child1, pM)
            child2 = mutasi(child2, pM)
            new_population.extend([child1, child2])
        populasi = sorted(new_population, key=lambda x: -nilaiFitness(x, node))[:jumlahPopulasi]
    return populasi

"""# inisialisasi variabel
variabel node berisi koordinat (x, y) dari setiap node yang harus dikunjungi.
"""

node = [[0, 0.65, 2.35, 2.1, 1.1, 1.6, 1.6, 2.4],
        [0.95, 0, 1.55, 2.3, 1.9, 2.2, 0.95, 3],
        [1.75, 1.55, 0, 2.95, 2.75, 2.85, 1.55, 3.65],
        [2.1, 2.3, 2.8, 0, 2.1, 2.3, 1.6, 2.9],
        [1, 0.95, 1.55, 1.6, 0, 1.5, 1.7, 2.9],
        [1.2, 1.4, 2.85, 2.3, 0.7, 8, 1.5, 2.8],
        [1.1, 0.95, 1.55, 1.6, 1.3, 1.5, 0, 2.4],
        [2.5, 3, 3.65, 2.9, 2.4, 2.8, 2.3, 0]]

jumlahGenerasi = 100
jumlahIndividu = 20
panjangTournament = round(jumlahIndividu/2)
pC = 0.7
pM = 0.1

"""# main program
hasil terbaik (rute terpendek) ditampilkan setelah menjalankan algoritma genetika.
jarak total (cost) dihitung berdasarkan nilai fitness rute terbaik.
"""

#menghasilkan populasi awal
pop = generateKromosom(jumlahIndividu, len(node))

#menjalankan algoritma genetika
pop = steadyState(jumlahGenerasi, pop, jumlahIndividu, pC, pM, node)

#menampilkan hasil
best_route = pop[0]
best_distance = 1 / nilaiFitness(best_route, node)

print("Generasi ke-" + str(jumlahGenerasi))
print("Rute terbaik : " + str(best_route))
print("Cost (jarak) : {:.2f} KM".format(best_distance))