import random as rnd
import csv
import numpy as np

datos = [
    ["Dia", "Clima", "Temperatura", "Humedad", "Viento", "Juego"]
]

with open('datos.csv','w', newline='', encoding='utf-8') as archivo:
  escritor = csv.writer(archivo)
  escritor.writerows(datos)

clima = {"S", "N", "LL"}
temperatura = {"C", "T", "F"}
humedad = {"A", "N"}
viento = {"D", "F"}
juego = {"S", "N"}


for n in range(100):
  uno = rnd.choice(list(clima))
  dos = rnd.choice(list(temperatura))
  tres = rnd.choice(list(humedad))
  cuatro = rnd.choice(list(viento))
  cinco = rnd.choice(list(juego))

  with open('datos.csv','a', newline='', encoding='utf-8') as archivo:
    escritor = csv.writer(archivo)
    escritor.writerow([n + 1, uno, dos, tres, cuatro, cinco])






