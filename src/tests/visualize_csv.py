import matplotlib.pyplot as plt
import pandas as pd


# Read the data from the CSV file
data = pd.read_csv('/home/user/catkin_ws/src/collision_aurora/src/closest_distances_to_obstacles.csv')
print(data.head())


# Plot the data
print(data['timestamp'][0])
plt.scatter(data['timestamp']-data['timestamp'][0], data['closest distances to obstacles'])
#aggiungi un titlo al grafico
plt.title('Reprojection error as function of the distances from cameras')
#aggiungi un titlo all asse x e y
plt.xlabel('timestamp')
plt.ylabel('closest_distances_to_obstacles')
plt.show()
