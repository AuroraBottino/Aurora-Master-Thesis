import matplotlib.pyplot as plt
import pandas as pd


# Read the data from the CSV file
data = pd.read_csv('/home/user/catkin_ws/src/collision_aurora/src/reprojection_error.csv')


# Plot the data
plt.scatter(data['distances_from_camera'], data['reprojection_error'])
#aggiungi un titlo al grafico
plt.title('Reprojection error as function of the distances from cameras')
#aggiungi un titlo all asse x e y
plt.xlabel('Distances from cameras')
plt.ylabel('Reprojection error')
plt.show()
