import geopandas as gpd
import math
import matplotlib.pyplot as plt
import random
from shapely.affinity import translate
from shapely.geometry import LineString, Point
from shapely.ops import split
from geopy.distance import distance as geodesic_distance


class City:
    def __init__(self, name: str, geo_lat: float, geo_lon: float):
        self.name = name
        self.geo_lat = geo_lat
        self.geo_lon = geo_lon

    def __str__(self):
        return self.name

    def get_coordinates_lat_lon(self):
        return self.geo_lat, self.geo_lon

    def get_coordinates_lon_lat(self):
        return self.geo_lon, self.geo_lat

    def distance_to(self, city, units: str = 'km'):
        distance = geodesic_distance(self.get_coordinates_lat_lon(), city.get_coordinates_lat_lon())
        if units == 'km':
            return distance.km
        else:
            raise NotImplementedError


def distance_between_cities_in_order(permutation_of_cities):
    distance = 0
    for i in range(1, len(permutation_of_cities)):
        distance += permutation_of_cities[i-1].distance_to(permutation_of_cities[i])
    distance += permutation_of_cities[-1].distance_to(permutation_of_cities[0])
    return distance


def inverse_neighbour(permutation, i, j):
    i, j = (i, j) if i < j else (j, i)
    return [permutation[i + j - t] if i <= t <= j else permutation[t] for t in range(len(permutation))]


def insert_neighbour(permutation, i, j):
    i, j = (i, j) if i < j else (j, i)
    return [permutation[j if t == i else t - 1] if i <= t <= j else permutation[t] for t in range(len(permutation))]


def swap_neighbour(permutation, i, j):
    swap = permutation.copy()
    swap[i] = permutation[j]
    swap[j] = permutation[i]
    return swap


def greedy_hybrid_neighbour(permutation, energy_fn):
    i, j = random.sample(list(range(len(permutation))), 2)
    inverse = inverse_neighbour(permutation, i, j)
    insert = insert_neighbour(permutation, i, j)
    swap = swap_neighbour(permutation, i, j)
    neighbours = (inverse, insert, swap)
    min_idx, min_energy = 0, 1e12
    for i in range(len(neighbours)):
        e = energy_fn(neighbours[i])
        if e < min_energy:
            min_idx = i
            min_energy = e
    return neighbours[min_idx], min_energy


def build_route(permutation):
    return translate(
        LineString(
            [city.get_coordinates_lon_lat() for city in permutation] +
            [permutation[0].get_coordinates_lon_lat()]
        ),
        xoff=-90
    )


def find_approximately_optimal_permutation(
        permutation,
        energy_fn,
        neighbour_fn,
        initial_temp=1000,
        cooling_rate=0.003,
        log_interval=100,
        save_convergence_info=False,
        visualize=False,
        ax=None,
        visualization_interval=0.01
):
    temp = initial_temp
    iterations_since_last_report = log_interval + 1

    if save_convergence_info:
        temperatures = []
        energies = []

    if visualize:
        route = build_route(permutation)
        line = ax.plot(*route.xy, color='black')[0]

    while temp > 1:
        current_energy = energy_fn(permutation)

        if iterations_since_last_report > log_interval and not visualize:
            print(f'Temperature: {temp:4.3f} Distance: {current_energy:7.3f}')
            iterations_since_last_report = 0
        else:
            iterations_since_last_report += 1

        if save_convergence_info:
            temperatures.append(temp)
            energies.append(current_energy)

        new_permutation, new_energy = neighbour_fn(permutation, energy_fn)
        if new_energy < current_energy or \
                random.uniform(0, 1) < math.exp((current_energy - new_energy) / temp):
            permutation = new_permutation
        temp *= 1 - cooling_rate

        if visualize:
            route = build_route(permutation)
            ax.set_title(f'Temperature: {temp:4.3f} Distance: {current_energy:7.3f}')
            line.set_data(*route.xy)
            plt.pause(visualization_interval)

    print(f'Found solution with total distance: {current_energy:7.3f}')

    if save_convergence_info:
        temperatures.append(temp)
        energies.append(current_energy)
        return permutation, current_energy, temperatures, energies
    else:
        return permutation, current_energy, [], []


def shift_geom(shift, gdataframe):
    shift -= 180
    moved_map = []
    splitted_map = []
    border = LineString([(shift,90),(shift,-90)])

    for row in gdataframe["geometry"]:
        splitted_map.append(split(row, border))
    for element in splitted_map:
        items = list(element)
        for item in items:
            minx, miny, maxx, maxy = item.bounds
            if minx >= shift:
                moved_map.append(translate(item, xoff=-180-shift))
            else:
                moved_map.append(translate(item, xoff=180-shift))

    return gpd.GeoDataFrame({"geometry": moved_map})


def get_cities_geodataframe(cities):
    geometry = [translate(Point(xy), xoff=-90) for xy in zip(cities['geo_lon'], cities['geo_lat'])]
    return gpd.GeoDataFrame(cities, geometry=geometry)


def get_list_of_cities(cities):
    list_top_n_cities = []
    for i in range(len(cities)):
        list_top_n_cities.append(
            City(
                name=cities['address'].iloc[i],
                geo_lat=cities['geo_lat'].iloc[i],
                geo_lon=cities['geo_lon'].iloc[i]
            )
        )
    return list_top_n_cities


def get_russia_geodataframe():
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    russia = world[world['name'] == 'Russia']
    russia_shifted = shift_geom(90, russia)
    return russia_shifted
