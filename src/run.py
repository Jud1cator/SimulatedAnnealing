import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser

from utils import *


def main(args):
    city_dataset = pd.read_csv(args.dataset)

    top_n_cities = city_dataset.sort_values(
        by=['population'], ascending=False
    )[:args.top_n][['address', 'geo_lat', 'geo_lon']]

    cities_gdf = get_cities_geodataframe(top_n_cities)

    list_top_n_cities = get_list_of_cities(cities_gdf)

    russia_gdf = get_russia_geodataframe()

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    russia_gdf.plot(ax=ax)
    cities_gdf.plot(ax=ax, marker='o', color='red', markersize=15)

    good_city_order, total_distance, temps, dists = find_approximately_optimal_permutation(
        list_top_n_cities,
        energy_fn=distance_between_cities_in_order,
        neighbour_fn=greedy_hybrid_neighbour,
        initial_temp=args.initial_temperature,
        cooling_rate=args.cooling_rate,
        log_interval=args.log_interval,
        save_convergence_info=args.study_convergence,
        visualize=args.visualize,
        visualization_interval=0.01,
        ax=ax
    )

    if not args.visualize:
        ax.plot(*build_route(good_city_order).xy, color='black')

    ax.set_title(f'Total distance: {total_distance:7.3f}')
    if args.save_plots is not None:
        plt.savefig(os.path.join(args.save_plots, 'route.png'))
    plt.show()

    if args.study_convergence:
        temps = np.array(temps)
        dists = np.array(dists)
        plt.title(f'Initial temperature: {args.initial_temperature} Cooling rate: {args.cooling_rate}')
        plt.plot(temps / temps.max(), label='Normalized temperature')
        plt.plot(dists / dists.max(), label='Normalized distance')
        plt.legend()
        if args.save_plots is not None:
            plt.savefig(os.path.join(args.save_plots, 'convergence.png'))
        plt.show()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-d', '--dataset', required=True, type=str, help='Path to dataset with cities'
    )
    parser.add_argument(
        '-n', '--top-n', type=int, default=30, help='Top N populated cities to choose'
    )
    parser.add_argument(
        '-t', '--initial-temperature', type=int, default=100, help='Initial value of temperature'
    )
    parser.add_argument(
        '-c', '--cooling-rate', type=float, default=0.001, help='Cooling rate'
    )
    parser.add_argument(
        '-l', '--log-interval', type=int, default=100,
        help='How frequently log algorithm progress. Disabled if visualize=True'
    )
    parser.add_argument(
        '-r', '--study-convergence', action='store_true',
        help='Save plot which show how algorithm converged'
    )
    parser.add_argument(
        '-v', '--visualize', action='store_true', help='Add to visualize algorithm work'
    )
    parser.add_argument(
        '-s', '--save-plots', type=str, default=None, help='Where to save plots'
    )

    args = parser.parse_args()
    main(args)
