import os
import csv


def cache_data(func):
    def wrapper(a, b):
        storage_folder = 'storage'
        if not os.path.exists(storage_folder):
            os.makedirs(storage_folder)

        data_file = os.path.join(storage_folder, f'{func.__name__}_{a}_{b}.csv')
        if os.path.exists(data_file):
            with open(data_file, 'r', newline='') as file:
                reader = csv.reader(file)
                data = next(reader)[0]
        else:
            data = func(a, b)
            with open(data_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([data])

        return data

    return wrapper
