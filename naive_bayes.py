#!/usr/bin/env python

__copyright__ = "Copyright 2020, Piotr Obst"

from copy import deepcopy
import os
from random import shuffle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


FILENAME_COUNTER = 1
SAVE_TO_FILE = 1  # 0 = display on the screen, don't save; 1 = save to file, don't display
MAX_NUMBER_LENGTH = 6
VERBOSE = False


def set_verbosity(value: bool):
    global VERBOSE
    VERBOSE = value


class Feature:

    def __init__(self, display_name: str, num_of_responses: str):
        self.display_name = display_name
        self.categories = dict()
        self.matrix = list()
        self.num_of_responses = num_of_responses

    def add_category(self, key: str):
        if key in self.categories:
            return
        self.categories[key] = len(self.categories)
        line = list()
        for _ in range(self.num_of_responses):
            line.append(0)
        self.matrix.append(line)

    def get_category_id(self, key: str) -> int:
        if key not in self.categories:
            print(self.display_name, ' ', key)
            raise KeyError(f'Unknown key \'{key}\' in feature: {self.display_name}')
        return self.categories[key]

    def add_entry(self, category: str, response_id: int):
        self.matrix[self.get_category_id(category)][response_id] += 1

    def debug_print(self, responses: Dict[str, int] = None):
        print('Feature: ', self.display_name)
        print(' ' * 20, end = '')
        for i in range(self.num_of_responses):
            if responses is None:
                print(str(i).ljust(MAX_NUMBER_LENGTH), end = '')
            else:
                print(list(responses.keys())[i].ljust(MAX_NUMBER_LENGTH), end = '')
        print()
        for key in self.categories:
            id = self.categories[key]
            print(key.ljust(20, '.'), end = '')
            for i in range(self.num_of_responses):
                print(str(self.matrix[id][i]).ljust(MAX_NUMBER_LENGTH), end = '')
            print()

    def get_category_probability(self, category: str, response_number: int, response_entries: int) -> float:
        return self.matrix[self.get_category_id(category)][response_number] / response_entries


class NaiveBayes:

    def __init__(self):
        self.features = list()
        self.responses = dict()
        self.total_entries_per_response = list()
        self.num_of_entries = 0

    def add_feature(self, feature: Feature):
        self.features.append(feature)

    def add_response(self, key: str):
        if key in self.responses:
            return
        self.responses[key] = len(self.responses)
        self.total_entries_per_response.append(0)

    def load_training_dataset(self, dataset: List[List[str]]):
        for line in dataset:
            self.load_line(line)

    def get_response_id(self, key: str) -> int:
        if key not in self.responses:
            pass  # TODO: raise exception
        return self.responses[key]

    def load_line(self, line: List[str]):
        response_id = self.get_response_id(line[-1])
        for i in range(len(self.features)):
            self.features[i].add_entry(line[i], response_id)
        self.total_entries_per_response[response_id] += 1
        self.num_of_entries += 1

    def get_response_probability(self, response: str, categories: List[str]):  # not normalized
        probability = 1
        response_id = self.get_response_id(response)
        for i in range(len(self.features)):
            probability *= self.features[i].get_category_probability(categories[i], response_id, self.total_entries_per_response[response_id])
        probability *= (self.total_entries_per_response[response_id] / self.num_of_entries)
        return probability

    def get_probabilities_for_responses(self, categories: List[str]):  # normalized
        probabilities = list()
        sum_of_probabilities = 0
        for response in self.responses:
            probability = self.get_response_probability(response, categories)
            probabilities.append(probability)
            sum_of_probabilities += probability
        if sum_of_probabilities != 0:
            for i in range(len(probabilities)):  # normalize
                probabilities[i] /= sum_of_probabilities
        return probabilities

    def debug_print(self):
        for feature in self.features:
            feature.debug_print(self.responses)
            print('Total'.ljust(20, '.'), end = '')
            for i in range(len(self.responses)):
                print(str(self.total_entries_per_response[i]).ljust(MAX_NUMBER_LENGTH), end = '')
            print()
            print()


class Util:

    @staticmethod
    def divide_dataset(dataset: List[List[str]], num_of_divisions: int, shift: int) -> Tuple[List[List[str]], List[List[str]]]:  # Tuple(training_dataset, test_dataset)
        training = list()
        test = list()
        min_i = num_of_divisions * shift
        max_i = len(dataset) / num_of_divisions + min_i
        for i in range(len(dataset)):
            if i >= min_i and i < max_i:
                test.append(dataset[i])
            else:
                training.append(dataset[i])
        return (training, test)

    @staticmethod
    def load_file(filename: str, class_attribute_index: int, delimiter: str) -> List[List[str]]:
        dataset = list()
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                dataset_line = list()
                stripped = line.strip()  # remove newline char
                attributes = stripped.split(delimiter)
                if len(attributes) < 2:  # remove lines without delimiters
                    continue
                for i in range(len(attributes)):
                    if i != class_attribute_index:
                        dataset_line.append(attributes[i])
                dataset_line.append(attributes[class_attribute_index])  # class attribute as the last element
                dataset.append(dataset_line)
        shuffle(dataset)
        return dataset

    @staticmethod
    def show_plot(plt):
        global FILENAME_COUNTER
        plt.plot([0, 1], [0, 1], f"r:", label = f"random classifier")
        plt.xlabel("False-positives rate")
        plt.ylabel("True-positives rate")
        axes = plt.gca()
        axes.set_ylim([-0.01, 1.01])
        axes.set_xlim([-0.01, 1.01])
        plt.legend()
        plt.grid()
        if SAVE_TO_FILE:
        	folder_name = "graphs"
        	if not os.path.exists(folder_name):
        		os.makedirs(folder_name)
        	plt.savefig(f'{folder_name}/{FILENAME_COUNTER}.png')
        else:
        	plt.show()
        plt.clf()
        FILENAME_COUNTER += 1

    @staticmethod
    def execute(nb: NaiveBayes, dataset: List[List[str]], positive: str, negative: str, plt_title: str, plt_line_type: str, data_points: int = 100):
        nb1 = deepcopy(nb)
        nb2 = deepcopy(nb)
        nb3 = deepcopy(nb)
        # for cross-validation
        training1, test1 = Util.divide_dataset(dataset, num_of_divisions = 3, shift = 0)
        training2, test2 = Util.divide_dataset(dataset, num_of_divisions = 3, shift = 1)
        training3, test3 = Util.divide_dataset(dataset, num_of_divisions = 3, shift = 2)
        nb1.load_training_dataset(training1)
        nb2.load_training_dataset(training2)
        nb3.load_training_dataset(training3)

        if VERBOSE is True:
            print("nb1 data example:")
            nb1.debug_print()

        x_data = list()
        y_data = list()

        print("calculating ROC curve")
        tests = [test1, test2, test3]
        nbs = [nb1, nb2, nb3]
        for i in range(data_points):
            if (i + 1) % (data_points / 10) == 0:
                print(f'{int(i / data_points * 100) + 1}%')
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            cut_off_point = i / data_points
            for j in range(len(tests)):
                for line in tests[j]:
                    correct_response = line[-1]
                    probabilities = nbs[j].get_probabilities_for_responses(line[:-1])
                    result = negative
                    if probabilities[1] >= cut_off_point:
                        result = positive
                    if result == negative and correct_response == negative:
                        true_negatives += 1
                    elif result == negative and correct_response == positive:
                        false_negatives += 1
                    elif result == positive and correct_response == negative:
                        false_positives += 1
                    elif result == positive and correct_response == positive:
                        true_positives += 1
            true_positives_rate = true_positives / ( true_positives + false_negatives )
            true_negatives_rate = true_negatives / ( true_negatives + false_positives )
            false_positives_rate = 1 - true_negatives_rate
            x_data.append(false_positives_rate)
            y_data.append(true_positives_rate)

        # process duplicated x values
        xy_data_sum = dict()
        xy_data_num = dict()
        for i in range(len(x_data)):
            if x_data[i] in xy_data_sum:
                xy_data_sum[x_data[i]] += y_data[i]
                xy_data_num[x_data[i]] += 1
            else:
                xy_data_sum[x_data[i]] = y_data[i]
                xy_data_num[x_data[i]] = 1
        for key in xy_data_sum.keys():
            xy_data_sum[key] /= xy_data_num[key]  # average y values for duplicated x values
        plt.title(plt_title)
        plt.plot(xy_data_sum.keys(), xy_data_sum.values(), plt_line_type, label = "naive binary Bayes classifier")
        Util.show_plot(plt)
