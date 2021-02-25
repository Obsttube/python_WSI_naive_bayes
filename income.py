#!/usr/bin/env python

__copyright__ = "Copyright 2020, Piotr Obst"

from naive_bayes import Feature, NaiveBayes, Util


class Income:

    def execute():
        '''
            https://archive.ics.uci.edu/ml/datasets/Adult
            Attribute Information:
            0. age: continuous.
                modified to:
                0= <=20
                1=21-25
                2=26-30
                ...
                8=56-60
                9=61-65
                10=66+
            1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
            2. (dropped) fnlwgt: continuous. https://www.kaggle.com/uciml/adult-census-income/discussion/32698
            3. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
            4. (dropped) education-num: continuous.
            5. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
            6. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
            7. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
            8. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
            9. sex: Female, Male.
            10. (dropped) capital-gain: continuous.
            11. (dropped) capital-loss: continuous.
            12. hours-per-week: continuous.
                modified to:
                0= <5
                1=5-14
                2=15-24
                3=25-34
                4=35-44
                5=45-54
                6=55-64
                7=65-74
                8=75-84
                9=85-94
                10=95+
            13. native-country: United-States (all other dropped) dropped: Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
            14. class-attribute: yearly earnings: >50K or <=50K

            (missing values will be dropped)
        '''
        print("loading data")
        dataset = Util.load_file('income.data', 14, ', ')
        # after loading, the class-attribute is the last element
        print("removing duplicates")
        for line in list(dataset):
            if line[13] != "United-States":  # remove entries from countries other than the USA
                dataset.remove(line)
                continue
            for attribute in line:
                if attribute == "?":  # remove lines with missing values
                    dataset.remove(line)
                    break
        print("dropping and categorizing data")
        for line in dataset:
            del line[13]  # drop native-country, cause now all entries are from the USA
            del line[11]  # drop capital-loss
            del line[10]  # drop capital-gain
            del line[4]  # drop education-num
            del line[2]  # drop fnlwgt
            if int(line[0]) < 21:  # modify age
                line[0] = '0'
            elif int(line[0]) < 26:
                line[0] = '1'
            elif int(line[0]) < 31:
                line[0] = '2'
            elif int(line[0]) < 36:
                line[0] = '3'
            elif int(line[0]) < 41:
                line[0] = '4'
            elif int(line[0]) < 46:
                line[0] = '5'
            elif int(line[0]) < 51:
                line[0] = '6'
            elif int(line[0]) < 56:
                line[0] = '7'
            elif int(line[0]) < 61:
                line[0] = '8'
            elif int(line[0]) < 66:
                line[0] = '9'
            else:
                line[0] = '10'

            if int(line[8]) < 5:  # modify hours-per-week
                line[8] = '0'
            elif int(line[8]) < 15:
                line[8] = '1'
            elif int(line[8]) < 25:
                line[8] = '2'
            elif int(line[8]) < 35:
                line[8] = '3'
            elif int(line[8]) < 45:
                line[8] = '4'
            elif int(line[8]) < 55:
                line[8] = '5'
            elif int(line[8]) < 65:
                line[8] = '6'
            elif int(line[8]) < 75:
                line[8] = '7'
            elif int(line[8]) < 85:
                line[8] = '8'
            elif int(line[8]) < 95:
                line[8] = '9'
            else:
                line[8] = '10'
        '''
            after modifications:
            0. age: continuous.
                modified to:
                0= <=20
                1=21-25
                2=26-30
                ...
                8=56-60
                9=61-65
                10=66+
            1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
            2. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
            3. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
            4. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
            5. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
            6. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
            7. sex: Female, Male.
            8. hours-per-week: continuous.
                modified to:
                0= <5
                1=5-14
                2=15-24
                3=25-34
                4=35-44
                5=45-54
                6=55-64
                7=65-74
                8=75-84
                9=85-94
                10=95+
            9. class-attribute: yearly earnings: >50K or <=50K
        '''
        nb = NaiveBayes()
        positive = ">50K"
        negative = "<=50K"
        nb.add_response(negative)  # <=50K
        nb.add_response(positive)  # >50K
        num_of_responses = 2

        age = Feature("age", num_of_responses)
        age.add_category("0")  # <=20
        age.add_category("1")  # 21-25
        age.add_category("2")  # 26-30
        age.add_category("3")  # 31-35
        age.add_category("4")  # 36-40
        age.add_category("5")  # 41-45
        age.add_category("6")  # 46-50
        age.add_category("7")  # 51-55
        age.add_category("8")  # 56-60
        age.add_category("9")  # 61-65
        age.add_category("10")  # 66+
        nb.add_feature(age)

        workclass = Feature("workclass", num_of_responses)
        workclass.add_category("Private")
        workclass.add_category("Self-emp-not-inc")
        workclass.add_category("Self-emp-inc")
        workclass.add_category("Federal-gov")
        workclass.add_category("Local-gov")
        workclass.add_category("State-gov")
        workclass.add_category("Without-pay")
        workclass.add_category("Never-worked")
        nb.add_feature(workclass)

        education = Feature("education", num_of_responses)
        education.add_category("Bachelors")
        education.add_category("Some-college")
        education.add_category("11th")
        education.add_category("HS-grad")
        education.add_category("Prof-school")
        education.add_category("Assoc-acdm")
        education.add_category("Assoc-voc")
        education.add_category("9th")
        education.add_category("7th-8th")
        education.add_category("12th")
        education.add_category("Masters")
        education.add_category("1st-4th")
        education.add_category("10th")
        education.add_category("Doctorate")
        education.add_category("5th-6th")
        education.add_category("Preschool")
        nb.add_feature(education)

        marital_status = Feature("marital_status", num_of_responses)
        marital_status.add_category("Married-civ-spouse")
        marital_status.add_category("Divorced")
        marital_status.add_category("Never-married")
        marital_status.add_category("Separated")
        marital_status.add_category("Widowed")
        marital_status.add_category("Married-spouse-absent")
        marital_status.add_category("Married-AF-spouse")
        nb.add_feature(marital_status)

        occupation = Feature("occupation", num_of_responses)
        occupation.add_category("Tech-support")
        occupation.add_category("Craft-repair")
        occupation.add_category("Other-service")
        occupation.add_category("Sales")
        occupation.add_category("Exec-managerial")
        occupation.add_category("Prof-specialty")
        occupation.add_category("Handlers-cleaners")
        occupation.add_category("Machine-op-inspct")
        occupation.add_category("Adm-clerical")
        occupation.add_category("Farming-fishing")
        occupation.add_category("Transport-moving")
        occupation.add_category("Priv-house-serv")
        occupation.add_category("Protective-serv")
        occupation.add_category("Armed-Forces")
        nb.add_feature(occupation)

        relationship = Feature("relationship", num_of_responses)
        relationship.add_category("Wife")
        relationship.add_category("Own-child")
        relationship.add_category("Husband")
        relationship.add_category("Not-in-family")
        relationship.add_category("Other-relative")
        relationship.add_category("Unmarried")
        nb.add_feature(relationship)

        race = Feature("race", num_of_responses)
        race.add_category("White")
        race.add_category("Asian-Pac-Islander")
        race.add_category("Amer-Indian-Eskimo")
        race.add_category("Other")
        race.add_category("Black")
        nb.add_feature(race)

        sex = Feature("sex", num_of_responses)
        sex.add_category("Female")
        sex.add_category("Male")
        nb.add_feature(sex)

        hours_per_week = Feature("hours_per_week", num_of_responses)
        hours_per_week.add_category("0")  # <5
        hours_per_week.add_category("1")  # 5-14
        hours_per_week.add_category("2")  # 15-24
        hours_per_week.add_category("3")  # 25-34
        hours_per_week.add_category("4")  # 35-44
        hours_per_week.add_category("5")  # 45-54
        hours_per_week.add_category("6")  # 55-64
        hours_per_week.add_category("7")  # 65-74
        hours_per_week.add_category("8")  # 75-84
        hours_per_week.add_category("9")  # 85-94
        hours_per_week.add_category("10")  # 95+
        nb.add_feature(hours_per_week)

        Util.execute(nb, dataset, positive, negative, "ROC curve - yearly income >$50k (income.data)", "b-")
