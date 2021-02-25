#!/usr/bin/env python

__copyright__ = "Copyright 2020, Piotr Obst"

from naive_bayes import Feature, NaiveBayes, Util


class CMC:

    def execute():
        '''
            https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice
            Attribute Information:
            0. Wife's age (numerical)       -> modified to:
                0= <=20
                1=21-25
                2=26-30
                3=31-35
                4=36-40
                5=41-45
                6= 46+
            1. Wife's education (categorical) 1=low, 2, 3, 4=high
            2. Husband's education (categorical) 1=low, 2, 3, 4=high
            3. Number of children ever born (numerical)     -> modified to:
                0=0
                1=1
                ...
                8=8
                9=9/10/11/12/... (9 = "9 or more")
            4. Wife's religion (binary) 0=Non-Islam, 1=Islam
            5. Wife's now working? (binary) 0=Yes, 1=No
            6. Husband's occupation (categorical) 1, 2, 3, 4
            7. Standard-of-living index (categorical) 1=low, 2, 3, 4=high
            8. Media exposure (binary) 0=Good, 1=Not good
            9. Contraceptive method used (class attribute) 1=No-use, 2=Long-term, 3=Short-term     -> modified to 1=No-use, 2=Short-or-long-term-use; '3' changed to '2'
        '''
        dataset = Util.load_file('cmc.data', 9, ',')
        # modify some attributes
        for line in dataset:
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
            else:
                line[0] = '6'

            if int(line[3]) > 9:  # number of children
                line[3] = '9'

            if line[9] == '3':  # class attribute (contraceptive method used)
                line[9] = '2'
        nb = NaiveBayes()
        # let's say positives are subjects using short/long-term contraceptive methods
        # and negatives are subjects not using any protection
        positive = "2"
        negative = "1"
        nb.add_response(negative)  # no contraceptive use
        nb.add_response(positive)  # long- or short-term contraceptive use
        num_of_responses = 2

        wife_age = Feature("wife_age", num_of_responses)
        wife_age.add_category("0")  # <=20
        wife_age.add_category("1")  # 21-25
        wife_age.add_category("2")  # 26-30
        wife_age.add_category("3")  # 31-35
        wife_age.add_category("4")  # 36-40
        wife_age.add_category("5")  # 41-45
        wife_age.add_category("6")  # 46+
        nb.add_feature(wife_age)

        wife_education = Feature("wife_education", num_of_responses)
        wife_education.add_category("1")  # low
        wife_education.add_category("2")
        wife_education.add_category("3")
        wife_education.add_category("4")  # high
        nb.add_feature(wife_education)

        husband_education = Feature("husband_education", num_of_responses)
        husband_education.add_category("1")  # low
        husband_education.add_category("2")
        husband_education.add_category("3")
        husband_education.add_category("4")  # high
        nb.add_feature(husband_education)

        children = Feature("children", num_of_responses)
        children.add_category("0")  # 0
        children.add_category("1")  # 1
        children.add_category("2")  # 2
        children.add_category("3")  # 3
        children.add_category("4")  # 4
        children.add_category("5")  # 5
        children.add_category("6")  # 6
        children.add_category("7")  # 7
        children.add_category("8")  # 8
        children.add_category("9")  # 9+
        nb.add_feature(children)

        wife_religion = Feature("wife_religion", num_of_responses)
        wife_religion.add_category("0")  # non-Islam
        wife_religion.add_category("1")  # Islam
        nb.add_feature(wife_religion)

        wife_working = Feature("wife_working", num_of_responses)
        wife_working.add_category("0")  # yes
        wife_working.add_category("1")  # no
        nb.add_feature(wife_working)

        husband_occupation = Feature("husband_occupation", num_of_responses)
        husband_occupation.add_category("1")
        husband_occupation.add_category("2")
        husband_occupation.add_category("3")
        husband_occupation.add_category("4")
        nb.add_feature(husband_occupation)

        living_standard = Feature("living_standard", num_of_responses)
        living_standard.add_category("1")  # low
        living_standard.add_category("2")
        living_standard.add_category("3")
        living_standard.add_category("4")  # high
        nb.add_feature(living_standard)

        media_exposure = Feature("media_exposure", num_of_responses)
        media_exposure.add_category("0")  # good
        media_exposure.add_category("1")  # not good
        nb.add_feature(media_exposure)

        Util.execute(nb, dataset, positive, negative, "ROC curve - contraceptive use (cmc.data)", "b-", data_points = 1000)
