#!/usr/bin/env python

__copyright__ = "Copyright 2020, Piotr Obst"

from naive_bayes import Feature, NaiveBayes, Util


class Mushroom:

    def execute():
        '''
            https://archive.ics.uci.edu/ml/datasets/Mushroom
            Attribute Information:
            0. edibility (class attribute) e=edible, p=poisonous/unknown edibility/not recommended
            1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
            2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
            3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
            4. bruises?: bruises=t,no=f
            5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
            6. gill-attachment: attached=a,descending=d,free=f,notched=n
            7. gill-spacing: close=c,crowded=w,distant=d
            8. gill-size: broad=b,narrow=n
            9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
            10. stalk-shape: enlarging=e,tapering=t
            11. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
            12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
            13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
            14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
            15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
            16. veil-type: partial=p,universal=u
            17. veil-color: brown=n,orange=o,white=w,yellow=y
            18. ring-number: none=n,one=o,two=t
            19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
            20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
            21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
            22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d
        '''
        dataset = Util.load_file('mushroom.data', 0, ',')

        nb = NaiveBayes()
        # let's say positives are edible mushrooms
        # and negatives are poisonous/unknown edibility/not recommended
        positive = "e"
        negative = "p"
        nb.add_response(negative)  # poisonous/unknown edibility/not recommended
        nb.add_response(positive)  # edible
        num_of_responses = 2

        cap_shape = Feature("cap_shape", num_of_responses)
        cap_shape.add_category("b")  # bell
        cap_shape.add_category("c")  # conical
        cap_shape.add_category("x")  # convex
        cap_shape.add_category("f")  # flat
        cap_shape.add_category("k")  # knobbed
        cap_shape.add_category("s")  # sunken
        nb.add_feature(cap_shape)

        cap_surface = Feature("cap_surface", num_of_responses)
        cap_surface.add_category("f")  # fibrous
        cap_surface.add_category("g")  # grooves
        cap_surface.add_category("y")  # scaly
        cap_surface.add_category("s")  # smooth
        nb.add_feature(cap_surface)

        cap_color = Feature("cap_color", num_of_responses)
        cap_color.add_category("n")  # brown
        cap_color.add_category("b")  # buff
        cap_color.add_category("c")  # cinnamon
        cap_color.add_category("g")  # gray
        cap_color.add_category("r")  # green
        cap_color.add_category("p")  # pink
        cap_color.add_category("u")  # purple
        cap_color.add_category("e")  # red
        cap_color.add_category("w")  # white
        cap_color.add_category("y")  # yellow
        nb.add_feature(cap_color)

        bruises = Feature("bruises", num_of_responses)
        bruises.add_category("t")  # bruises
        bruises.add_category("f")  # no bruises
        nb.add_feature(bruises)

        odor = Feature("odor", num_of_responses)
        odor.add_category("a")  # almond
        odor.add_category("l")  # anise
        odor.add_category("c")  # creosote
        odor.add_category("y")  # fishy
        odor.add_category("f")  # foul
        odor.add_category("m")  # musty
        odor.add_category("n")  # none
        odor.add_category("p")  # pungent
        odor.add_category("s")  # spicy
        nb.add_feature(odor)

        gill_attachment = Feature("gill_attachment", num_of_responses)
        gill_attachment.add_category("a")  # attached
        gill_attachment.add_category("d")  # descending
        gill_attachment.add_category("f")  # free
        gill_attachment.add_category("n")  # notched
        nb.add_feature(gill_attachment)

        gill_spacing = Feature("gill_spacing", num_of_responses)
        gill_spacing.add_category("c")  # close
        gill_spacing.add_category("w")  # crowded
        gill_spacing.add_category("d")  # distant
        nb.add_feature(gill_spacing)

        gill_size = Feature("gill_size", num_of_responses)
        gill_size.add_category("b")  # broad
        gill_size.add_category("n")  # narrow
        nb.add_feature(gill_size)

        gill_color = Feature("gill_color", num_of_responses)
        gill_color.add_category("k")  # black
        gill_color.add_category("n")  # brown
        gill_color.add_category("b")  # buff
        gill_color.add_category("h")  # chocolate
        gill_color.add_category("g")  # gray
        gill_color.add_category("r")  # green
        gill_color.add_category("o")  # orange
        gill_color.add_category("p")  # pink
        gill_color.add_category("u")  # purple
        gill_color.add_category("e")  # red
        gill_color.add_category("w")  # white
        gill_color.add_category("y")  # yellow
        nb.add_feature(gill_color)

        stalk_shape = Feature("stalk_shape", num_of_responses)
        stalk_shape.add_category("e")  # enlarging
        stalk_shape.add_category("t")  # tapering
        nb.add_feature(stalk_shape)

        stalk_root = Feature("stalk_root", num_of_responses)
        stalk_root.add_category("b")  # bulbous
        stalk_root.add_category("c")  # club
        stalk_root.add_category("u")  # cup
        stalk_root.add_category("e")  # equal
        stalk_root.add_category("z")  # rhizomorphs
        stalk_root.add_category("r")  # rooted
        stalk_root.add_category("?")  # missing
        nb.add_feature(stalk_root)

        stalk_surface_above_ring = Feature("stalk_surface_above_ring", num_of_responses)
        stalk_surface_above_ring.add_category("f")  # fibrous
        stalk_surface_above_ring.add_category("y")  # scaly
        stalk_surface_above_ring.add_category("k")  # silky
        stalk_surface_above_ring.add_category("s")  # smooth
        nb.add_feature(stalk_surface_above_ring)

        stalk_surface_below_ring = Feature("stalk_surface_below_ring", num_of_responses)
        stalk_surface_below_ring.add_category("f")  # fibrous
        stalk_surface_below_ring.add_category("y")  # scaly
        stalk_surface_below_ring.add_category("k")  # silky
        stalk_surface_below_ring.add_category("s")  # smooth
        nb.add_feature(stalk_surface_below_ring)

        stalk_color_above_ring = Feature("stalk_color_above_ring", num_of_responses)
        stalk_color_above_ring.add_category("n")  # brown
        stalk_color_above_ring.add_category("b")  # buff
        stalk_color_above_ring.add_category("c")  # cinnamon
        stalk_color_above_ring.add_category("g")  # gray
        stalk_color_above_ring.add_category("o")  # orange
        stalk_color_above_ring.add_category("p")  # pink
        stalk_color_above_ring.add_category("e")  # red
        stalk_color_above_ring.add_category("w")  # white
        stalk_color_above_ring.add_category("y")  # yellow
        nb.add_feature(stalk_color_above_ring)

        stalk_color_below_ring = Feature("stalk_color_below_ring", num_of_responses)
        stalk_color_below_ring.add_category("n")  # brown
        stalk_color_below_ring.add_category("b")  # buff
        stalk_color_below_ring.add_category("c")  # cinnamon
        stalk_color_below_ring.add_category("g")  # gray
        stalk_color_below_ring.add_category("o")  # orange
        stalk_color_below_ring.add_category("p")  # pink
        stalk_color_below_ring.add_category("e")  # red
        stalk_color_below_ring.add_category("w")  # white
        stalk_color_below_ring.add_category("y")  # yellow
        nb.add_feature(stalk_color_below_ring)

        veil_type = Feature("veil_type", num_of_responses)
        veil_type.add_category("p")  # partial
        veil_type.add_category("u")  # universal
        nb.add_feature(veil_type)

        veil_color = Feature("veil_color", num_of_responses)
        veil_color.add_category("n")  # brown
        veil_color.add_category("o")  # orange
        veil_color.add_category("w")  # white
        veil_color.add_category("y")  # yellow
        nb.add_feature(veil_color)

        ring_number = Feature("ring_number", num_of_responses)
        ring_number.add_category("n")  # none
        ring_number.add_category("o")  # one
        ring_number.add_category("t")  # two
        nb.add_feature(ring_number)

        ring_type = Feature("ring_type", num_of_responses)
        ring_type.add_category("c")  # cobwebby
        ring_type.add_category("e")  # evanescent
        ring_type.add_category("f")  # flaring
        ring_type.add_category("l")  # large
        ring_type.add_category("n")  # none
        ring_type.add_category("p")  # pendant
        ring_type.add_category("s")  # sheathing
        ring_type.add_category("z")  # zone
        nb.add_feature(ring_type)

        spore_print_color = Feature("spore_print_color", num_of_responses)
        spore_print_color.add_category("k")  # black
        spore_print_color.add_category("n")  # brown
        spore_print_color.add_category("b")  # buff
        spore_print_color.add_category("h")  # chocolate
        spore_print_color.add_category("r")  # green
        spore_print_color.add_category("o")  # orange
        spore_print_color.add_category("u")  # purple
        spore_print_color.add_category("w")  # white
        spore_print_color.add_category("y")  # yellow
        nb.add_feature(spore_print_color)

        population = Feature("population", num_of_responses)
        population.add_category("a")  # abundant
        population.add_category("c")  # clustered
        population.add_category("n")  # numerous
        population.add_category("s")  # scattered
        population.add_category("v")  # several
        population.add_category("y")  # solitary
        nb.add_feature(population)

        habitat = Feature("habitat", num_of_responses)
        habitat.add_category("g")  # grasses
        habitat.add_category("l")  # leaves
        habitat.add_category("m")  # meadows
        habitat.add_category("p")  # paths
        habitat.add_category("u")  # urban
        habitat.add_category("w")  # waste
        habitat.add_category("d")  # woods
        nb.add_feature(habitat)

        Util.execute(nb, dataset, positive, negative, "ROC curve - mushroom edibility (mushroom.data)", "b:o")
