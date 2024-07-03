# The number of neurons in the different data sets data set
WORM_ATLAS_NUM_NEURONS = 279
MEI_ZHEN_NUM_NEURONS = 180

HATCHING_TIME = 800  # min
# According to Mei Zhen after 45 hours a worm is mature
ADULT_WORM_AGE = HATCHING_TIME + 45 * 60  # min

SINGLE_DEVELOPMENTAL_AGE = {0: ADULT_WORM_AGE}
FULL_DEVELOPMENTAL_AGES = {0: HATCHING_TIME, 1: HATCHING_TIME + 5 * 60, 2: HATCHING_TIME + 8 * 60,
                           3: HATCHING_TIME + 16 * 60, 4: HATCHING_TIME + 23 * 60, 5: HATCHING_TIME + 27 * 60,
                           6: ADULT_WORM_AGE}
# Consistent with the data set of birth times
THREE_DEVELOPMENTAL_AGES_CONSISTENT = {0: 1000, 1: HATCHING_TIME + 27 * 60, 2: ADULT_WORM_AGE}
FULL_DEVELOPMENT_AGES_CONSISTENT = {0: HATCHING_TIME, 1: 1000, 2: 1140, 3: 1560, 4: 2260, 5: HATCHING_TIME + 27 * 60,
                                    6: ADULT_WORM_AGE}

# accroding to 'Toward a more accurate 3D atlas of C. elegans neurons'
WORM_LENGTH_NORMALIZATION = 800  # microns

# According to Mei Zhen the worm elongates 5-fold during development (from hatching)
# According to Nicosia et. al starting from 50 micron in the egg.
C_ELEGANS_ELONGATION = WORM_LENGTH_NORMALIZATION // 50

# According to the handbook of the hermaphrodite in the worm atlas (fig 6), in microns.
C_ELEGANS_LENGTHS = {0: 50, HATCHING_TIME: 250, HATCHING_TIME + 12 * 60: 370, HATCHING_TIME + 20 * 60: 500,
                     HATCHING_TIME + 28 * 60: 635, ADULT_WORM_AGE: WORM_LENGTH_NORMALIZATION}
C_ELEGANS_NORMALIZED_LENGTHS = {0: 50 / WORM_LENGTH_NORMALIZATION, HATCHING_TIME: 250 / WORM_LENGTH_NORMALIZATION,
                                HATCHING_TIME + 12 * 60: 370 / WORM_LENGTH_NORMALIZATION,
                                HATCHING_TIME + 20 * 60: 500 / WORM_LENGTH_NORMALIZATION,
                                HATCHING_TIME + 28 * 60: 635 / WORM_LENGTH_NORMALIZATION,
                                ADULT_WORM_AGE: WORM_LENGTH_NORMALIZATION / WORM_LENGTH_NORMALIZATION}
