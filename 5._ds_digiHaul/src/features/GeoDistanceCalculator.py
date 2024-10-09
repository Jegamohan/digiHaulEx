##############################################################
#                                                            #
# GeoDistanceCalculator
#                                                            #
##############################################################

import pandas as pd
import numpy as np


import pandas as pd
import numpy as np


class GeoDistanceCalculator:
    def __init__(self, lat1, lon1, lat2, lon2):
        self.lat1 = lat1
        self.lon1 = lon1
        self.lat2 = lat2
        self.lon2 = lon2

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        r = 6371000  # Radius of Earth in meters
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = (
            np.sin(delta_phi / 2) ** 2
            + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return r * c  # Distance in meters

    def calculate_distance(self):
        return self.haversine(self.lat1, self.lon1, self.lat2, self.lon2)

    @classmethod
    def from_row(cls, row):
        """Create an instance of the class from a DataFrame row."""
        return cls(
            row["FIRST_COLLECTION_LATITUDE"],
            row["FIRST_COLLECTION_LONGITUDE"],
            row["LAST_DELIVERY_LATITUDE"],
            row["LAST_DELIVERY_LONGITUDE"],
        )

    def calculate_delivery_distance(self):
        """Calculate delivery distance using the instance variables."""
        return self.calculate_distance()
