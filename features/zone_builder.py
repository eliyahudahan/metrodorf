"""
Zone Builder for Rhine-Ruhr Polycentric Region
Based on Levy (2024) - Polycentric Urban Networks
"""

import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZoneBuilder:
    """
    Builds polycentric zones for Rhine-Ruhr region
    Each city is a zone center with Gaussian decay influence
    """
    
    def __init__(self):
        # Rhine-Ruhr zone centers (latitude, longitude)
        self.zones = {
            'Dortmund': {'lat': 51.5136, 'lon': 7.4653, 'population': 588000},
            'Essen': {'lat': 51.4556, 'lon': 7.0116, 'population': 583000},
            'Duisburg': {'lat': 51.4344, 'lon': 6.7623, 'population': 498000},
            'Düsseldorf': {'lat': 51.2277, 'lon': 6.7735, 'population': 620000},
            'Cologne': {'lat': 50.9422, 'lon': 6.9581, 'population': 1086000},
            'Bonn': {'lat': 50.7359, 'lon': 7.0999, 'population': 327000},
            'Bochum': {'lat': 51.4818, 'lon': 7.2162, 'population': 365000},
            'Wuppertal': {'lat': 51.2562, 'lon': 7.1507, 'population': 355000}
        }
        
        # Load station network (from your collector)
        self.stations_df = pd.read_csv("data/raw/rhine_ruhr_network.csv")
        logger.info(f"✅ Loaded {len(self.stations_df)} stations")
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate distance between two points in km
        """
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        return R * c
    
    def gaussian_decay(self, distance, sigma=50):
        """
        Gaussian decay function - influence decreases with distance
        sigma = 50 km (typical zone size in Rhine-Ruhr)
        """
        return np.exp(-(distance**2) / (2 * sigma**2))
    
    def assign_zone_influence(self, station_lat, station_lon):
        """
        For a station, calculate influence from each zone center
        Returns dict of zone_name -> influence_score
        """
        influences = {}
        
        for zone_name, zone_data in self.zones.items():
            # Distance from station to zone center
            dist = self.haversine_distance(
                station_lat, station_lon,
                zone_data['lat'], zone_data['lon']
            )
            
            # Gaussian decay influence
            influence = self.gaussian_decay(dist)
            
            # Weight by population (bigger cities have more influence)
            pop_weight = zone_data['population'] / 1e6  # normalize
            
            influences[zone_name] = influence * pop_weight
        
        return influences
    
    def build_zone_features(self):
        """
        Create feature matrix for all stations
        Each station gets influence scores from all zones
        """
        features = []
        
        for _, station in self.stations_df.iterrows():
            # Get station coordinates (from your network file)
            if 'latitude' in station and float(station['latitude']) != 0:
                lat = float(station['latitude'])
                lon = float(station['longitude'])
            else:
                # Use zone center as approximation
                zone_city = station.get('zone_city', 'Dortmund')
                lat = self.zones[zone_city]['lat']
                lon = self.zones[zone_city]['lon']
            
            # Calculate influences
            influences = self.assign_zone_influence(lat, lon)
            
            # Create feature row
            feature_row = {
                'station_name': station.get('name', 'Unknown'),
                'station_city': station.get('zone_city', 'Unknown'),
                'latitude': lat,
                'longitude': lon,
            }
            # Add influence scores
            feature_row.update(influences)
            
            features.append(feature_row)
        
        df = pd.DataFrame(features)
        
        # Save for ML models
        df.to_csv("data/processed/zone_features.csv", index=False)
        logger.info(f"✅ Created zone features for {len(df)} stations")
        
        return df
    
    def calculate_zone_interaction(self, zone1, zone2):
        """
        Calculate interaction strength between two zones
        Used for modeling delay propagation
        """
        dist = self.haversine_distance(
            self.zones[zone1]['lat'], self.zones[zone1]['lon'],
            self.zones[zone2]['lat'], self.zones[zone2]['lon']
        )
        
        # Interaction strength follows Gaussian decay
        interaction = self.gaussian_decay(dist, sigma=30)
        
        return {
            'zone1': zone1,
            'zone2': zone2,
            'distance_km': dist,
            'interaction_strength': interaction
        }
    
    def build_interaction_matrix(self):
        """
        Create matrix of how zones influence each other
        """
        zone_names = list(self.zones.keys())
        n_zones = len(zone_names)
        
        matrix = np.zeros((n_zones, n_zones))
        
        for i, z1 in enumerate(zone_names):
            for j, z2 in enumerate(zone_names):
                if i != j:
                    interaction = self.calculate_zone_interaction(z1, z2)
                    matrix[i, j] = interaction['interaction_strength']
        
        df = pd.DataFrame(matrix, index=zone_names, columns=zone_names)
        df.to_csv("data/processed/zone_interaction_matrix.csv")
        
        logger.info(f"✅ Built {n_zones}x{n_zones} interaction matrix")
        return df

# Test the zone builder
if __name__ == "__main__":
    zb = ZoneBuilder()
    
    print("\n🔍 Testing Zone Builder")
    print("="*50)
    
    # Test 1: Build zone features
    print("\n📊 Building zone features...")
    features = zb.build_zone_features()
    print(f"   Created features for {len(features)} stations")
    print(f"   Columns: {list(features.columns)}")
    
    # Test 2: Show influences for a sample station
    print("\n📍 Sample: Dortmund Hbf influences")
    dortmund_influence = features[features['station_city'] == 'Dortmund'].iloc[0]
    for zone in zb.zones.keys():
        if zone in dortmund_influence:
            print(f"   {zone}: {dortmund_influence[zone]:.3f}")
    
    # Test 3: Build interaction matrix
    print("\n🔄 Building zone interaction matrix...")
    matrix = zb.build_interaction_matrix()
    print(matrix)