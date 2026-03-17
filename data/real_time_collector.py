"""
Real-time data collector for Deutsche Bahn delays
Uses multiple API endpoints with conservative rate limiting
- IRIS: 1 request/10 seconds (most reliable)
- v6: 1 request/10 seconds (unstable, falls back quickly)
- VBB: 1 request/10 seconds (least reliable)
Always falls back to synthetic data when APIs unavailable
"""

import requests
import pandas as pd
import logging
from datetime import datetime
import time
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

class RealTimeCollector:
    """
    Collects real-time train data from multiple German public transport APIs
    Uses conservative rate limiting (1 request per 10 seconds per API)
    Falls back gracefully when APIs are unavailable
    """
    
    def __init__(self, data_dir="data/raw"):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Metrodorf/1.0 (Research Project)"})
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Station mappings with multiple ID formats
        self.stations = {
            'Cologne Hbf': {'eva': '8000207', 'ds100': 'KK', 'lat': 50.9422, 'lon': 6.9581},
            'Düsseldorf Hbf': {'eva': '8000085', 'ds100': 'KD', 'lat': 51.2277, 'lon': 6.7735},
            'Duisburg Hbf': {'eva': '8000086', 'ds100': 'EDG', 'lat': 51.4344, 'lon': 6.7623},
            'Essen Hbf': {'eva': '8000098', 'ds100': 'EE', 'lat': 51.4556, 'lon': 7.0116},
            'Dortmund Hbf': {'eva': '8000080', 'ds100': 'EDO', 'lat': 51.5136, 'lon': 7.4653},
            'Bochum Hbf': {'eva': '8000039', 'ds100': 'EBO', 'lat': 51.4818, 'lon': 7.2162},
            'Wuppertal Hbf': {'eva': '8000266', 'ds100': 'KW', 'lat': 51.2562, 'lon': 7.1507},
            'Bonn Hbf': {'eva': '8000044', 'ds100': 'KB', 'lat': 50.7359, 'lon': 7.0999}
        }
        
        # Rate limiting: 1 request per 10 seconds per API (6 requests/minute max total)
        self.last_request_time = {
            'iris': 0,
            'v6': 0,
            'vbb': 0
        }
        self.min_request_interval = 10  # seconds
        
        # API failure tracking
        self.api_failures = {
            'iris': 0,
            'v6': 0,
            'vbb': 0
        }
        self.max_failures = 3  # Disable API after 3 consecutive failures
        # Initialize database connection
        self.db = DatabaseManager()
        self.db.create_tables()  # Ensure tables exist
        # Track overall API availability
        self.api_available = self._check_any_api()
        if self.api_available:
            logger.info("✅ At least one real-time API available")
        else:
            logger.warning("⚠️ No real-time APIs available - using synthetic fallback")
    
    def _wait_for_rate_limit(self, api_name):
        """Enforce 10-second minimum interval between requests to same API"""
        now = time.time()
        time_since_last = now - self.last_request_time.get(api_name, 0)
        if time_since_last < self.min_request_interval:
           sleep_time = self.min_request_interval - time_since_last
           logger.debug(f"⏳ Rate limit for {api_name}: waiting {sleep_time:.1f}s")
           time.sleep(sleep_time)
           self.last_request_time[api_name] = time.time()  # type: ignore
              
    def _check_any_api(self):
        """Check if at least one API is reachable"""
        # Try IRIS first (most reliable)
        try:
            self._wait_for_rate_limit('iris')
            response = self.session.get(
                "https://iris.noncd.db.de/iris-tts/timetable/station/8000080",
                timeout=3
            )
            if response.status_code == 200:
                return True
        except:
            pass
        
        # Try v6 as backup
        try:
            self._wait_for_rate_limit('v6')
            response = self.session.get(
                "https://v6.db.transport.rest/stops/8000080",
                timeout=3
            )
            if response.status_code == 200:
                return True
        except:
            pass
        
        return False
    
    def _get_from_iris(self, station_id):
        """
        Get station info from IRIS API (most reliable)
        Returns: station dict or None
        Rate: 1 request per 10 seconds
        """
        if self.api_failures['iris'] >= self.max_failures:
            logger.debug("IRIS API disabled (too many failures)")
            return None
        
        self._wait_for_rate_limit('iris')
        
        try:
            url = f"https://iris.noncd.db.de/iris-tts/timetable/station/{station_id}"
            response = self.session.get(url, timeout=3)
            
            if response.status_code == 200:
                root = ET.fromstring(response.text)
                for station in root.findall('.//station'):
                    self.api_failures['iris'] = 0  # Reset on success
                    return {
                        'name': station.get('name'),
                        'eva': station.get('eva'),
                        'ds100': station.get('ds100'),
                        'source': 'iris'
                    }
            elif response.status_code == 429:
                logger.warning("IRIS API rate limited, waiting 60s")
                time.sleep(60)
                return None
            else:
                self.api_failures['iris'] += 1
                logger.warning(f"IRIS API returned {response.status_code}")
                return None
                
        except Exception as e:
            self.api_failures['iris'] += 1
            logger.debug(f"IRIS API failed: {e}")
            return None
    
    def _get_from_v6(self, station_id):
        """
        Get departures from v6.db.transport.rest
        Returns: list of departures or None
        Rate: 1 request per 10 seconds
        """
        if self.api_failures['v6'] >= self.max_failures:
            logger.debug("v6 API disabled (too many failures)")
            return None
        
        self._wait_for_rate_limit('v6')
        
        try:
            url = f"https://v6.db.transport.rest/stops/{station_id}/departures"
            response = self.session.get(
                url, 
                params={"duration": 60, "limit": 10},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                self.api_failures['v6'] = 0
                return data.get('departures', [])
            elif response.status_code == 429:
                logger.warning("v6 API rate limited, waiting 60s")
                time.sleep(60)
                return None
            elif response.status_code == 503:
                logger.debug("v6 API temporarily unavailable")
                self.api_failures['v6'] += 1
                return None
            else:
                self.api_failures['v6'] += 1
                return None
                
        except Exception as e:
            self.api_failures['v6'] += 1
            logger.debug(f"v6 API failed: {e}")
            return None
    
    def _get_from_vbb(self, station_id):
        """
        Get departures from VBB API (least reliable, last resort)
        Returns: list of departures or None
        Rate: 1 request per 10 seconds
        """
        if self.api_failures['vbb'] >= self.max_failures:
            logger.debug("VBB API disabled (too many failures)")
            return None
        
        self._wait_for_rate_limit('vbb')
        
        try:
            url = f"https://v5.vbb.transport.rest/stops/{station_id}/departures"
            response = self.session.get(
                url,
                params={"duration": 60},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                self.api_failures['vbb'] = 0
                return data if isinstance(data, list) else data.get('departures', [])
            else:
                self.api_failures['vbb'] += 1
                return None
                
        except Exception as e:
            self.api_failures['vbb'] += 1
            logger.debug(f"VBB API failed: {e}")
            return None
    
    def get_station_info(self, station_name):
        """
        Get station information using multiple APIs
        Tries IRIS first (most reliable), falls back to others
        """
        station_data = self.stations.get(station_name)
        if not station_data:
            logger.error(f"Unknown station: {station_name}")
            return None
        
        # Try IRIS first
        iris_info = self._get_from_iris(station_data['eva'])
        if iris_info:
            return iris_info
        
        # If IRIS fails, return basic info from local mapping
        return {
            'name': station_name,
            'eva': station_data['eva'],
            'ds100': station_data['ds100'],
            'source': 'local'
        }
    
    def get_departures(self, station_name, limit=10):
        """
        Get departures using multiple APIs
        Tries v6 first, falls back to VBB
        """
        station_data = self.stations.get(station_name)
        if not station_data:
            return []
        
        # Try v6 first
        departures = self._get_from_v6(station_data['eva'])
        if departures:
            return departures
        
        # Try VBB as backup
        departures = self._get_from_vbb(station_data['eva'])
        if departures:
            return departures
        
        return []
    
    def parse_departure(self, departure, station_name):
        """Parse departure data into training format"""
        try:
             # Extract delay in minutes
             delay = 0
             if 'delay' in departure and departure['delay'] is not None:
                 delay = departure['delay'] // 60  # Convert seconds to minutes
             elif 'delayInMinutes' in departure and departure['delayInMinutes'] is not None:
                 delay = departure['delayInMinutes']

             when = departure.get('when', '')
        
             hour = 0
             if when:
                try:
                     hour = datetime.fromisoformat(when.replace('Z', '+00:00')).hour
                except:
                     hour = 12
        
             is_peak = 1 if (7 <= hour <= 9) or (16 <= hour <= 18) else 0
        
             direction = (departure.get('direction') or '').lower()
             is_cologne = 1 if 'köln' in direction or 'cologne' in direction else 0
        
             parsed = {
                 'distance_km': self._estimate_distance(station_name, direction),
                 'time_of_day': hour,
                 'day_of_week': datetime.now().weekday(),
                 'is_peak_hour': is_peak,
                 'is_cologne_bottleneck': is_cologne,
                 'delay_minutes': max(0, delay),
                 'source': 'real',
                 'timestamp': datetime.now().isoformat()
             }
        
             # Save to database
             station_id = self.db.insert_station(
                 station_name,
                 self.stations[station_name]['eva'],
                 self.stations[station_name]['ds100'],
                 self.stations[station_name]['lat'],
                 self.stations[station_name]['lon']
             )
             if station_id:
                # Note: is_peak_hour and is_cologne_bottleneck converted to boolean inside insert_real_delay
                self.db.insert_real_delay(station_id, parsed)
                logger.debug(f"💾 Saved to DB: {parsed['delay_minutes']} min delay")
        
             return parsed
        
        except Exception as e:
           logger.error(f"Error parsing departure: {e}")
           return None
           
    
    def _estimate_distance(self, from_station, to_city):
        """Estimate distance between stations (simplified)"""
        distances = {
            ('Cologne Hbf', 'dortmund'): 70,
            ('Cologne Hbf', 'düsseldorf'): 40,
            ('Cologne Hbf', 'essen'): 65,
            ('Düsseldorf Hbf', 'dortmund'): 55,
            ('Düsseldorf Hbf', 'essen'): 30,
            ('Essen Hbf', 'bochum'): 15,
            ('Dortmund Hbf', 'essen'): 35,
        }
        return distances.get((from_station, to_city.lower()), 50)
    
    def generate_synthetic_sample(self):
        """Generate realistic synthetic sample based on Bologna 2025"""
        np.random.seed()
    
        # Base delay: mostly small, sometimes large (heavy tails)
        base_delay = np.random.exponential(3)
    
        # More realistic probabilities
        # In reality, only ~15% of trains are in peak hours, ~10% pass Cologne
        is_peak = np.random.choice([0, 1], p=[0.85, 0.15])     # 15% peak
        is_cologne = np.random.choice([0, 1], p=[0.9, 0.1])    # 10% Cologne
    
        peak_factor = 1.5 if is_peak else 1.0
        cologne_factor = 2.0 if is_cologne else 1.0
    
        delay = base_delay * peak_factor * cologne_factor
    
        # Add some noise to make it realistic
        noise = np.random.normal(0, 0.5)
        delay = max(0, delay + noise)
    
        return {
          'distance_km': np.random.uniform(10, 100),
          'time_of_day': np.random.randint(0, 24),
          'day_of_week': np.random.randint(0, 7),
          'is_peak_hour': is_peak,
          'is_cologne_bottleneck': is_cologne,
          'delay_minutes': round(delay, 1),
          'source': 'synthetic'
        }
    
        """Generate realistic synthetic sample based on Bologna 2025"""
        np.random.seed()
        
        base_delay = np.random.exponential(3)
        is_peak = np.random.choice([0, 1], p=[0.7, 0.3])
        is_cologne = np.random.choice([0, 1], p=[0.8, 0.2])
        
        peak_factor = 1.5 if is_peak else 1.0
        cologne_factor = 2.0 if is_cologne else 1.0
        
        delay = base_delay * peak_factor * cologne_factor
        
        return {
            'distance_km': np.random.uniform(10, 100),
            'time_of_day': np.random.randint(0, 24),
            'day_of_week': np.random.randint(0, 7),
            'is_peak_hour': is_peak,
            'is_cologne_bottleneck': is_cologne,
            'delay_minutes': round(delay, 1),
            'source': 'synthetic'
        }
    
    def collect_training_data(self, n_samples=1000, real_ratio=0.3):
        """
        Collect mixed dataset: real + synthetic
        Tries multiple APIs with conservative rate limiting
        """
        data = []
        real_samples = 0
        
        if self.api_available:
            target_real = int(n_samples * real_ratio)
            logger.info(f"📡 Attempting to collect up to {target_real} real samples...")
            
            for station_name in self.stations.keys():
                if real_samples >= target_real:
                    break
                
                # Get station info (minimal API calls)
                station_info = self.get_station_info(station_name)
                
                # Get departures (main data source)
                departures = self.get_departures(station_name, limit=5)
                
                if departures:
                    for dep in departures[:3]:  # Take up to 3 per station
                        if real_samples >= target_real:
                            break
                        parsed = self.parse_departure(dep, station_name)
                        if parsed:
                            data.append(parsed)
                            real_samples += 1
                
                # Extra wait between stations (respect all APIs)
                time.sleep(5)
            
            logger.info(f"✅ Collected {real_samples} real samples")
        
        # Fill remaining with synthetic
        synthetic_needed = n_samples - len(data)
        if synthetic_needed > 0:
            logger.info(f"🔄 Generating {synthetic_needed} synthetic samples...")
            for _ in range(synthetic_needed):
                data.append(self.generate_synthetic_sample())
        
        df = pd.DataFrame(data)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.data_dir / f"training_data_{timestamp}.csv"
        df.to_csv(filename, index=False)
        df.to_csv("data/processed/training_data.csv", index=False)
        
        logger.info(f"✅ Saved {len(df)} samples ({real_samples} real, {synthetic_needed} synthetic)")
        return df


# Quick test
if __name__ == "__main__":
    collector = RealTimeCollector()
    
    print("\n" + "="*60)
    print("🚆 REAL-TIME DATA COLLECTOR TEST")
    print("="*60)
    
    print(f"\n📡 API Available: {collector.api_available}")
    
    print("\n📊 Testing station info retrieval...")
    for station in ['Dortmund Hbf', 'Cologne Hbf'][:2]:
        info = collector.get_station_info(station)
        print(f"   {station}: {info}")
        time.sleep(2)
    
    print("\n📊 Testing departure retrieval...")
    for station in ['Dortmund Hbf', 'Essen Hbf'][:1]:
        deps = collector.get_departures(station, limit=3)
        print(f"   {station}: {len(deps)} departures")
        if deps:
            parsed = collector.parse_departure(deps[0], station)
            print(f"   Sample: {parsed}")
        time.sleep(2)
    
    print("\n📊 Collecting training data (10% real, 90% synthetic)...")
    df = collector.collect_training_data(n_samples=20, real_ratio=0.1)
    print(f"\n✅ Collected {len(df)} samples")
    print(df[['source', 'delay_minutes']].head(10))