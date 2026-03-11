"""
Real-time station data downloader for Metrodorf
Fetches station information from multiple sources:
1. v6.db.transport.rest API (primary, real-time)
2. IRIS API (backup, XML format)
3. Local cache (fallback)

Includes manual corrections for known API errors:
- Bingen → Bochum (API confusion)
"""

import requests
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Manual corrections for known API mistakes
STATION_CORRECTIONS = {
    'Bingen(Rhein) Hbf': 'Bochum Hbf',  # API returns Bingen for Bochum
    'Bingen Hbf': 'Bochum Hbf',          # Alternative name
}

class StationDownloader:
    """
    Downloads station data from multiple real-time sources
    Always tries live APIs first, falls back to cached data
    Includes manual corrections for API errors
    """
    
    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Metrodorf/1.0 (Research Project)"})
        
        # Rhine-Ruhr stations we care about
        self.target_stations = [
            "Dortmund Hbf", "Essen Hbf", "Duisburg Hbf", "Düsseldorf Hbf",
            "Cologne Hbf", "Bonn Hbf", "Bochum Hbf", "Wuppertal Hbf"
        ]
        
        # Local EVA mapping for fallback
        self.eva_map = {
            'Dortmund Hbf': '8000080',
            'Essen Hbf': '8000098',
            'Duisburg Hbf': '8000086',
            'Düsseldorf Hbf': '8000085',
            'Cologne Hbf': '8000207',
            'Bonn Hbf': '8000044',
            'Bochum Hbf': '8000039',
            'Wuppertal Hbf': '8000266'
        }
        
        # API endpoints
        self.apis = [
            {
                'name': 'v6',
                'url': "https://v6.db.transport.rest/stops",
                'type': 'json',
                'priority': 1
            },
            {
                'name': 'iris',
                'url': "https://iris.noncd.db.de/iris-tts/timetable/station",
                'type': 'xml',
                'priority': 2
            }
        ]
        
        self.stations = []
        
    def _get_from_v6(self, query):
        """Fetch station from v6.db.transport.rest API"""
        try:
            url = f"{self.apis[0]['url']}?query={query}"
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    station = data[0]
                    return {
                        'name': station.get('name'),
                        'id': station.get('id'),
                        'latitude': station.get('location', {}).get('latitude'),
                        'longitude': station.get('location', {}).get('longitude'),
                        'source': 'v6',
                        'timestamp': datetime.now().isoformat()
                    }
            elif response.status_code == 429:
                logger.warning("v6 API rate limited, waiting...")
                time.sleep(60)
            return None
        except Exception as e:
            logger.debug(f"v6 API failed for {query}: {e}")
            return None
    
    def _get_from_iris(self, eva_id):
        """Fetch station from IRIS API (XML format)"""
        try:
            url = f"{self.apis[1]['url']}/{eva_id}"
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                root = ET.fromstring(response.text)
                for station in root.findall('.//station'):
                    return {
                        'name': station.get('name'),
                        'eva': station.get('eva'),
                        'ds100': station.get('ds100'),
                        'source': 'iris',
                        'timestamp': datetime.now().isoformat()
                    }
            return None
        except Exception as e:
            logger.debug(f"IRIS API failed for {eva_id}: {e}")
            return None
    
    def _correct_station_name(self, station, original_query):
        """
        Apply manual corrections if needed
        Returns corrected station dict
        """
        if not station:
            return station
        
        station_name = station.get('name')
        
        # Check if this is a known API mistake
        if station_name in STATION_CORRECTIONS:
            corrected_name = STATION_CORRECTIONS[station_name]
            logger.warning(f"⚠️ API returned '{station_name}' for '{original_query}', correcting to '{corrected_name}'")
            
            # Create corrected station entry
            corrected_station = station.copy()
            corrected_station['name'] = corrected_name
            corrected_station['original_name'] = station_name
            corrected_station['corrected'] = True
            corrected_station['correction_reason'] = f"API returned {station_name} for {original_query}"
            return corrected_station
        
        # Also check if name completely wrong (like Bingen for Bochum)
        if original_query == "Bochum Hbf" and "Bingen" in station_name:
            corrected_name = "Bochum Hbf"
            logger.warning(f"⚠️ API returned '{station_name}' for '{original_query}', forcing to '{corrected_name}'")
            
            corrected_station = station.copy()
            corrected_station['name'] = corrected_name
            corrected_station['original_name'] = station_name
            corrected_station['corrected'] = True
            corrected_station['correction_reason'] = f"API returned {station_name} for {original_query}"
            return corrected_station
        
        return station
    
    def download_stations(self, force_refresh=False):
        """
        Download all target stations using available APIs
        Applies manual corrections for known errors
        Saves to CSV for future use
        """
        cache_file = self.data_dir / "stations_live.csv"
        df = None  # Initialize df

        # Use cache if exists and not forcing refresh
        if cache_file.exists() and not force_refresh:
            logger.info(f"📁 Loading stations from cache: {cache_file}")
            df = pd.read_csv(cache_file)
            # Convert to proper type
            raw_stations = df.to_dict('records')
            self.stations = [{str(k): v for k, v in s.items()} for s in raw_stations]
            return df

        logger.info("📡 Fetching stations from live APIs...")
        self.stations = []  # Reset stations list

        for station_name in self.target_stations:
            logger.info(f"🔍 Searching: {station_name}")

            # Try v6 API first
            station = self._get_from_v6(station_name)

            # If v6 fails, try IRIS with EVA ID
            if not station:
                eva_id = self.eva_map.get(station_name)
                if eva_id:
                    station = self._get_from_iris(eva_id)

            # Apply manual corrections if needed
            if station:
                station = self._correct_station_name(station, station_name)
                # Convert to proper type before appending
                station_typed = {str(k): v for k, v in station.items()}
                self.stations.append(station_typed)
                if station.get('corrected'):
                    logger.info(f"   ✅ Found (corrected): {station.get('name')} via {station.get('source')} (was: {station.get('original_name')})")
                else:
                    logger.info(f"   ✅ Found: {station.get('name')} via {station.get('source')}")
            else:
                logger.warning(f"   ❌ Could not find: {station_name}")
                # Add placeholder with basic info
                self.stations.append({
                    'name': station_name,
                    'eva': self.eva_map.get(station_name),
                    'source': 'local',
                    'timestamp': datetime.now().isoformat(),
                    'placeholder': True
                })

            # Be nice to APIs
            time.sleep(2)

        # Save to CSV
        df = pd.DataFrame(self.stations)
        df.to_csv(cache_file, index=False)
        logger.info(f"✅ Saved {len(self.stations)} stations to {cache_file}")

        return df
    
    def get_station_by_name(self, name):
        """Get station info by name (from memory or cache)"""
        if not self.stations:
            self.download_stations()
        
        for station in self.stations:
            if station.get('name') == name:
                return station
        return None


# Quick test
if __name__ == "__main__":
    downloader = StationDownloader()
    
    print("\n" + "="*60)
    print("🚆 STATION DOWNLOADER TEST")
    print("="*60)
    
    # Download all stations
    df = downloader.download_stations(force_refresh=True)
    print(f"\n✅ Downloaded {len(df)} stations")
    print(df[['name', 'source', 'corrected']].head(10))
    
    # Test lookup
    print("\n🔍 Testing lookup:")
    bochum = downloader.get_station_by_name('Bochum Hbf')
    print(f"   Bochum: {bochum}")
    
    # Show corrections
    corrected = df[df['corrected'] == True]
    if len(corrected) > 0:
        print("\n🔄 Stations that were corrected:")
        for _, row in corrected.iterrows():
            print(f"   {row['original_name']} → {row['name']} ({row['correction_reason']})")