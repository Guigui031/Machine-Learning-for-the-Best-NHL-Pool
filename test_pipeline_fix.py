"""
Test script to verify data pipeline fixes.
"""

import logging
from config import config
from data_download import download_players_points

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_config():
    """Test configuration loading."""
    print("🔧 Testing configuration...")
    print(f"✅ Config loaded: {config.data.current_season}")
    print(f"✅ API settings: {config.api.max_retries} retries, {config.api.request_delay}s delay")
    return True

def test_data_download():
    """Test data download function."""
    print("📥 Testing data download...")

    # Test downloading just one season
    season = config.data.current_season
    print(f"Attempting to download data for season {season}...")

    success = download_players_points(season, force_refresh=False)

    if success:
        print(f"✅ Successfully downloaded data for season {season}")

        # Check if file exists
        path = config.data.get_players_points_path(season)
        if path.exists():
            print(f"✅ File exists: {path}")
            print(f"✅ File size: {path.stat().st_size} bytes")
        else:
            print(f"❌ File not found: {path}")
            return False
    else:
        print(f"❌ Failed to download data for season {season}")
        return False

    return True

def main():
    """Run all tests."""
    print("🚀 Starting pipeline fix tests...\n")

    try:
        # Test configuration
        if not test_config():
            print("❌ Configuration test failed")
            return

        print()

        # Test data download
        if not test_data_download():
            print("❌ Data download test failed")
            return

        print("\n✅ All tests passed! The pipeline fixes are working.")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logger.exception("Test failed")

if __name__ == "__main__":
    main()