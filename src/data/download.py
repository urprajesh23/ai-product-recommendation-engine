"""
Data Download Module
Downloads Amazon Product Reviews dataset
"""

import pandas as pd
import gzip
import json
from pathlib import Path
import urllib.request
from tqdm import tqdm
import os

class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

class AmazonDataDownloader:
    """
    Download Amazon Product Reviews Dataset
    Dataset source: http://jmcauley.ucsd.edu/data/amazon/
    """
    
    # Available dataset URLs (5-core means users and items have at least 5 reviews)
    DATASETS = {
        'electronics': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz',
        'books': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz',
        'movies': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz',
        'clothing': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry_5.json.gz',
    }
    
    def __init__(self, data_dir='data/raw'):
        """
        Initialize downloader
        
        Args:
            data_dir: Directory to save downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download(self, category='electronics', force=False):
        """
        Download dataset for specified category
        
        Args:
            category: Dataset category (electronics, books, movies, clothing)
            force: Force re-download even if file exists
            
        Returns:
            Path to downloaded file
        """
        if category not in self.DATASETS:
            raise ValueError(f"Category must be one of {list(self.DATASETS.keys())}")
        
        url = self.DATASETS[category]
        filename = f'reviews_{category}.json.gz'
        output_path = self.data_dir / filename
        
        # Check if file already exists
        if output_path.exists() and not force:
            print(f"✅ File already exists: {output_path}")
            print(f"   Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
            return output_path
        
        print(f"📥 Downloading {category} dataset...")
        print(f"   URL: {url}")
        print(f"   Destination: {output_path}")
        
        try:
            download_url(url, output_path)
            file_size = output_path.stat().st_size / (1024*1024)
            print(f"✅ Download complete! Size: {file_size:.2f} MB")
            return output_path
            
        except Exception as e:
            print(f"❌ Download failed: {str(e)}")
            if output_path.exists():
                output_path.unlink()  # Remove partial download
            raise
    
    def parse_json_gz(self, filepath, max_rows=None):
        """
        Parse gzipped JSON file to DataFrame
        
        Args:
            filepath: Path to .json.gz file
            max_rows: Maximum number of rows to read (None for all)
            
        Returns:
            pandas DataFrame
        """
        print(f"📖 Parsing {filepath.name}...")
        
        data = []
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Reading lines")):
                if max_rows and i >= max_rows:
                    break
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"⚠️  Skipping malformed line {i}")
                    continue
        
        df = pd.DataFrame(data)
        print(f"✅ Parsed {len(df):,} reviews")
        
        return df
    
    def download_and_parse(self, category='electronics', max_rows=None, force=False):
        """
        Download and parse dataset in one step
        
        Args:
            category: Dataset category
            max_rows: Maximum rows to parse (None for all)
            force: Force re-download
            
        Returns:
            pandas DataFrame
        """
        filepath = self.download(category, force=force)
        df = self.parse_json_gz(filepath, max_rows=max_rows)
        
        return df
    
    def save_to_csv(self, df, category='electronics'):
        """
        Save DataFrame to CSV for faster loading
        
        Args:
            df: pandas DataFrame
            category: Dataset category (for filename)
        """
        csv_path = self.data_dir / f'reviews_{category}.csv'
        print(f"💾 Saving to CSV: {csv_path}")
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved {len(df):,} rows")
        
        return csv_path


def main():
    """Main execution function"""
    print("=" * 60)
    print("Amazon Product Reviews Dataset Downloader")
    print("=" * 60)
    
    # Initialize downloader
    downloader = AmazonDataDownloader(data_dir='data/raw')
    
    # Download and parse electronics dataset
    # Start with a smaller sample for testing (remove max_rows for full dataset)
    df = downloader.download_and_parse(
        category='electronics',
        max_rows=500000,  # Start with 500k rows for testing
        force=False
    )
    
    # Display basic info
    print("\n" + "=" * 60)
    print("Dataset Information")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # Save to CSV for faster loading next time
    downloader.save_to_csv(df, category='electronics')
    
    print("\n✅ Download complete! Data saved to data/raw/")


if __name__ == "__main__":
    main()