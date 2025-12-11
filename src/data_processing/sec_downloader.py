"""
SEC EDGAR Downloader Module
Financial Document Intelligence Platform

Downloads 10-K and 10-Q filings from SEC EDGAR database.
Respects rate limits and handles errors gracefully.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

import requests
from bs4 import BeautifulSoup
from loguru import logger


@dataclass
class FilingInfo:
    """Information about a SEC filing."""
    ticker: str
    company_name: str
    filing_type: str  # 10-K or 10-Q
    filing_date: str
    accession_number: str
    document_url: str
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[int] = None


class SECDownloader:
    """
    Download SEC filings from EDGAR database.
    
    Features:
    - Downloads 10-K and 10-Q filings
    - Respects SEC rate limits (10 requests/second)
    - Saves with organized file structure
    - Caches company CIK lookups
    """
    
    # SEC EDGAR base URLs
    BASE_URL = "https://www.sec.gov"
    EDGAR_URL = f"{BASE_URL}/cgi-bin/browse-edgar"
    SEARCH_URL = f"{BASE_URL}/cgi-bin/srch-ia"
    
    # Rate limiting
    REQUEST_DELAY = 0.1  # 100ms between requests (10 requests/second max)
    
    # Company tickers to CIK mapping cache
    CIK_CACHE_FILE = "cik_cache.json"
    
    def __init__(self, 
                 output_dir: str = "data/raw/filings",
                 user_agent: str = None):
        """
        Initialize the SEC downloader.
        
        Args:
            output_dir: Directory to save downloaded filings
            user_agent: User agent string (required by SEC)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # SEC requires a user agent with contact info
        if user_agent is None:
            user_agent = "Financial-Doc-Intelligence research@example.com"
        
        self.headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov"
        }
        
        self.last_request_time = 0
        self.cik_cache = self._load_cik_cache()
    
    def _load_cik_cache(self) -> Dict[str, str]:
        """Load cached CIK numbers."""
        cache_path = self.output_dir / self.CIK_CACHE_FILE
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
        return {}
    
    def _save_cik_cache(self):
        """Save CIK cache to disk."""
        cache_path = self.output_dir / self.CIK_CACHE_FILE
        with open(cache_path, 'w') as f:
            json.dump(self.cik_cache, f)
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Dict = None) -> requests.Response:
        """Make a rate-limited request to SEC."""
        self._rate_limit()
        
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {url} - {e}")
            raise
    
    def get_cik(self, ticker: str) -> Optional[str]:
        """
        Get CIK (Central Index Key) for a company ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            CIK number as string, or None if not found
        """
        ticker = ticker.upper()
        
        # Check cache first
        if ticker in self.cik_cache:
            return self.cik_cache[ticker]
        
        # Look up CIK from SEC
        try:
            # Use SEC company tickers endpoint
            url = f"https://www.sec.gov/cgi-bin/browse-edgar"
            params = {
                'action': 'getcompany',
                'company': ticker,
                'type': '10-',
                'dateb': '',
                'owner': 'include',
                'count': '10',
                'output': 'atom'
            }
            
            response = self._make_request(url, params)
            
            # Parse the response to find CIK
            soup = BeautifulSoup(response.content, 'lxml-xml')
            
            # Look for CIK in the response
            cik_match = re.search(r'CIK=(\d+)', response.text)
            if cik_match:
                cik = cik_match.group(1).zfill(10)  # Pad to 10 digits
                self.cik_cache[ticker] = cik
                self._save_cik_cache()
                return cik
            
        except Exception as e:
            logger.warning(f"Could not find CIK for {ticker}: {e}")
        
        return None
    
    def get_company_filings(self,
                           ticker: str,
                           filing_type: str = "10-K",
                           count: int = 10) -> List[FilingInfo]:
        """
        Get list of filings for a company.
        
        Args:
            ticker: Stock ticker symbol
            filing_type: Type of filing ('10-K' or '10-Q')
            count: Maximum number of filings to return
            
        Returns:
            List of FilingInfo objects
        """
        cik = self.get_cik(ticker)
        if not cik:
            logger.error(f"Could not find CIK for ticker: {ticker}")
            return []
        
        filings = []
        
        try:
            # Use submissions endpoint
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            response = self._make_request(url)
            data = response.json()
            
            company_name = data.get('name', ticker)
            recent_filings = data.get('filings', {}).get('recent', {})
            
            forms = recent_filings.get('form', [])
            dates = recent_filings.get('filingDate', [])
            accession_numbers = recent_filings.get('accessionNumber', [])
            primary_docs = recent_filings.get('primaryDocument', [])
            
            found_count = 0
            for i, form in enumerate(forms):
                if found_count >= count:
                    break
                
                # Check if this is the filing type we want
                if form == filing_type or (filing_type == "10-K" and form in ["10-K", "10-K/A"]):
                    accession = accession_numbers[i].replace('-', '')
                    primary_doc = primary_docs[i]
                    
                    doc_url = (f"https://www.sec.gov/Archives/edgar/data/"
                              f"{cik}/{accession}/{primary_doc}")
                    
                    filing_info = FilingInfo(
                        ticker=ticker,
                        company_name=company_name,
                        filing_type=form,
                        filing_date=dates[i],
                        accession_number=accession_numbers[i],
                        document_url=doc_url
                    )
                    
                    # Extract fiscal year from filing date
                    try:
                        date_obj = datetime.strptime(dates[i], "%Y-%m-%d")
                        filing_info.fiscal_year = date_obj.year
                    except:
                        pass
                    
                    filings.append(filing_info)
                    found_count += 1
            
            logger.info(f"Found {len(filings)} {filing_type} filings for {ticker}")
            
        except Exception as e:
            logger.error(f"Error fetching filings for {ticker}: {e}")
        
        return filings
    
    def download_filing(self,
                       ticker: str,
                       filing_type: str,
                       year: int,
                       quarter: Optional[int] = None) -> Optional[Path]:
        """
        Download a specific filing.
        
        Args:
            ticker: Stock ticker symbol
            filing_type: '10-K' or '10-Q'
            year: Fiscal year
            quarter: Quarter number for 10-Q filings (1-4)
            
        Returns:
            Path to downloaded file, or None if not found
        """
        # Get available filings
        filings = self.get_company_filings(ticker, filing_type, count=20)
        
        # Find matching filing
        target_filing = None
        for filing in filings:
            if filing.fiscal_year == year:
                if filing_type == "10-Q" and quarter:
                    # Check quarter (approximate from date)
                    try:
                        date_obj = datetime.strptime(filing.filing_date, "%Y-%m-%d")
                        filing_quarter = (date_obj.month - 1) // 3 + 1
                        if filing_quarter == quarter:
                            target_filing = filing
                            break
                    except:
                        pass
                else:
                    target_filing = filing
                    break
        
        if not target_filing:
            logger.warning(f"Could not find {filing_type} for {ticker} {year}")
            return None
        
        return self._download_document(target_filing)
    
    def _download_document(self, filing: FilingInfo) -> Optional[Path]:
        """Download the filing document."""
        # Create output path
        year = filing.fiscal_year or 'unknown'
        filename = f"{filing.ticker}_{filing.filing_type}_{year}.htm"
        output_path = self.output_dir / filing.ticker / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        if output_path.exists():
            logger.info(f"File already exists: {output_path}")
            return output_path
        
        try:
            logger.info(f"Downloading: {filing.document_url}")
            response = self._make_request(filing.document_url)
            
            # Save the file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Saved to: {output_path}")
            
            # Also save metadata
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump({
                    'ticker': filing.ticker,
                    'company_name': filing.company_name,
                    'filing_type': filing.filing_type,
                    'filing_date': filing.filing_date,
                    'accession_number': filing.accession_number,
                    'document_url': filing.document_url,
                    'fiscal_year': filing.fiscal_year,
                    'downloaded_at': datetime.now().isoformat()
                }, f, indent=2)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error downloading filing: {e}")
            return None
    
    def download_company_filings(self,
                                ticker: str,
                                start_year: int,
                                end_year: int,
                                filing_types: List[str] = None) -> List[Path]:
        """
        Download all filings for a company in a date range.
        
        Args:
            ticker: Stock ticker symbol
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            filing_types: List of filing types to download (default: ['10-K', '10-Q'])
            
        Returns:
            List of paths to downloaded files
        """
        if filing_types is None:
            filing_types = ['10-K', '10-Q']
        
        downloaded_files = []
        
        for filing_type in filing_types:
            filings = self.get_company_filings(ticker, filing_type, count=50)
            
            for filing in filings:
                if filing.fiscal_year and start_year <= filing.fiscal_year <= end_year:
                    path = self._download_document(filing)
                    if path:
                        downloaded_files.append(path)
        
        logger.info(f"Downloaded {len(downloaded_files)} filings for {ticker}")
        return downloaded_files
    
    def download_batch(self,
                      tickers: List[str],
                      start_year: int,
                      end_year: int,
                      filing_types: List[str] = None) -> Dict[str, List[Path]]:
        """
        Download filings for multiple companies.
        
        Args:
            tickers: List of stock ticker symbols
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            filing_types: Types of filings to download
            
        Returns:
            Dict mapping ticker to list of downloaded file paths
        """
        results = {}
        
        for ticker in tickers:
            logger.info(f"Processing {ticker}...")
            results[ticker] = self.download_company_filings(
                ticker, start_year, end_year, filing_types
            )
            time.sleep(1)  # Extra delay between companies
        
        return results


# Utility functions
def download_top_companies(output_dir: str = "data/raw/filings",
                          start_year: int = 2020,
                          end_year: int = 2023) -> Dict[str, List[Path]]:
    """
    Download filings for top 10 companies mentioned in the project spec.
    
    Args:
        output_dir: Directory to save files
        start_year: Start year
        end_year: End year
        
    Returns:
        Dict mapping ticker to downloaded file paths
    """
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
               'META', 'NVDA', 'JPM', 'BAC', 'GS']
    
    downloader = SECDownloader(output_dir=output_dir)
    return downloader.download_batch(tickers, start_year, end_year)


if __name__ == "__main__":
    import sys
    
    downloader = SECDownloader()
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
        year = int(sys.argv[2]) if len(sys.argv) > 2 else 2023
        
        # Download 10-K for the specified ticker and year
        path = downloader.download_filing(ticker, "10-K", year)
        if path:
            print(f"Downloaded: {path}")
        else:
            print(f"Could not download 10-K for {ticker} {year}")
    else:
        print("Usage: python sec_downloader.py <TICKER> [YEAR]")
        print("Example: python sec_downloader.py AAPL 2023")
