import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta

class ValuationCalculator:
    """
    A comprehensive valuation calculator for financial analysis
    """
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.info = self.stock.info
        self.financials = None
        self.cash_flow = None
        self.balance_sheet = None
        self._load_financial_data()
    
    def _load_financial_data(self):
        """Load all financial statements"""
        try:
            self.financials = self.stock.financials
            self.cash_flow = self.stock.cash_flow
            self.balance_sheet = self.stock.balance_sheet
            self.quarterly_financials = self.stock.quarterly_financials
            self.quarterly_cash_flow = self.stock.quarterly_cash_flow
        except Exception as e:
            print(f"Error loading financial data: {e}")
    
    def get_financial_metrics(self) -> Dict:
        """Extract all required financial metrics"""
        try:
            metrics = {
                'ticker': self.ticker,
                'current_price': self.info.get('currentPrice', 0),
                'shares_outstanding': self.info.get('sharesOutstanding', 0),
                'market_cap': self.info.get('marketCap', 0),
                'enterprise_value': self.info.get('enterpriseValue', 0),
                'revenue_ttm': self.info.get('totalRevenue', 0),
                'ebitda_ttm': self.info.get('ebitda', 0),
                'earnings_ttm': self.info.get('trailingEps', 0) * self.info.get('sharesOutstanding', 0),
                'free_cash_flow_ttm': self.info.get('freeCashflow', 0),
                'book_value': self.info.get('bookValue', 0),
                'dividend_yield': self.info.get('dividendYield', 0) or 0,
                'growth_rate': self._calculate_growth_rate(),
                'beta': self.info.get('beta', 1),
                'debt': self.info.get('totalDebt', 0),
                'cash': self.info.get('totalCash', 0),
                'pe_ratio': self.info.get('trailingPE', 0),
                'peg_ratio': self.info.get('pegRatio', 0),
                'price_to_book': self.info.get('priceToBook', 0),
                'price_to_sales': self.info.get('priceToSalesTrailing12Months', 0),
                'ev_to_ebitda': self.info.get('enterpriseToEbitda', 0),
            }
            
            # Calculate additional metrics
            metrics['eps'] = self.info.get('trailingEps', 0)
            metrics['owner_earnings'] = self._calculate_owner_earnings()
            
            return metrics
        except Exception as e:
            print(f"Error getting financial metrics: {e}")
            return {}
    
    def _calculate_growth_rate(self) -> float:
        """Calculate historical revenue growth rate"""
        try:
            if self.financials is not None and 'Total Revenue' in self.financials.index:
                revenues = self.financials.loc['Total Revenue'].dropna()
                if len(revenues) >= 2:
                    # Calculate CAGR
                    years = len(revenues)
                    growth_rate = (revenues.iloc[0] / revenues.iloc[-1]) ** (1/(years-1)) - 1
                    return abs(growth_rate)  # Return absolute value
            
            # Fallback to analyst estimates
            return self.info.get('revenueGrowth', 0.05) or 0.05
        except:
            return 0.05  # Default 5% growth
    
    def _calculate_owner_earnings(self) -> float:
        """Calculate Buffett's Owner Earnings"""
        try:
            fcf = self.info.get('freeCashflow', 0)
            if fcf > 0:
                return fcf
            
            # Alternative calculation
            if self.cash_flow is not None:
                operating_cf = self.cash_flow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in self.cash_flow.index else 0
                capex = abs(self.cash_flow.loc['Capital Expenditure'].iloc[0]) if 'Capital Expenditure' in self.cash_flow.index else 0
                return operating_cf - capex
            
            return 0
        except:
            return 0
    
    def calculate_dcf(self, growth_rate: float = None, discount_rate: float = 0.10, 
                      terminal_growth: float = 0.03, years: int = 10) -> Tuple[float, str]:
        """Calculate DCF intrinsic value per share"""
        try:
            metrics = self.get_financial_metrics()
            fcf = metrics['free_cash_flow_ttm']
            shares = metrics['shares_outstanding']
            current_price = metrics['current_price']
            
            if growth_rate is None:
                growth_rate = metrics['growth_rate']
            
            if fcf <= 0 or shares <= 0:
                return 0, "N/A"
            
            # Project future cash flows
            fcf_projections = []
            for i in range(1, years + 1):
                if i <= 5:
                    projected_fcf = fcf * (1 + growth_rate) ** i
                else:
                    # Decay growth rate
                    adj_growth = growth_rate * (0.9 ** (i - 5))
                    projected_fcf = fcf_projections[-1] * (1 + max(adj_growth, terminal_growth))
                fcf_projections.append(projected_fcf)
            
            # Calculate present values
            pv_fcf = sum([fcf / (1 + discount_rate) ** i for i, fcf in enumerate(fcf_projections, 1)])
            
            # Terminal value
            terminal_fcf = fcf_projections[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)
            pv_terminal = terminal_value / (1 + discount_rate) ** years
            
            # Total value per share
            total_value = pv_fcf + pv_terminal + metrics['cash'] - metrics['debt']
            intrinsic_value = total_value / shares
            
            # Determine signal
            margin = (intrinsic_value - current_price) / current_price
            if margin > 0.20:
                signal = "BUY"
            elif margin < -0.20:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            return intrinsic_value, signal
        except Exception as e:
            print(f"DCF calculation error: {e}")
            return 0, "N/A"
    
    def calculate_payback_time(self) -> Tuple[float, str]:
        """Calculate payback time in years"""
        try:
            metrics = self.get_financial_metrics()
            market_cap = metrics['market_cap']
            owner_earnings = metrics['owner_earnings']
            growth_rate = metrics['growth_rate']
            
            if owner_earnings <= 0:
                return float('inf'), "N/A"
            
            cumulative = 0
            years = 0
            current_earnings = owner_earnings
            
            while cumulative < market_cap and years < 100:
                years += 1
                current_earnings *= (1 + growth_rate)
                cumulative += current_earnings
            
            # Determine signal
            if years <= 10:
                signal = "BUY"
            elif years <= 20:
                signal = "HOLD"
            else:
                signal = "SELL"
            
            return years, signal
        except:
            return float('inf'), "N/A"
    
    def calculate_owner_earnings_yield(self) -> Tuple[float, str]:
        """Calculate owner earnings yield (10 Cap)"""
        try:
            metrics = self.get_financial_metrics()
            owner_earnings = metrics['owner_earnings']
            market_cap = metrics['market_cap']
            
            if market_cap <= 0:
                return 0, "N/A"
            
            yield_pct = (owner_earnings / market_cap) * 100
            
            # Determine signal (10% is the benchmark)
            if yield_pct >= 10:
                signal = "BUY"
            elif yield_pct >= 5:
                signal = "HOLD"
            else:
                signal = "SELL"
            
            return yield_pct, signal
        except:
            return 0, "N/A"
    
    def calculate_graham_value(self, aaa_yield: float = 4.4) -> Tuple[float, str]:
        """Calculate Ben Graham's intrinsic value"""
        try:
            metrics = self.get_financial_metrics()
            eps = metrics['eps']
            growth_rate = metrics['growth_rate'] * 100  # Convert to percentage
            current_price = metrics['current_price']
            
            if eps <= 0:
                return 0, "N/A"
            
            # V = EPS × (8.5 + 2g) × 4.4/Y
            intrinsic_value = eps * (8.5 + 2 * growth_rate) * 4.4 / aaa_yield
            
            # Determine signal
            margin = (intrinsic_value - current_price) / current_price
            if margin > 0.20:
                signal = "BUY"
            elif margin < -0.20:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            return intrinsic_value, signal
        except:
            return 0, "N/A"
    
    def analyze_multiples(self) -> Dict[str, Tuple[float, str]]:
        """Analyze P/E and EV/EBITDA multiples"""
        try:
            metrics = self.get_financial_metrics()
            pe_ratio = metrics['pe_ratio']
            ev_ebitda = metrics['ev_to_ebitda']
            
            results = {}
            
            # P/E Analysis
            if pe_ratio > 0:
                if pe_ratio < 15:
                    pe_signal = "BUY"
                elif pe_ratio < 25:
                    pe_signal = "HOLD"
                else:
                    pe_signal = "SELL"
            else:
                pe_signal = "N/A"
            results['PE'] = (pe_ratio, pe_signal)
            
            # EV/EBITDA Analysis
            if ev_ebitda > 0:
                if ev_ebitda < 10:
                    ev_signal = "BUY"
                elif ev_ebitda < 15:
                    ev_signal = "HOLD"
                else:
                    ev_signal = "SELL"
            else:
                ev_signal = "N/A"
            results['EV_EBITDA'] = (ev_ebitda, ev_signal)
            
            return results
        except:
            return {'PE': (0, "N/A"), 'EV_EBITDA': (0, "N/A")}
    
    def calculate_asset_based_value(self) -> Tuple[float, str]:
        """Calculate asset-based valuation"""
        try:
            metrics = self.get_financial_metrics()
            book_value = metrics['book_value']
            current_price = metrics['current_price']
            price_to_book = metrics['price_to_book']
            
            if book_value <= 0:
                return 0, "N/A"
            
            # Determine signal based on P/B ratio
            if price_to_book < 1:
                signal = "BUY"
            elif price_to_book < 3:
                signal = "HOLD"
            else:
                signal = "SELL"
            
            return book_value, signal
        except:
            return 0, "N/A"
    
    def calculate_sotp(self) -> Tuple[float, str]:
        """Calculate Sum of the Parts valuation"""
        try:
            # This is simplified - in reality, you'd need segment data
            metrics = self.get_financial_metrics()
            enterprise_value = metrics['enterprise_value']
            shares = metrics['shares_outstanding']
            current_price = metrics['current_price']
            
            if enterprise_value <= 0 or shares <= 0:
                return 0, "N/A"
            
            # Simplified SOTP using EV
            sotp_value = (enterprise_value + metrics['cash'] - metrics['debt']) / shares
            
            # Determine signal
            margin = (sotp_value - current_price) / current_price
            if margin > 0.20:
                signal = "BUY"
            elif margin < -0.20:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            return sotp_value, signal
        except:
            return 0, "N/A"
    
    def calculate_ddm(self, required_return: float = 0.10) -> Tuple[float, str]:
        """Calculate Dividend Discount Model value"""
        try:
            metrics = self.get_financial_metrics()
            dividend_yield = metrics['dividend_yield']
            current_price = metrics['current_price']
            
            if dividend_yield <= 0:
                return 0, "N/A"
            
            # Get dividend per share
            annual_dividend = current_price * dividend_yield
            growth_rate = min(metrics['growth_rate'], required_return - 0.01)  # Ensure growth < required return
            
            # Gordon Growth Model: V = D1 / (r - g)
            next_dividend = annual_dividend * (1 + growth_rate)
            intrinsic_value = next_dividend / (required_return - growth_rate)
            
            # Determine signal
            margin = (intrinsic_value - current_price) / current_price
            if margin > 0.20:
                signal = "BUY"
            elif margin < -0.20:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            return intrinsic_value, signal
        except:
            return 0, "N/A"
    
    def calculate_peg_ratios(self) -> Tuple[Dict[str, float], str]:
        """Calculate PEG ratios for various metrics"""
        try:
            metrics = self.get_financial_metrics()
            growth_rate_pct = metrics['growth_rate'] * 100
            
            peg_ratios = {}
            
            # PEG for P/E
            if metrics['pe_ratio'] > 0 and growth_rate_pct > 0:
                peg_ratios['PE_PEG'] = metrics['pe_ratio'] / growth_rate_pct
            else:
                peg_ratios['PE_PEG'] = 0
            
            # PEG for P/S
            if metrics['price_to_sales'] > 0 and growth_rate_pct > 0:
                peg_ratios['PS_PEG'] = metrics['price_to_sales'] / growth_rate_pct
            else:
                peg_ratios['PS_PEG'] = 0
            
            # PEG for P/B
            if metrics['price_to_book'] > 0 and growth_rate_pct > 0:
                peg_ratios['PB_PEG'] = metrics['price_to_book'] / growth_rate_pct
            else:
                peg_ratios['PB_PEG'] = 0
            
            # PEG for P/FCF
            if metrics['market_cap'] > 0 and metrics['free_cash_flow_ttm'] > 0 and growth_rate_pct > 0:
                p_fcf = metrics['market_cap'] / metrics['free_cash_flow_ttm']
                peg_ratios['PFCF_PEG'] = p_fcf / growth_rate_pct
            else:
                peg_ratios['PFCF_PEG'] = 0
            
            # Determine overall signal
            valid_pegs = [v for v in peg_ratios.values() if v > 0]
            if valid_pegs:
                avg_peg = sum(valid_pegs) / len(valid_pegs)
                if avg_peg < 1:
                    signal = "BUY"
                elif avg_peg < 2:
                    signal = "HOLD"
                else:
                    signal = "SELL"
            else:
                signal = "N/A"
            
            return peg_ratios, signal
        except:
            return {}, "N/A"
    
    def get_comprehensive_valuation(self) -> Dict:
        """Get all valuation metrics in one call"""
        results = {
            'financial_metrics': self.get_financial_metrics(),
            'dcf': self.calculate_dcf(),
            'payback_time': self.calculate_payback_time(),
            'owner_earnings_yield': self.calculate_owner_earnings_yield(),
            'graham_value': self.calculate_graham_value(),
            'multiples': self.analyze_multiples(),
            'asset_based': self.calculate_asset_based_value(),
            'sotp': self.calculate_sotp(),
            'ddm': self.calculate_ddm(),
            'peg_ratios': self.calculate_peg_ratios(),
        }
        return results