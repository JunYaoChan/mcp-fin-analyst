import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool, FileWriterTool
from typing import Dict, List, Optional, Tuple

# Get current date
CURRENT_DATE = datetime.now().strftime("%B %d, %Y")  # July 31, 2025
CURRENT_DATE_SHORT = datetime.now().strftime("%Y-%m-%d")  # 2025-07-31

# Define structured outputs
class CompanyFinancials(BaseModel):
    """Structured output for company financial data"""
    ticker: str = Field(..., description="Stock ticker symbol")
    current_price: float = Field(..., description="Current stock price")
    price_date: str = Field(..., description="Date of current price data")
    shares_outstanding: float = Field(..., description="Number of shares outstanding")
    market_cap: float = Field(..., description="Market capitalization")
    enterprise_value: float = Field(..., description="Enterprise value")
    revenue_ttm: float = Field(..., description="Trailing twelve months revenue")
    revenue_period: str = Field(..., description="Period for TTM revenue (e.g., 'TTM ending Q2 2025')")
    ebitda_ttm: float = Field(..., description="Trailing twelve months EBITDA")
    earnings_ttm: float = Field(..., description="Trailing twelve months earnings")
    earnings_period: str = Field(..., description="Period for TTM earnings")
    free_cash_flow_ttm: float = Field(..., description="Trailing twelve months free cash flow")
    fcf_period: str = Field(..., description="Period for TTM free cash flow")
    book_value: float = Field(..., description="Book value per share")
    book_value_date: str = Field(..., description="Date of book value data")
    dividend_yield: float = Field(..., description="Dividend yield")
    growth_rate: float = Field(..., description="Expected growth rate")
    growth_source: str = Field(..., description="Source of growth rate estimate")
    beta: float = Field(..., description="Stock beta")
    debt: float = Field(..., description="Total debt")
    cash: float = Field(..., description="Cash and equivalents")
    balance_sheet_date: str = Field(..., description="Date of balance sheet data")
    data_sources: Dict[str, str] = Field(..., description="Dictionary of data sources and their dates")
    data_as_of: str = Field(default=CURRENT_DATE, description="Date when data was retrieved")

class ValuationMetrics(BaseModel):
    """Structured output for all valuation metrics"""
    dcf_value: float = Field(..., description="DCF intrinsic value per share")
    dcf_signal: str = Field(..., description="DCF signal: BUY/HOLD/SELL")
    payback_time: float = Field(..., description="Payback time in years")
    payback_signal: str = Field(..., description="Payback signal: BUY/HOLD/SELL")
    owner_earnings_yield: float = Field(..., description="Owner earnings yield percentage")
    owner_yield_signal: str = Field(..., description="Owner earnings signal: BUY/HOLD/SELL")
    graham_value: float = Field(..., description="Ben Graham intrinsic value")
    graham_signal: str = Field(..., description="Graham signal: BUY/HOLD/SELL")
    pe_ratio: float = Field(..., description="P/E ratio")
    pe_signal: str = Field(..., description="P/E signal: BUY/HOLD/SELL")
    asset_based_value: float = Field(..., description="Asset-based value per share")
    asset_signal: str = Field(..., description="Asset-based signal: BUY/HOLD/SELL")
    sotp_value: float = Field(..., description="Sum of the parts value")
    sotp_signal: str = Field(..., description="SOTP signal: BUY/HOLD/SELL")
    ddm_value: Optional[float] = Field(None, description="Dividend discount model value")
    ddm_signal: str = Field(..., description="DDM signal: BUY/HOLD/SELL/N/A")
    peg_ratios: Dict[str, float] = Field(..., description="PEG ratios for P/E, P/S, P/B, P/FCF")
    peg_signal: str = Field(..., description="PEG signal: BUY/HOLD/SELL")
    valuation_date: str = Field(default=CURRENT_DATE, description="Date of valuation analysis")

# Initialize LLM
llm = LLM(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

# Initialize tools
search_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))
file_writer = FileWriterTool()

# Agent 1: Financial Data Retriever (Updated with current date)
financial_data_agent = Agent(
    role="Financial Data Analyst",
    goal=f"Retrieve the most current and comprehensive financial data for {{company}} as of {CURRENT_DATE}",
    backstory=f"""You are an expert financial analyst with access to real-time market data as of {CURRENT_DATE}. 
                 You specialize in gathering the most current and accurate financial metrics including revenue, earnings, 
                 cash flow, debt levels, and market data. You always verify that your data is the most recent available
                 and clearly state the data vintage in your analysis.""",
    tools=[search_tool],
    llm=llm,
    verbose=True,
)

# Agent 2: Valuation Analyst (Updated with current date awareness)
valuation_analyst = Agent(
    role="Senior Valuation Analyst",
    goal=f"Calculate all requested valuation metrics for {{company}} using the most current financial data available as of {CURRENT_DATE}",
    backstory=f"""You are a seasoned valuation expert with deep knowledge of various valuation methodologies 
                 including DCF, Graham formulas, multiples analysis, and asset-based valuations. 
                 You excel at calculating precise valuations using the most current data as of {CURRENT_DATE} 
                 and determining appropriate investment signals based on today's market conditions.""",
    llm=llm,
    verbose=True,
)

# Agent 3: Report Generator (Updated with current date)
report_generator = Agent(
    role="Investment Report Writer",
    goal=f"Generate a comprehensive investment report for {{company}} dated {CURRENT_DATE} using the calculated valuation metrics",
    backstory=f"""You are an experienced investment analyst who creates clear, actionable investment reports 
                 dated {CURRENT_DATE}. You excel at synthesizing complex valuation data into easy-to-understand 
                 recommendations with proper justification, always ensuring the report reflects the current 
                 market environment and most recent available data.""",
    tools=[file_writer],
    llm=llm,
    verbose=True,
)

# Task 1: Retrieve Financial Data (Updated with specific date requirements)
data_retrieval_task = Task(
    description=f"""Retrieve the most comprehensive and current financial data for {{company}} as of {CURRENT_DATE}. 
    You must gather the following data ensuring it is the most recent available AND CAPTURE THE SOURCE DATE FOR EACH DATA POINT:
    
    1. **Current Market Data (as of {CURRENT_DATE_SHORT})**:
       - Current stock price (most recent closing price) - RECORD THE EXACT DATE
       - Shares outstanding (latest count) - RECORD THE FILING DATE
       - Market capitalization (current) - RECORD CALCULATION DATE
       - Enterprise value (current) - RECORD CALCULATION DATE
    
    2. **Financial Performance (most recent TTM data)**:
       - TTM revenue, EBITDA, earnings, and free cash flow - RECORD THE TTM PERIOD (e.g., "TTM ending Q2 2025")
       - Latest quarterly results if available - RECORD THE QUARTER AND FILING DATE
       - Year-over-year growth comparisons - RECORD THE COMPARISON PERIODS
    
    3. **Balance Sheet Data (most recent quarter)**:
       - Total debt, cash and equivalents, book value - RECORD THE QUARTER END DATE
       - Working capital and current ratio - RECORD THE STATEMENT DATE
    
    4. **Forward-Looking Data**:
       - Analyst estimates for next 12 months (as of {CURRENT_DATE}) - RECORD THE ESTIMATE DATE AND SOURCE
       - Historical growth rates and projected growth - RECORD THE SOURCE AND DATE
       - Recent analyst revisions and target prices - RECORD THE REVISION DATES
    
    5. **Market Metrics (current)**:
       - Dividend yield and payout ratio - RECORD THE EX-DIVIDEND DATE AND CALCULATION DATE
       - Beta and volatility measures - RECORD THE CALCULATION PERIOD
       - Trading volumes and liquidity metrics - RECORD THE DATA PERIOD
    
    6. **Business Segments** (if applicable):
       - Revenue breakdown by segment (latest available) - RECORD THE REPORTING PERIOD
       - Geographic revenue distribution - RECORD THE REPORTING PERIOD
    
    **CRITICAL REQUIREMENT**: For every single data point, you must capture and record:
    - The exact date when the data was retrieved
    - The period the data represents (e.g., "Q2 2025", "TTM ending June 30, 2025")
    - The source of the data (e.g., "10-K filing", "Yahoo Finance", "Analyst consensus")
    - The filing date or publication date of the source
    
    **Important**: Use search tools for real-time data and supplement with web searches for the most current 
    analyst estimates, recent earnings reports, and any material news from {CURRENT_DATE} or recent days.
    Create a comprehensive data_sources dictionary that maps each data point to its source and date.
    
    This information is crucial for transparency and allowing users to assess the recency and reliability of the analysis.""",
    expected_output=f"A complete CompanyFinancials object with all required financial metrics as of {CURRENT_DATE}, including detailed source dates and data vintage information",
    output_pydantic=CompanyFinancials,
    agent=financial_data_agent,
)

# Task 2: Calculate Valuation Metrics (Updated with current date context)
valuation_task = Task(
    description=f"""Using the financial data provided, calculate all the following valuation metrics as of {CURRENT_DATE}:
    
    **Valuation Methods to Calculate:**
    
    1. **DCF (Discounted Cash Flow)**: 
       - Project FCF for 10 years using current growth assumptions
       - Use current risk-free rate and market risk premium as of {CURRENT_DATE}
       - Apply appropriate discount rate reflecting current market conditions
    
    2. **Payback Time**: 
       - Years to recover investment based on current owner earnings
       - Factor in current growth trajectory
    
    3. **Owner Earnings Yield**: 
       - (FCF / Current Market Cap) * 100
       - Compare to current 10-year treasury yield for context
    
    4. **Ben Graham Formula**: 
       - V = EPS × (8.5 + 2g) × 4.4/Y
       - Use current AAA corporate bond yield (Y) as of {CURRENT_DATE}
    
    5. **P/E Multiple Analysis**: 
       - Compare to current industry averages and historical norms
       - Consider current market P/E environment
    
    6. **Asset-based Valuation**: 
       - Book value adjusted for current market conditions
       - Consider asset repricing in current environment
    
    7. **SOTP (Sum of Parts)**: 
       - Value each business segment using current market multiples
       - Apply current industry-specific valuation multiples
    
    8. **DDM (Dividend Discount Model)**: 
       - If applicable, use current dividend yield and growth assumptions
       - Factor in current interest rate environment
    
    9. **PEG Ratios**: 
       - Calculate for P/E, P/S, P/B, and P/FCF using current metrics
       - Compare to current market PEG norms
    
    **Signal Determination (based on current market conditions as of {CURRENT_DATE})**:
    - **BUY**: >20% undervalued relative to current market conditions
    - **HOLD**: -20% to +20% of fair value in current market
    - **SELL**: >20% overvalued in current market environment
    
    Always provide context on how current market conditions ({CURRENT_DATE}) affect these valuations.""",
    expected_output=f"Complete ValuationMetrics with all calculations and signals as of {CURRENT_DATE}",
    output_pydantic=ValuationMetrics,
    agent=valuation_analyst,
    context=[data_retrieval_task],
)

# Task 3: Generate Investment Report (Fixed - removed template variables)
report_task = Task(
    description=f"""Create a comprehensive investment report for {{company}} dated {CURRENT_DATE} using the financial data and valuation metrics from previous tasks.

    You must create a professional investment report that includes:

    1. **Report Header**: Company name, current date ({CURRENT_DATE}), and current stock price
    
    2. **Data Sources and Vintage**: A detailed section showing the date and source of each key data point used in the analysis, including:
       - Stock price data (date and source)
       - Financial statements (quarter/year and filing date)
       - Analyst estimates (date of estimates and source)
       - Market data (date retrieved)
       - Any other key data points with their respective dates and sources
    
    3. **Investment Decision Matrix**: A table showing each valuation method, its signal (BUY/HOLD/SELL), and reasoning
    
    4. **Final Assessment**: Overall recommendation based on the majority of signals
    
    5. **Vote Tally**: Count of BUY, HOLD, SELL, and N/A signals from all methods
    
    6. **Key Considerations**: 
       - Strengths supporting the primary signal
       - Risks and cautionary factors
    
    7. **Current Market Context**: How current market conditions affect the analysis
    
    8. **Growth-Adjusted Analysis**: Assessment considering growth prospects
    
    9. **Final Recommendation**: Clear actionable recommendation with target price ranges
    
    10. **Data Disclaimer**: A note about the recency and reliability of the data sources used
    
    **Format Requirements**:
    - Use markdown formatting
    - Include tables for the decision matrix and data sources
    - Bold key findings and signals
    - Use headers and subheaders for organization
    - Include specific numbers and percentages from the analysis
    - Provide clear reasoning for each signal
    - Clearly show data vintage for transparency
    
    **File Naming**: Save the report as '[COMPANY_TICKER]_investment_report_{CURRENT_DATE_SHORT}.md'
    
    Use all the data from the CompanyFinancials and ValuationMetrics objects provided by previous tasks to populate the report with specific numbers, calculations, and recommendations. Ensure the data sources section clearly shows when each piece of information was retrieved and what period it represents.""",
    expected_output=f"A complete investment report dated {CURRENT_DATE} saved as a markdown file",
    agent=report_generator,
    context=[data_retrieval_task, valuation_task],
)

# Custom calculation functions (updated with current date awareness)
def calculate_dcf(fcf, growth_rate, discount_rate=0.10, terminal_growth=0.03, years=10):
    """Calculate DCF intrinsic value with current market assumptions"""
    # You might want to update discount_rate based on current risk-free rate
    current_risk_free_rate = 0.045  # Update this based on current 10-year treasury
    market_risk_premium = 0.055     # Update based on current market conditions
    # discount_rate could be: current_risk_free_rate + market_risk_premium
    
    fcf_projections = []
    for i in range(1, years + 1):
        if i <= 5:
            fcf_projections.append(fcf * (1 + growth_rate) ** i)
        else:
            # Decay growth rate
            adj_growth = growth_rate * (0.9 ** (i - 5))
            fcf_projections.append(fcf_projections[-1] * (1 + adj_growth))
    
    # Calculate present values
    pv_fcf = sum([fcf / (1 + discount_rate) ** i for i, fcf in enumerate(fcf_projections, 1)])
    
    # Terminal value
    terminal_fcf = fcf_projections[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1 + discount_rate) ** years
    
    return pv_fcf + pv_terminal

def calculate_payback_time(market_cap, owner_earnings, growth_rate):
    """Calculate years to recover investment"""
    if owner_earnings <= 0:
        return float('inf')
    
    cumulative = 0
    years = 0
    current_earnings = owner_earnings
    
    while cumulative < market_cap and years < 100:
        years += 1
        current_earnings *= (1 + growth_rate)
        cumulative += current_earnings
    
    return years

def calculate_graham_value(eps, growth_rate, aaa_yield=4.4):
    """Ben Graham's intrinsic value formula with current bond yields"""
    # Update aaa_yield with current AAA corporate bond yield
    current_aaa_yield = 4.8  # Update this with current AAA corporate bond yield
    
    if eps <= 0:
        return 0
    return eps * (8.5 + 2 * growth_rate * 100) * 4.4 / current_aaa_yield

# Create the valuation crew
valuation_crew = Crew(
    agents=[financial_data_agent, valuation_analyst, report_generator],
    tasks=[data_retrieval_task, valuation_task, report_task],
    process=Process.sequential,
    verbose=True,
)

# Function to run valuation analysis (updated with current date)
def run_valuation_analysis(company: str) -> str:
    """
    Run comprehensive valuation analysis for a company using current market data
    
    Args:
        company: Company name or ticker symbol
        
    Returns:
        Path to the generated investment report
    """
    try:
        print(f"Starting valuation analysis for {company} as of {CURRENT_DATE}")
        print(f"Ensuring all data is current as of {CURRENT_DATE_SHORT}")
        
        result = valuation_crew.kickoff(inputs={
            "company": company,
            "current_date": CURRENT_DATE,
            "current_date_short": CURRENT_DATE_SHORT
        })
        
        report_filename = f"{company}_investment_report_{CURRENT_DATE_SHORT}.md"
        return f"Report generated: {report_filename} (dated {CURRENT_DATE})"
    except Exception as e:
        return f"Error running valuation analysis: {str(e)}"

# Integration with existing code
if __name__ == "__main__":
    print(f"Financial Analysis System - Current Date: {CURRENT_DATE}")
    print(f"System will retrieve the most current data available as of {CURRENT_DATE_SHORT}")
    
    # Example usage
    company = input("Enter company to analyze (name or ticker): ")
    result = run_valuation_analysis(company)
    print(result)