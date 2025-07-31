import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool, FileWriterTool
from typing import Dict, List, Optional, Tuple

# Define structured outputs
class CompanyFinancials(BaseModel):
    """Structured output for company financial data"""
    ticker: str = Field(..., description="Stock ticker symbol")
    current_price: float = Field(..., description="Current stock price")
    shares_outstanding: float = Field(..., description="Number of shares outstanding")
    market_cap: float = Field(..., description="Market capitalization")
    enterprise_value: float = Field(..., description="Enterprise value")
    revenue_ttm: float = Field(..., description="Trailing twelve months revenue")
    ebitda_ttm: float = Field(..., description="Trailing twelve months EBITDA")
    earnings_ttm: float = Field(..., description="Trailing twelve months earnings")
    free_cash_flow_ttm: float = Field(..., description="Trailing twelve months free cash flow")
    book_value: float = Field(..., description="Book value per share")
    dividend_yield: float = Field(..., description="Dividend yield")
    growth_rate: float = Field(..., description="Expected growth rate")
    beta: float = Field(..., description="Stock beta")
    debt: float = Field(..., description="Total debt")
    cash: float = Field(..., description="Cash and equivalents")

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

# Initialize LLM
llm = LLM(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

# Initialize tools
search_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))
file_writer = FileWriterTool()

# Agent 1: Financial Data Retriever
financial_data_agent = Agent(
    role="Financial Data Analyst",
    goal="Retrieve comprehensive financial data for {company} including all necessary metrics for valuation analysis",
    backstory="""You are an expert financial analyst with access to real-time market data. 
                 You specialize in gathering accurate financial metrics including revenue, earnings, 
                 cash flow, debt levels, and market data.""",
    tools=[search_tool],
    llm=llm,
    verbose=True,
)

# Agent 2: Valuation Analyst
valuation_analyst = Agent(
    role="Senior Valuation Analyst",
    goal="Calculate all requested valuation metrics for {company} using the financial data provided",
    backstory="""You are a seasoned valuation expert with deep knowledge of various valuation methodologies 
                 including DCF, Graham formulas, multiples analysis, and asset-based valuations. 
                 You excel at calculating precise valuations and determining appropriate signals.""",
    llm=llm,
    verbose=True,
)

# Agent 3: Report Generator
report_generator = Agent(
    role="Investment Report Writer",
    goal="Generate a comprehensive investment report for {company} using the calculated valuation metrics",
    backstory="""You are an experienced investment analyst who creates clear, actionable investment reports. 
                 You excel at synthesizing complex valuation data into easy-to-understand recommendations 
                 with proper justification.""",
    tools=[file_writer],
    llm=llm,
    verbose=True,
)

# Task 1: Retrieve Financial Data
data_retrieval_task = Task(
    description="""Retrieve comprehensive financial data for {company}. You must gather:
    1. Current stock price and shares outstanding
    2. Market cap and enterprise value
    3. TTM revenue, EBITDA, earnings, and free cash flow
    4. Balance sheet data: total debt, cash, book value
    5. Historical growth rates and analyst estimates
    6. Dividend yield and payout ratio
    7. Beta and other risk metrics
    8. Segment revenue breakdown if applicable
    
    Use yfinance and web searches to ensure all data is current and accurate.""",
    expected_output="A complete CompanyFinancials object with all required financial metrics",
    output_pydantic=CompanyFinancials,
    agent=financial_data_agent,
)

# Task 2: Calculate Valuation Metrics
valuation_task = Task(
    description="""Using the financial data provided, calculate all the following valuation metrics:
    
    1. **DCF (Discounted Cash Flow)**: Project FCF for 10 years, apply appropriate discount rate
    2. **Payback Time**: Years to recover investment based on owner earnings
    3. **Owner Earnings Yield**: (FCF / Market Cap) * 100
    4. **Ben Graham Formula**: V = EPS × (8.5 + 2g) × 4.4/Y
    5. **P/E Multiple Analysis**: Compare to industry and historical averages
    6. **Asset-based Valuation**: Book value adjusted for intangibles
    7. **SOTP (Sum of Parts)**: Value each business segment separately
    8. **DDM (Dividend Discount Model)**: If applicable, using Gordon Growth Model
    9. **PEG Ratios**: Calculate for P/E, P/S, P/B, and P/FCF
    
    For each metric, determine signal (BUY/HOLD/SELL) based on:
    - BUY: >20% undervalued
    - HOLD: -20% to +20% of fair value
    - SELL: >20% overvalued""",
    expected_output="Complete ValuationMetrics with all calculations and signals",
    output_pydantic=ValuationMetrics,
    agent=valuation_analyst,
    context=[data_retrieval_task],
)

# Task 3: Generate Investment Report
report_task = Task(
    description="""Create a comprehensive investment report following this exact template:

    # Investment Analysis Report: {company}
    
    ## Investment Decision Matrix
    
    | Method | Signal | Reason |
    |--------|--------|---------|
    | DCF | **{dcf_signal}** | {dcf_reason} |
    | Payback Time | **{payback_signal}** | {payback_reason} |
    | Owner Earnings Yield | **{yield_signal}** | {yield_reason} |
    | Ben Graham Formula | **{graham_signal}** | {graham_reason} |
    | P/E Multiples | **{pe_signal}** | {pe_reason} |
    | Asset-Based | **{asset_signal}** | {asset_reason} |
    | SOTP | **{sotp_signal}** | {sotp_reason} |
    | DDM | **{ddm_signal}** | {ddm_reason} |
    | PEG Ratios | **{peg_signal}** | {peg_reason} |
    
    ## Final Assessment: **{final_verdict}**
    
    ### Vote Tally:
    - **BUY:** {buy_count} methods
    - **HOLD:** {hold_count} methods
    - **SELL:** {sell_count} methods
    - **N/A:** {na_count} methods
    
    ## Key Considerations:
    
    ### Why {primary_signal} Rather Than {alternative_signal}:
    1. **{strength_1_title}:** {strength_1_detail}
    2. **{strength_2_title}:** {strength_2_detail}
    3. **{strength_3_title}:** {strength_3_detail}
    4. **{strength_4_title}:** {strength_4_detail}
    
    ### Why Caution Is Warranted:
    1. **{risk_1_title}:** {risk_1_detail}
    2. **{risk_2_title}:** {risk_2_detail}
    3. **{risk_3_title}:** {risk_3_detail}
    4. **{risk_4_title}:** {risk_4_detail}
    
    ## Growth-Adjusted Analysis:
    {growth_analysis}
    
    ## Recommendation: **{recommendation}**
    {detailed_recommendation}
    
    **Risk-Adjusted Target:** ${target_low}-${target_high} range represents more attractive entry points.
    
    Save this report as '{company}_investment_report.md'""",
    expected_output="A complete investment report saved as a markdown file",
    agent=report_generator,
    context=[data_retrieval_task, valuation_task],
)

# Custom calculation functions for the valuation analyst to use
def calculate_dcf(fcf, growth_rate, discount_rate=0.10, terminal_growth=0.03, years=10):
    """Calculate DCF intrinsic value"""
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
    """Ben Graham's intrinsic value formula"""
    # V = EPS × (8.5 + 2g) × 4.4/Y
    if eps <= 0:
        return 0
    return eps * (8.5 + 2 * growth_rate * 100) * 4.4 / aaa_yield

# Create the valuation crew
valuation_crew = Crew(
    agents=[financial_data_agent, valuation_analyst, report_generator],
    tasks=[data_retrieval_task, valuation_task, report_task],
    process=Process.sequential,
    verbose=True,
)

# Function to run valuation analysis
def run_valuation_analysis(company: str) -> str:
    """
    Run comprehensive valuation analysis for a company
    
    Args:
        company: Company name or ticker symbol
        
    Returns:
        Path to the generated investment report
    """
    try:
        result = valuation_crew.kickoff(inputs={"company": company})
        return f"Report generated: {company}_investment_report.md"
    except Exception as e:
        return f"Error running valuation analysis: {str(e)}"

# Integration with existing code
if __name__ == "__main__":
    # Example usage
    company = input("Enter company to analyze (name or ticker): ")
    result = run_valuation_analysis(company)
    print(result)