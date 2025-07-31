import re
import json
import os
import yfinance as yf
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import CodeInterpreterTool, FileReadTool, FileWriterTool
from valuation_calculator import ValuationCalculator
from typing import Dict, List, Optional

class QueryAnalysisOutput(BaseModel):
    """Structured output for the query analysis task."""
    symbols: list[str] = Field(..., description="List of stock ticker symbols (e.g., ['TSLA', 'AAPL']).")
    timeframe: str = Field(..., description="Time period (e.g., '1d', '1mo', '1y').")
    action: str = Field(..., description="Action to be performed (e.g., 'fetch', 'plot').")

class ValuationOutput(BaseModel):
    """Structured output for valuation analysis"""
    company: str = Field(..., description="Company name")
    ticker: str = Field(..., description="Stock ticker")
    current_price: float = Field(..., description="Current stock price")
    
    # DCF Analysis
    dcf_value: float = Field(..., description="DCF intrinsic value")
    dcf_signal: str = Field(..., description="DCF signal: BUY/HOLD/SELL")
    dcf_details: str = Field(..., description="DCF calculation details")
    
    # Payback Time
    payback_years: float = Field(..., description="Payback time in years")
    payback_signal: str = Field(..., description="Payback signal: BUY/HOLD/SELL")
    payback_details: str = Field(..., description="Payback calculation details")
    
    # Owner Earnings Yield
    owner_yield: float = Field(..., description="Owner earnings yield percentage")
    owner_yield_signal: str = Field(..., description="Owner yield signal: BUY/HOLD/SELL")
    owner_yield_details: str = Field(..., description="Owner yield details")
    
    # Graham Value
    graham_value: float = Field(..., description="Ben Graham intrinsic value")
    graham_signal: str = Field(..., description="Graham signal: BUY/HOLD/SELL")
    graham_details: str = Field(..., description="Graham calculation details")
    
    # P/E Analysis
    pe_ratio: float = Field(..., description="P/E ratio")
    pe_signal: str = Field(..., description="P/E signal: BUY/HOLD/SELL")
    pe_details: str = Field(..., description="P/E analysis details")
    
    # Asset-Based
    book_value: float = Field(..., description="Book value per share")
    asset_signal: str = Field(..., description="Asset-based signal: BUY/HOLD/SELL")
    asset_details: str = Field(..., description="Asset-based analysis details")
    
    # SOTP
    sotp_value: float = Field(..., description="Sum of parts value")
    sotp_signal: str = Field(..., description="SOTP signal: BUY/HOLD/SELL")
    sotp_details: str = Field(..., description="SOTP calculation details")
    
    # DDM
    ddm_value: float = Field(..., description="Dividend discount model value")
    ddm_signal: str = Field(..., description="DDM signal: BUY/HOLD/SELL")
    ddm_details: str = Field(..., description="DDM calculation details")
    
    # PEG Ratios
    avg_peg: float = Field(..., description="Average PEG ratio")
    peg_signal: str = Field(..., description="PEG signal: BUY/HOLD/SELL")
    peg_details: str = Field(..., description="PEG analysis details")
    
    # Summary
    buy_count: int = Field(..., description="Number of BUY signals")
    hold_count: int = Field(..., description="Number of HOLD signals")
    sell_count: int = Field(..., description="Number of SELL signals")
    na_count: int = Field(..., description="Number of N/A signals")
    final_recommendation: str = Field(..., description="Final investment recommendation")

llm = LLM(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

# Original agents for crew
query_parser_agent = Agent(
    role="Stock Data Analyst",
    goal="Extract stock details and fetch required data from this user query: {query}.",
    backstory="You are a financial analyst specializing in stock market data retrieval.",
    llm=llm,
    verbose=True,
    memory=True,
)

code_writer_agent = Agent(
    role="Senior Python Developer",
    goal="Write Python code to visualize stock data.",
    backstory="""You are a Senior Python developer specializing in stock market data visualization. 
                 You are also a Pandas, Matplotlib and yfinance library expert.
                 You are skilled at writing production-ready Python code""",
    llm=llm,
    verbose=True,
)

code_execution_agent = Agent(
    role="Senior Code Execution Expert",
    goal="Review and execute the generated Python code by code writer agent to visualize stock data and fix any errors encountered.",
    backstory="You are a code execution expert. You are skilled at executing Python code.",
    allow_code_execution=False,
    allow_delegation=True,
    llm=llm,
    verbose=True,
)
data_collector_agent = Agent(
    role="Financial Data Collector",
    goal="Retrieve comprehensive financial data for {company} using the ValuationCalculator",
    backstory="""You are an expert at collecting financial data using various APIs and tools. 
                 You have deep knowledge of financial statements and can extract all necessary 
                 metrics for valuation analysis. You know how to work with the updated 
                 ValuationCalculator that returns dictionary results with 'value', 'signal', 
                 and 'details' keys.""",
    llm=llm,
    verbose=True,
)

# Updated valuation expert agent instructions
valuation_expert_agent = Agent(
    role="Valuation Expert",
    goal="Calculate all valuation metrics for {company} and determine investment signals",
    backstory="""You are a CFA-certified valuation expert with 20 years of experience. 
                 You excel at applying various valuation methodologies including DCF, 
                 Graham formulas, and comparative analysis to determine fair values.
                 You understand that the ValuationCalculator now returns structured 
                 dictionaries and you know how to extract values, signals, and details 
                 from these results.""",
    llm=llm,
    verbose=True,
)

report_writer_agent = Agent(
    role="Investment Report Writer",
    goal="Create a comprehensive investment report with clear recommendations for {company}",
    backstory="""You are a senior investment analyst who writes institutional-quality reports. 
                 You excel at synthesizing complex valuation data into actionable insights 
                 with clear buy/hold/sell recommendations.""",
    tools=[FileWriterTool()],
    llm=llm,
    verbose=True,
)

# Original tasks for crew
query_parsing_task = Task(
    description="Analyze the user query and extract stock details.",
    expected_output="A dictionary with keys: 'symbol', 'timeframe', 'action'.",
    output_pydantic=QueryAnalysisOutput,
    agent=query_parser_agent,
)

code_writer_task = Task(
    description="""Write Python code to visualize stock data based on the inputs from the stock analyst
                   where you would find stock symbol, timeframe and action.""",
    expected_output="A clean and executable Python script file (.py) for stock visualization.",
    agent=code_writer_agent,
)

code_execution_task = Task(
    description="""Review and execute the generated Python code by code writer agent to visualize stock data and fix any errors encountered.""",
    expected_output="A clean, working and executable Python script file (.py) for stock visualization.",
    agent=code_execution_agent,
)

data_collection_task = Task(
    description="""Use the ValuationCalculator to retrieve all financial data for {company}. 
    Steps:
    1. Initialize ValuationCalculator with the company ticker
    2. Call get_comprehensive_valuation() to get all metrics
    3. Extract and organize all financial data including:
       - Current price, market cap, enterprise value
       - Revenue, EBITDA, earnings, free cash flow
       - Balance sheet items (debt, cash, book value)
       - Growth rates and other metrics
    
    Return a comprehensive dictionary with all financial metrics.""",
    expected_output="Complete financial data dictionary with all metrics needed for valuation",
    agent=data_collector_agent,
)

valuation_analysis_task = Task(
    description="""Using the financial data provided by the data collector, analyze all valuation metrics:
    
    1. DCF Analysis - Calculate intrinsic value and determine if stock is undervalued/overvalued
    2. Payback Time - Calculate years to recover investment based on owner earnings
    3. Owner Earnings Yield - Compare to 10% benchmark (10-cap rule)
    4. Ben Graham Value - Apply intrinsic value formula: V = EPS × (8.5 + 2g) × 4.4/Y
    5. P/E Multiple - Compare to industry standards (BUY <15, HOLD 15-25, SELL >25)
    6. Asset-Based Value - Analyze price to book ratio
    7. SOTP Analysis - Sum of parts valuation using enterprise value
    8. DDM - Dividend discount model if dividends are paid
    9. PEG Ratios - Growth-adjusted valuations for P/E, P/S, P/B, P/FCF
    
    For each method, extract the value and signal from the dictionary results.
    Count all signals and determine the overall investment recommendation based on majority vote.
    
    The ValuationCalculator now returns dictionaries with 'value', 'signal', and 'details' keys.""",
    expected_output="Complete valuation analysis with all metrics, signals, and overall recommendation",
    output_pydantic=ValuationOutput,
    agent=valuation_expert_agent,
    context=[data_collection_task],
)

report_generation_task = Task(
    description="""Create a comprehensive investment report using the valuation analysis results.
    
    Your report should follow this structure:
    
    
    1. **Investment Decision Matrix Table**:
       - Create a table showing each valuation method, its signal, and reasoning
       - Include: DCF, Payback Time, Owner Earnings Yield, Ben Graham Formula, P/E Multiples, 
         Asset-Based, SOTP, DDM, and PEG Ratios
    
    2. **Final Assessment**:
       - State the overall recommendation (BUY/HOLD/SELL) based on majority vote
       - Show vote tally (number of BUY, HOLD, SELL, N/A signals)
    
    3. **Key Considerations**:
       - Explain why the primary signal makes sense with 4 key strengths
       - List 4 reasons why caution is warranted (risks)
    
    4. **Growth-Adjusted Analysis**:
       - Discuss how growth assumptions impact the valuation
    
    5. **Final Recommendation**:
       - Provide detailed recommendation with risk-adjusted target price range
    
    Use the valuation results from the previous task to populate all content.
    Make the report professional and actionable for institutional investors.
    Save the report as {company}_investment_report.md""",
    expected_output="Professional investment report saved as markdown file",
    agent=report_writer_agent,
    context=[data_collection_task, valuation_analysis_task],
)

# Create the crews
crew = Crew(
    agents=[query_parser_agent, code_writer_agent, code_execution_agent],
    tasks=[query_parsing_task, code_writer_task, code_execution_task],
    process=Process.sequential
)

crew1 = Crew(
    agents=[data_collector_agent, valuation_expert_agent, report_writer_agent],
    tasks=[data_collection_task, valuation_analysis_task, report_generation_task],
    process=Process.sequential,
)

# Function to be wrapped inside MCP tool
def run_financial_analysis(query):
    """Run stock visualization analysis"""
    result = crew.kickoff(inputs={"query": query})
    return result.raw



def estimate_stock_price(query):
    """Run comprehensive valuation analysis and generate investment report"""
    # Extract company/ticker from query
    import re
    # Simple extraction - you might want to enhance this
    words = query.split()
    company = None
    
    # Look for common patterns
    for word in words:
        # Check if it looks like a ticker (all caps, 1-5 letters)
        if word.isupper() and 1 <= len(word) <= 5:
            company = word
            break
    
    # If no ticker found, try to extract company name
    if not company:
        # Remove common words and take the rest as company name
        stop_words = ['analyze', 'evaluate', 'value', 'stock', 'company', 'the', 'for', 'of']
        company_words = [w for w in words if w.lower() not in stop_words]
        company = ' '.join(company_words) if company_words else query
    
    try:
        # Initialize the calculator
        calculator = get_valuation_data(company)
        
        # Run the valuation crew
        result = crew1.kickoff(inputs={
            "company": company,
            "calculator": calculator
        })
        
        return f"Valuation analysis complete for {company}. Report saved as {company}_investment_report.md"
    except Exception as e:
        return f"Error running valuation analysis: {str(e)}"

# Helper function for agents to use the calculator
def get_valuation_data(ticker: str) -> Dict:
    """Helper function for agents to get valuation data"""
    try:
        calculator = ValuationCalculator(ticker)
        return calculator.get_comprehensive_valuation()  # Returns Dict, not the object
    except Exception as e:

    
        return {"error": str(e)}
    
    
# Example of how to extract signals from the new dictionary format
def extract_signals_from_results(results: Dict) -> Dict[str, str]:
    """Extract all signals from valuation results"""
    signals = {}
    
    # Extract signals from main metrics
    for key, result in results.items():
        if key == 'financial_metrics':
            continue
        elif key == 'multiples':
            # Handle nested multiples dictionary
            for mult_key, mult_result in result.items():
                if isinstance(mult_result, dict) and 'signal' in mult_result:
                    signals[f"{key}_{mult_key}"] = mult_result['signal']
        elif isinstance(result, dict) and 'signal' in result:
            signals[key] = result['signal']
    
    return signals
if __name__ == "__main__":
    # Test the valuation crew
    company = input("Enter company to analyze (name or ticker): ")
    result = estimate_stock_price(f"analyze {company}")
    print(result)