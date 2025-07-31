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
    metrics: Dict = Field(..., description="All calculated valuation metrics")
    signals: Dict[str, str] = Field(..., description="Buy/Hold/Sell signals for each method")
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

# New agents for valuation crew (crew1)
data_collector_agent = Agent(
    role="Financial Data Collector",
    goal="Retrieve comprehensive financial data for {company} using the ValuationCalculator",
    backstory="""You are an expert at collecting financial data using various APIs and tools. 
                 You have deep knowledge of financial statements and can extract all necessary 
                 metrics for valuation analysis.""",
    llm=llm,
    verbose=True,
)

valuation_expert_agent = Agent(
    role="Valuation Expert",
    goal="Calculate all valuation metrics for {company} and determine investment signals",
    backstory="""You are a CFA-certified valuation expert with 20 years of experience. 
                 You excel at applying various valuation methodologies including DCF, 
                 Graham formulas, and comparative analysis to determine fair values.""",
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

# New tasks for valuation crew (crew1)
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
    description="""Using the financial data provided, analyze all valuation metrics:
    
    1. DCF Analysis - Determine if stock is undervalued/overvalued
    2. Payback Time - Calculate years to recover investment
    3. Owner Earnings Yield - Compare to 10% benchmark
    4. Ben Graham Value - Apply intrinsic value formula
    5. P/E Multiple - Compare to industry standards
    6. Asset-Based Value - Analyze price to book
    7. SOTP Analysis - Sum of parts valuation
    8. DDM - Dividend discount model (if applicable)
    9. PEG Ratios - Growth-adjusted valuations
    
    For each metric, provide:
    - Calculated value
    - Buy/Hold/Sell signal
    - Brief reasoning
    
    Count the signals and determine overall recommendation.""",
    expected_output="Complete valuation analysis with all metrics, signals, and overall recommendation",
    output_pydantic=ValuationOutput,
    agent=valuation_expert_agent,
    context=[data_collection_task],
)

report_generation_task = Task(
    description="""Create a comprehensive investment report following the exact template:

    # Investment Decision Matrix
    
    | Method | Signal | Reason |
    |--------|--------|---------|
    | DCF | **{signal}** | {reason} |
    | Payback Time | **{signal}** | {reason} |
    | Owner Earnings Yield | **{signal}** | {reason} |
    | Ben Graham Formula | **{signal}** | {reason} |
    | P/E Multiples | **{signal}** | {reason} |
    | Asset-Based | **{signal}** | {reason} |
    | SOTP | **{signal}** | {reason} |
    | DDM | **{signal}** | {reason} |
    | PEG Ratios | **{signal}** | {reason} |
    
    ## Final Assessment: **{verdict}**
    
    ### Vote Tally:
    - **BUY:** {buy_count} methods
    - **HOLD:** {hold_count} methods  
    - **SELL:** {sell_count} methods
    - **N/A:** {na_count} methods
    
    ## Key Considerations:
    
    ### Why {primary_signal}:
    1. **{strength_1}**
    2. **{strength_2}**
    3. **{strength_3}**
    4. **{strength_4}**
    
    ### Why Caution Is Warranted:
    1. **{risk_1}**
    2. **{risk_2}**
    3. **{risk_3}**
    4. **{risk_4}**
    
    ## Growth-Adjusted Analysis:
    {growth_analysis}
    
    ## Recommendation: **{recommendation}**
    {detailed_recommendation}
    
    **Risk-Adjusted Target:** ${target_range}
    
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
    verbose=True
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
        calculator = ValuationCalculator(company)
        
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
        return calculator.get_comprehensive_valuation()
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Test the valuation crew
    company = input("Enter company to analyze (name or ticker): ")
    result = estimate_stock_price(f"analyze {company}")
    print(result)