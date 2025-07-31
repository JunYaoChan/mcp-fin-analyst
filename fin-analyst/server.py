from mcp.server.fastmcp import FastMCP
from finance_crew import run_financial_analysis, estimate_stock_price
import json

# create FastMCP instance
mcp = FastMCP("financial-analyst")

@mcp.tool()
def analyze_stock(query: str) -> str:
    """
    Analyzes stock market data based on the query and generates executable Python code for analysis and visualization.
    Returns a formatted Python script ready for execution.
    
    The query is a string that must contain the stock symbol (e.g., TSLA, AAPL, NVDA, etc.), 
    timeframe (e.g., 1d, 1mo, 1y), and action to perform (e.g., plot, analyze, compare).

    Example queries:
    - "Show me Tesla's stock performance over the last 3 months"
    - "Compare Apple and Microsoft stocks for the past year"
    - "Analyze the trading volume of Amazon stock for the last month"

    Args:
        query (str): The query to analyze the stock market data.
    
    Returns:
        str: A nicely formatted python code as a string.
    """
    try:
        result = run_financial_analysis(query)
        return result
    except Exception as e:
        return f"Error: {e}"

@mcp.tool()
def comprehensive_valuation(company: str) -> str:
    """
    Performs comprehensive valuation analysis for a company using multiple methodologies:
    - Discounted Cash Flow (DCF)
    - Payback Time
    - Owner Earnings Yield (10 Cap)
    - Ben Graham's Intrinsic Value Formula
    - P/E and EV/EBITDA Multiples
    - Asset-based valuation
    - Sum of the Parts (SOTP)
    - Dividend Discount Model (DDM)
    - PEG Ratios (P/E, P/S, P/B, P/FCF)
    
    Generates a detailed investment report with Buy/Hold/Sell recommendations.
    
    Args:
        company (str): Company name or ticker symbol (e.g., "NVDA", "Apple", "TSLA")
    
    Returns:
        str: Path to the generated investment report or error message
    """
    try:
        result = estimate_stock_price(f"analyze {company}")
        return result
    except Exception as e:
        return f"Error performing valuation analysis: {e}"

@mcp.tool()
def save_code(code: str, filename: str = "stock_analysis.py") -> str:
    """
    Expects a nicely formatted, working and executable python code as input in form of a string. 
    Save the given code to a file, make sure the code is a valid python file, nicely formatted and ready to execute.

    Args:
        code (str): The nicely formatted, working and executable python code as string.
        filename (str): The filename to save the code to (default: stock_analysis.py)
    
    Returns:
        str: A message indicating the code was saved successfully.
    """
    try:
        with open(filename, 'w') as f:
            f.write(code)
        return f"Code saved to {filename}"
    except Exception as e:
        return f"Error: {e}"

@mcp.tool()
def run_code_and_show_plot() -> str:
    """
    Run the code in stock_analysis.py and generate the plot
    """
    try:
        with open('stock_analysis.py', 'r') as f:
            exec(f.read())
        return "Code executed successfully. Plot should be displayed."
    except Exception as e:
        return f"Error executing code: {e}"

@mcp.tool()
def quick_valuation_metrics(ticker: str) -> str:
    """
    Get quick valuation metrics for a stock without generating a full report.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL", "MSFT", "GOOGL")
    
    Returns:
        str: JSON string with key valuation metrics
    """
    try:
        from valuation_calculator import ValuationCalculator
        
        calculator = ValuationCalculator(ticker)
        metrics = calculator.get_financial_metrics()
        
        # Calculate key metrics
        dcf_value, dcf_signal = calculator.calculate_dcf()
        payback_years, payback_signal = calculator.calculate_payback_time()
        yield_pct, yield_signal = calculator.calculate_owner_earnings_yield()
        graham_value, graham_signal = calculator.calculate_graham_value()
        
        quick_metrics = {
            "ticker": ticker,
            "current_price": metrics.get("current_price", 0),
            "dcf_value": round(dcf_value, 2),
            "dcf_signal": dcf_signal,
            "payback_years": round(payback_years, 1),
            "payback_signal": payback_signal,
            "owner_earnings_yield": f"{round(yield_pct, 2)}%",
            "yield_signal": yield_signal,
            "graham_value": round(graham_value, 2),
            "graham_signal": graham_signal,
            "pe_ratio": round(metrics.get("pe_ratio", 0), 2),
            "price_to_book": round(metrics.get("price_to_book", 0), 2),
            "recommendation": "See full report for detailed analysis"
        }
        
        return json.dumps(quick_metrics, indent=2)
    except Exception as e:
        return f"Error getting quick metrics: {e}"

# Run the server locally
if __name__ == "__main__":
    mcp.run(transport='stdio')