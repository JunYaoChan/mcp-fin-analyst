# MCP Financial Analyst 

A comprehensive financial analysis tool powered by CrewAI that provides intelligent stock analysis, valuation modeling, and investment research through a Model Context Protocol (MCP) server.

## 🚀 Features

### Stock Analysis & Visualization
- **Intelligent Query Processing**: Natural language queries for stock analysis
- **Automated Code Generation**: Generates executable Python scripts for data visualization
- **Multi-Stock Comparison**: Compare performance across multiple securities
- **Technical Analysis**: Volume analysis, price trends, and market metrics

### Comprehensive Valuation Engine
- **Discounted Cash Flow (DCF)** - Intrinsic value based on projected cash flows
- **Payback Time Analysis** - Investment recovery period calculation
- **Owner Earnings Yield** - Warren Buffett's 10-cap methodology
- **Ben Graham Formula** - Classic value investing approach with current bond yields
- **Multiple Analysis** - P/E, EV/EBITDA, Price-to-Book ratios
- **Asset-Based Valuation** - Book value and tangible asset analysis
- **Sum of the Parts (SOTP)** - Segment-based valuation for conglomerates
- **Dividend Discount Model (DDM)** - For dividend-paying stocks
- **PEG Ratios** - Growth-adjusted valuation metrics

### Investment Reports
- **Professional Grade Reports** - Institutional-quality investment analysis
- **Buy/Hold/Sell Signals** - Clear recommendations based on multiple methodologies
- **Risk Assessment** - Comprehensive risk analysis and considerations
- **Data Transparency** - Full source attribution and data vintage tracking

## 📋 Requirements

- **Python**: 3.11 or higher
- **API Keys**: DeepSeek API key (required), Serper API key (optional for web search)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fin-analyst
   ```

2. **Install dependencies**
   ```bash
   pip install -e .
   ```

3. **Set up environment variables**
   Create an `env` file or set environment variables:
   ```bash
   DEEPSEEK_API_KEY="your-deepseek-api-key"
   SERPER_API_KEY="your-serper-api-key"  # Optional
   ```

4. **Run the MCP server**
   ```bash
   python main.py
   ```

## 🎯 Usage

### As MCP Server 

The tool runs as an MCP server that can be integrated with Claude or other AI assistants:

```bash
python main.py
```

### Available MCP Tools

#### 1. **analyze_stock(query: str)**
Generates executable Python code for stock analysis and visualization.

**Example queries:**
- "Show me Tesla's stock performance over the last 3 months"
- "Compare Apple and Microsoft stocks for the past year"
- "Analyze the trading volume of Amazon stock for the last month"

#### 2. **comprehensive_valuation(company: str)**
Performs complete valuation analysis using multiple methodologies.

**Example:**
```python
comprehensive_valuation("NVDA")
```

**Output:** Detailed investment report with Buy/Hold/Sell recommendation

#### 3. **quick_valuation_metrics(ticker: str)**
Returns key valuation metrics in JSON format without generating a full report.

#### 4. **save_code(code: str, filename: str)**
Saves generated Python code to a file for execution.

#### 5. **run_code_and_show_plot()**
Executes saved stock analysis code and displays visualizations.

### Direct Python Usage

```python
from finance_crew import run_financial_analysis, estimate_stock_price
from valuation_crew import run_valuation_analysis

# Generate stock analysis code
analysis_code = run_financial_analysis("Plot TSLA stock for 6 months")

# Run comprehensive valuation
valuation_report = run_valuation_analysis("AAPL")
```

## 🏗️ Architecture

### Core Components

1. **CrewAI Agents**
   - **Query Parser Agent**: Extracts stock symbols, timeframes, and actions
   - **Code Writer Agent**: Generates Python visualization code
   - **Code Execution Agent**: Reviews and executes generated code
   - **Financial Data Agent**: Retrieves comprehensive financial data
   - **Valuation Expert Agent**: Performs valuation calculations
   - **Report Writer Agent**: Creates professional investment reports

2. **Valuation Engine**
   - Multi-methodology approach with 9+ valuation models
   - Real-time data integration via yfinance
   - Current market conditions awareness
   - Signal aggregation and consensus building

3. **MCP Server Integration**
   - FastMCP framework for seamless AI assistant integration
   - Structured tool definitions with comprehensive documentation
   - Error handling and robust execution

### Data Flow

```
User Query → Query Parser → Data Collection → Valuation Analysis → Report Generation
     ↓
Stock Visualization → Code Generation → Code Execution → Plot Display
```

## 📊 Valuation Methodologies

### Signal Classification
- **BUY**: >20% undervalued relative to calculated fair value
- **HOLD**: -20% to +20% of fair value range
- **SELL**: >20% overvalued relative to fair value
- **N/A**: Insufficient data or methodology not applicable

### Final Recommendation Logic
The system aggregates signals from all applicable methodologies and provides a consensus recommendation based on the majority vote, with detailed reasoning for each component.

## 🔧 Configuration

### LLM Configuration
The system uses DeepSeek's chat model via CrewAI:

```python
llm = LLM(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)
```

### Data Sources
- **Primary**: Yahoo Finance (yfinance)
- **Supplementary**: Web search for recent analyst estimates and news
- **Market Data**: Real-time pricing and financial metrics

## 📈 Output Examples

### Stock Analysis Output
```python
# Generated code example
import yfinance as yf
import matplotlib.pyplot as plt

# Fetch Tesla data for 3 months
tsla = yf.Ticker("TSLA")
data = tsla.history(period="3mo")

# Create visualization
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'])
plt.title("Tesla Stock Performance - Last 3 Months")
plt.show()
```

### Valuation Report Structure
```markdown
# NVDA Investment Analysis Report
**Date**: July 31, 2025
**Current Price**: $XXX.XX

## Investment Decision Matrix
| Method | Fair Value | Signal | Reasoning |
|--------|------------|---------|-----------|
| DCF | $XXX | BUY | Strong FCF growth projected |
| Graham | $XXX | HOLD | Moderate undervaluation |
...

## Final Recommendation: BUY
**Vote Tally**: 6 BUY, 2 HOLD, 1 SELL

**Target Price Range**: $XXX - $XXX
```

## 🛡️ Error Handling

The system includes robust error handling for:
- Invalid stock symbols
- Data retrieval failures  
- API rate limiting
- Calculation edge cases
- File I/O operations

**Built with ❤️ using CrewAI, Python, and financial data APIs**
