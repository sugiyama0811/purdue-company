import os
from pydantic import BaseModel
from crewai.flow.flow import Flow, start, listen, router
from crewai import LLM

from purdue_company.me.crew import MECrew
from purdue_company.trading.crew import TradingCrew
from purdue_company.passive_income.crew import PassiveIncomeCrew


class CompanyState(BaseModel):
    # Input
    user_request: str = ""
    division: str = ""  # "me" | "trading" | "passive" | "unknown"

    # ME inputs/outputs
    me_query: str = ""
    me_result: str = ""

    # Trading inputs/outputs
    ticker: str = ""
    trading_result: str = ""

    # Passive income inputs/outputs
    income_query: str = ""
    income_result: str = ""

    # Final
    final_output: str = ""
    output_file: str = ""


class CompanyFlow(Flow[CompanyState]):
    """
    Top-level orchestrator for the three-division company:
      1. ME Academic Division
      2. Financial Trading Division
      3. Passive Income Division
    """

    def _llm(self) -> LLM:
        return LLM(
            model=os.getenv("MODEL", "anthropic/claude-sonnet-4-6"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=1024,
        )

    @start()
    def classify_request(self):
        """Classify the user request and route to the correct division."""
        llm = self._llm()

        prompt = f"""You are a request router for a company with three divisions:

1. ME Academic Division — handles all Mechanical Engineering topics:
   coursework, homework help, concept explanations, research papers, project
   guidance, simulations, design problems, thermodynamics, fluid mechanics,
   heat transfer, solid mechanics, materials, controls, robotics, FEA, CAD,
   manufacturing, MATLAB, ANSYS, SolidWorks, etc.

2. Financial Trading Division — handles day trading analysis for stocks and
   futures. Stock tickers: AAPL, NVDA, TSLA, etc. Futures: ES=F (S&P 500),
   NQ=F (Nasdaq), YM=F (Dow), MES=F, MNQ=F (Micro futures).

3. Passive Income Division — handles research on legal ways to make money
   passively using AI and technology (dividends, REITs, digital products, SaaS,
   content, algorithms) without daily active involvement.

User request: "{self.state.user_request}"

Respond with EXACTLY one of these formats (no other text):
ME: <the exact question or topic to research>
TRADING: <ticker symbol in correct yfinance format, e.g. AAPL or ES=F or NQ=F>
PASSIVE: <the specific passive income focus extracted from the request>
UNKNOWN: <reason why this doesn't fit any division>

Examples:
- "help me understand Navier-Stokes" → ME: Navier-Stokes equations in fluid mechanics
- "explain heat exchanger design" → ME: heat exchanger design for ME coursework
- "analyze ES futures" → TRADING: ES=F
- "NQ day trading setup" → TRADING: NQ=F
- "what's the setup on NVDA" → TRADING: NVDA
- "find me passive income ideas" → PASSIVE: general passive income strategies with AI
- "how to make money without working" → PASSIVE: passive income streams for students
- "dividend stocks for students" → PASSIVE: dividend stock investing for students
"""
        response = llm.call(prompt).strip()

        if response.startswith("ME:"):
            self.state.division = "me"
            self.state.me_query = response[3:].strip()
        elif response.startswith("TRADING:"):
            self.state.division = "trading"
            self.state.ticker = response[8:].strip().upper()
        elif response.startswith("PASSIVE:"):
            self.state.division = "passive"
            self.state.income_query = response[8:].strip()
        else:
            self.state.division = "unknown"

        return self.state.division

    @router(classify_request)
    def route_to_division(self, division: str):
        if division == "me":
            return "run_me"
        elif division == "trading":
            return "run_trading"
        elif division == "passive":
            return "run_passive"
        else:
            return "handle_unknown"

    @listen("run_me")
    def run_me_division(self):
        """Execute the ME Academic crew."""
        print(f"\n[ME Academic Division] Query: {self.state.me_query}\n")
        os.makedirs("output", exist_ok=True)

        result = MECrew().crew().kickoff(inputs={"me_query": self.state.me_query})
        self.state.me_result = str(result)
        self.state.final_output = self.state.me_result
        self.state.output_file = "output/me_report.md"
        return self.state.final_output

    @listen("run_trading")
    def run_trading_division(self):
        """Execute the Financial Trading crew."""
        print(f"\n[Financial Trading Division] Analyzing: {self.state.ticker}\n")
        os.makedirs("output", exist_ok=True)

        result = TradingCrew().crew().kickoff(inputs={"ticker": self.state.ticker})
        self.state.trading_result = str(result)
        self.state.final_output = self.state.trading_result
        self.state.output_file = "output/trading_report.md"
        return self.state.final_output

    @listen("run_passive")
    def run_passive_income_division(self):
        """Execute the Passive Income crew."""
        print(f"\n[Passive Income Division] Researching: {self.state.income_query}\n")
        os.makedirs("output", exist_ok=True)

        result = PassiveIncomeCrew().crew().kickoff(
            inputs={"income_query": self.state.income_query}
        )
        self.state.income_result = str(result)
        self.state.final_output = self.state.income_result
        self.state.output_file = "output/passive_income_report.md"
        return self.state.final_output

    @listen("handle_unknown")
    def handle_unknown_request(self):
        """Handle requests that don't match any division."""
        msg = (
            f"Could not route: '{self.state.user_request}'\n\n"
            "This company handles:\n\n"
            "  ME Academic Division:\n"
            "    Mechanical engineering coursework, research, design, simulations\n"
            "    Examples: 'explain Bernoulli equation'\n"
            "              'help with heat exchanger design'\n"
            "              'research papers on topology optimization'\n\n"
            "  Financial Trading Division:\n"
            "    Day trading analysis for stocks and index futures\n"
            "    Examples: 'analyze NVDA'\n"
            "              'ES futures day trading setup'\n"
            "              'NQ=F technical analysis'\n\n"
            "  Passive Income Division:\n"
            "    Legal passive income strategies using AI and technology\n"
            "    Examples: 'find passive income ideas for students'\n"
            "              'best dividend ETFs for F-1 visa holders'\n"
            "              'how to make money with AI tools'\n"
        )
        self.state.final_output = msg
        return msg
