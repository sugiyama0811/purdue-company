import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from purdue_company.trading.tools import StockDataTool


@CrewBase
class TradingCrew:
    """Financial Trading Division — day trading analysis for stocks and futures."""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def _llm(self) -> LLM:
        return LLM(
            model=os.getenv("MODEL", "anthropic/claude-sonnet-4-6"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=8096,
        )

    @agent
    def market_data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["market_data_analyst"],
            llm=self._llm(),
            tools=[StockDataTool()],
            verbose=True,
            max_iter=6,
        )

    @agent
    def pattern_strategist(self) -> Agent:
        return Agent(
            config=self.agents_config["pattern_strategist"],
            llm=self._llm(),
            tools=[StockDataTool()],
            verbose=True,
            max_iter=6,
        )

    @agent
    def risk_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["risk_manager"],
            llm=self._llm(),
            tools=[SerperDevTool()],
            verbose=True,
            max_iter=5,
        )

    @agent
    def trading_report_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["trading_report_writer"],
            llm=self._llm(),
            verbose=True,
            max_iter=4,
        )

    @task
    def fetch_and_analyze_task(self) -> Task:
        return Task(config=self.tasks_config["fetch_and_analyze_task"])

    @task
    def identify_setups_task(self) -> Task:
        return Task(config=self.tasks_config["identify_setups_task"])

    @task
    def risk_assessment_task(self) -> Task:
        return Task(config=self.tasks_config["risk_assessment_task"])

    @task
    def write_trading_report_task(self) -> Task:
        return Task(config=self.tasks_config["write_trading_report_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_llm=self._llm(),
            verbose=True,
            output_log_file="output/trading_crew.log",
        )
