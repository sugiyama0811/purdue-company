import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import ScrapeWebsiteTool
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from purdue_company.tools import get_llm, DuckDuckGoSearchTool


@CrewBase
class PassiveIncomeCrew:
    """Passive Income Division — AI-assisted research for legal passive income strategies."""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def opportunity_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["opportunity_researcher"],
            llm=get_llm(),
            tools=[DuckDuckGoSearchTool(), ScrapeWebsiteTool()],
            verbose=True,
            max_iter=10,
        )

    @agent
    def feasibility_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["feasibility_analyst"],
            llm=get_llm(),
            tools=[DuckDuckGoSearchTool()],
            verbose=True,
            max_iter=8,
        )

    @agent
    def income_strategist(self) -> Agent:
        return Agent(
            config=self.agents_config["income_strategist"],
            llm=get_llm(),
            verbose=True,
            max_iter=6,
        )

    @agent
    def income_report_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["income_report_writer"],
            llm=get_llm(),
            verbose=True,
            max_iter=4,
        )

    @task
    def research_opportunities_task(self) -> Task:
        return Task(config=self.tasks_config["research_opportunities_task"])

    @task
    def analyze_feasibility_task(self) -> Task:
        return Task(config=self.tasks_config["analyze_feasibility_task"])

    @task
    def build_strategy_task(self) -> Task:
        return Task(config=self.tasks_config["build_strategy_task"])

    @task
    def write_income_report_task(self) -> Task:
        return Task(config=self.tasks_config["write_income_report_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_llm=get_llm(),
            verbose=True,
            output_log_file="output/passive_income_crew.log",
        )
