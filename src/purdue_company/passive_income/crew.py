import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List


@CrewBase
class PassiveIncomeCrew:
    """Passive Income Division — AI-assisted research for legal passive income strategies."""

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
    def opportunity_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["opportunity_researcher"],
            llm=self._llm(),
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            verbose=True,
            max_iter=10,
        )

    @agent
    def feasibility_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["feasibility_analyst"],
            llm=self._llm(),
            tools=[SerperDevTool()],
            verbose=True,
            max_iter=8,
        )

    @agent
    def income_strategist(self) -> Agent:
        return Agent(
            config=self.agents_config["income_strategist"],
            llm=self._llm(),
            verbose=True,
            max_iter=6,
        )

    @agent
    def income_report_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["income_report_writer"],
            llm=self._llm(),
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
            manager_llm=self._llm(),
            verbose=True,
            output_log_file="output/passive_income_crew.log",
        )
