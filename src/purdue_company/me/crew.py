import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import ScrapeWebsiteTool
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from purdue_company.tools import get_llm, DuckDuckGoSearchTool


@CrewBase
class MECrew:
    """Mechanical Engineering Academic Division — coursework, research, and project help."""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def subject_expert(self) -> Agent:
        return Agent(
            config=self.agents_config["subject_expert"],
            llm=get_llm(),
            tools=[DuckDuckGoSearchTool(), ScrapeWebsiteTool()],
            verbose=True,
            max_iter=10,
        )

    @agent
    def problem_solver(self) -> Agent:
        return Agent(
            config=self.agents_config["problem_solver"],
            llm=get_llm(),
            tools=[DuckDuckGoSearchTool(), ScrapeWebsiteTool()],
            verbose=True,
            max_iter=8,
        )

    @agent
    def academic_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["academic_writer"],
            llm=get_llm(),
            verbose=True,
            max_iter=5,
        )

    @task
    def research_topic_task(self) -> Task:
        return Task(config=self.tasks_config["research_topic_task"])

    @task
    def solve_or_explain_task(self) -> Task:
        return Task(config=self.tasks_config["solve_or_explain_task"])

    @task
    def write_me_report_task(self) -> Task:
        return Task(config=self.tasks_config["write_me_report_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_llm=get_llm(),
            verbose=True,
            output_log_file="output/me_crew.log",
        )
