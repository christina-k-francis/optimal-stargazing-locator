"""
Prefect deployment script for Stargazing Orchestration
"""

from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from orchestration import stargazing_orchestration_flow


def main():
    # Build deployment for the orchestration DAG
    deployment = Deployment.build_from_flow(
        flow=stargazing_orchestration_flow,
        name="stargazing-orchestration-deployment",
        schedule=CronSchedule(cron="8 */12 * * *"),  # run every 12 hours
        work_pool_name="default-agent-pool",
        tags=["stargazing", "production", "nws-data"],
        description="Main orchestration flow for processing NWS data and calculating stargazing grades",
    )

    deployment.apply()
    print("✅ Deployment for stargazing orchestration applied successfully!")

if __name__ == "__main__":
    main()