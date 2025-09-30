"""
Prefect deployment script for Stargazing Orchestration
"""

from prefect import flow
from orchestrator import stargazing_orchestration_flow

def main():
    # Build deployment for the orchestration DAG
    deployment = flow.deploy().build_from_flow(
        flow=stargazing_orchestration_flow,
        name="stargazing-orchestration-deployment",
        schedule="0 */6 * * *",  # run every 6 hours
    )

    deployment.apply()
    print("Deployment for stargazing orchestration applied.")

if __name__ == "__main__":
    main()