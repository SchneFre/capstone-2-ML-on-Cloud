from sagemaker.processing import ScriptProcessor
from sagemaker import get_execution_role

role = get_execution_role()

processor = ScriptProcessor(
    image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    command=["python3"],
    role=role,
    instance_count=1,
    instance_type="ml.m5.large"
)

processor.run(
    code="s3://s3-gold-price-fjs/scripts/pipeline_runner.py"
)