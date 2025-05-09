import logging

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class TokkioAiqFunctionConfig(FunctionBaseConfig, name="tokkio_aiq"):
    """
    AIQ Toolkit function template. Please update the description.
    """
    # Add your custom configuration parameters here
    parameter: str = Field(default="default_value", description="Notional description for this parameter")


@register_function(config_type=TokkioAiqFunctionConfig)
async def tokkio_aiq_function(
    config: TokkioAiqFunctionConfig, builder: Builder
):
    # Implement your function logic here
    async def _response_fn(input_message: str) -> str:
        # Process the input_message and generate output
        output_message = f"Hello from tokkio_aiq workflow! You said: {input_message}"
        return output_message

    try:
        yield FunctionInfo.create(single_fn=_response_fn)
    except GeneratorExit:
        print("Function exited early!")
    finally:
        print("Cleaning up tokkio_aiq workflow.")