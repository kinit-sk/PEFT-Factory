from dataclasses import dataclass, field

from typing import Optional, Union

from peft import PeftConfig
from peft.utils import PeftType


@dataclass
class BitFitConfig(PeftConfig):
    target_modules: Optional[Union[list[str], str]] = field(default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with BitFit."
                "For example, '.*decoder.*' or '.*encoder.*'. "
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you shoud specify the target modules manually."
            ),
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.BITFIT