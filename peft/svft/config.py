from dataclasses import dataclass, field

from typing import Optional, Union

from peft import PeftConfig
from peft.utils import PeftType


@dataclass
class SVFTConfig(PeftConfig):
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with SVFT. "
                "For example, '.*decoder.*' or '.*encoder.*'. If not specified, modules will be chosen according to "
                "the model architecture. If the architecture is not known, an error will be raised; in this case, "
                "please specify target modules manually."
            ),
        },
    )
    rank: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of singular vectors retained during adaptation."},
    )
    off_diag: int = field(
        default=1,
        metadata={"help": "Number of off-diagonals that remain trainable when using banded sparsity."},
    )
    pattern: str = field(
        default="banded",
        metadata={"help": "Sparse pattern for singular-value corrections: banded | random | top_k."},
    )
    fill_orthonormal: bool = field(
        default=False,
        metadata={"help": "Extend truncated singular spaces with random orthonormal bases when rank is limited."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.SVFT
        self.pattern = (self.pattern or "banded").lower()