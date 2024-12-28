from logging import Handler
from src.utils import is_rank_zero
class RankZeroHandlerMixin(Handler):
    def emit(self, record):
        if is_rank_zero():
            super().emit(record)