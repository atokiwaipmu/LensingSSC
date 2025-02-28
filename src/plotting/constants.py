from enum import Enum, auto

class PlotType(Enum):
    ELL = auto()
    NU = auto()

class PlotConstants:
    FIGURE_WIDTH = 14
    FIGURE_HEIGHT_MULTIPLIER = 3
    TITLE_FONTSIZE = 16
    LEGEND_FONTSIZE = 12
    LEGEND_NCOL_OFFSET = 3

    MARKERS = {
        'TILED': '^',
        'BIGBOX': 'o'
    }

    LINESTYLES = {
        'TILED': '--',
        'BIGBOX': '-',
        'THEORY': ':'
    }