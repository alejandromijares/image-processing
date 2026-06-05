import asyncio
from viam.module.module import Module
from models.color_correction import ColorCorrection as ColorCorrectionModel


if __name__ == '__main__':
    asyncio.run(Module.run_from_registry())
