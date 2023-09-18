from yaml import load, FullLoader
from munch import Munch


def load_config():
    with open("config/config.yaml") as f:
        return load(f, Loader=FullLoader)


config = Munch.fromDict(load_config())
<<<<<<< HEAD
=======


DATA_DICT = dict(
    train=dict(
        flood=dict(
            Greece2018=dict(
                images="data/train/Sentinel-1/floods/Greece2018/EQUI7_EU020M/E054N006T3",
                reference="data/train/reference/floods/Greece2018",
            ),
            Myanmar2019=dict(
                images="data/train/Sentinel-1/floods/Myanmar2019/EQUI7_AS020M/E045N021T3",
                reference="data/train/reference/floods/Myanmar2019/",
            ),
            Texas2017=dict(
                images="data/train/Sentinel-1/floods/Texas2017",
                reference="data/train/reference/floods/Texas2017",
            ),
        ),
        water=dict(
            E033N009T3=dict(
                images="data/train/Sentinel-1/water/EQUI7_EU020M/E033N009T3",
                reference="data/train/reference/water/EQUI7_EU020M/E033N009T3",
            ),
            E042N012T3=dict(
                images="data/train/Sentinel-1/water/EQUI7_EU020M/E042N012T3",
                reference="data/train/reference/water/EQUI7_EU020M/E042N012T3",
            ),
            E051N015T3=dict(
                images="data/train/Sentinel-1/water/EQUI7_EU020M/E051N015T3",
                reference="data/train/reference/water/EQUI7_EU020M/E051N015T3",
            ),
            E051N027T3=dict(
                images="data/train/Sentinel-1/water/EQUI7_EU020M/E051N027T3",
                reference="data/train/reference/water/EQUI7_EU020M/E051N027T3",
            ),
            E060N021T3=dict(
                images="data/train/Sentinel-1/water/EQUI7_EU020M/E060N021T3",
                reference="data/train/reference/water/EQUI7_EU020M/E060N021T3",
            ),
        ),
    )
)
>>>>>>> master
