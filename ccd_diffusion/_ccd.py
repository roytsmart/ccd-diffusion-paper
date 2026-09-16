import astropy.units as u
import optika

__all__ = [
    "ccd",
    "ccd_aia",
]


def ccd() -> optika.sensors.materials.BackIlluminatedSiliconSensorMaterial:
    return optika.sensors.materials.e2v_ccd97(temperature=190 * u.K)


def ccd_aia() -> optika.sensors.materials.BackIlluminatedSiliconSensorMaterial:
    return optika.sensors.materials.e2v_ccd203()
