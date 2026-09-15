"""DEM discovery, validation, caching, lookup, gradients, and masks."""

from __future__ import annotations

import csv
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ...geometry.pds import PolarStereographic, parse_pds3_label, normalize_longitude, unwrap_longitude


@dataclass(frozen=True)
class DEMQuery:
    elevation_m: float | None
    valid: bool
    quality: str
    reason: str
    provider: str


@dataclass(frozen=True)
class DEMGrid:
    """A resampled geographic DEM query with an explicit validity mask."""

    latitudes: np.ndarray
    longitudes: np.ndarray
    elevations_m: np.ndarray
    valid_mask: np.ndarray
    quality: np.ndarray


@dataclass(frozen=True)
class DEMValidation:
    provider: str
    checked: int
    valid: int
    invalid: int
    gradients_available: int
    status: str


class DEMCache:
    """Small local provider cache; it never downloads or fabricates a DEM."""

    def __init__(self, providers: Iterable["DEMProvider"] = ()) -> None:
        self.providers = list(providers)

    @classmethod
    def from_project(cls, root: str | Path) -> "DEMCache":
        return cls(discover_dem_providers(root))

    def get_dem_for_region(self, latitude_range: tuple[float, float], longitude_range: tuple[float, float]) -> list["DEMProvider"]:
        return [provider.get_dem_for_region(latitude_range, longitude_range) for provider in self.providers if provider.intersects(latitude_range, longitude_range)]

    def first_for_region(self, latitude_range: tuple[float, float], longitude_range: tuple[float, float]) -> "DEMProvider":
        matches = self.get_dem_for_region(latitude_range, longitude_range)
        return matches[0] if matches else UnavailableDEMProvider("out_of_coverage")


class DEMProvider:
    """Interface for real DEM providers.

    Implementations return invalid queries rather than silently filling a
    missing or out-of-coverage elevation.
    """

    name = "abstract"

    def get_dem_for_region(self, latitude_range: tuple[float, float], longitude_range: tuple[float, float]) -> "DEMProvider":
        return self if self.intersects(latitude_range, longitude_range) else UnavailableDEMProvider("out_of_coverage")

    def get_elevation(self, latitude: float, longitude: float) -> DEMQuery:
        raise NotImplementedError

    def get_elevation_batch(self, latitudes: Sequence[float], longitudes: Sequence[float]) -> list[DEMQuery]:
        return [self.get_elevation(lat, lon) for lat, lon in zip(latitudes, longitudes)]

    def get_gradient(self, latitude: float, longitude: float) -> tuple[float, float] | None:
        step = 0.0001
        a = self.get_elevation(latitude - step, longitude)
        b = self.get_elevation(latitude + step, longitude)
        c = self.get_elevation(latitude, longitude - step)
        d = self.get_elevation(latitude, longitude + step)
        if not all(query.valid and query.elevation_m is not None for query in (a, b, c, d)):
            return None
        lat_scale = 1737400.0 * math.pi / 180.0
        lon_scale = lat_scale * math.cos(math.radians(latitude))
        return ((b.elevation_m - a.elevation_m) / (2.0 * step * lat_scale), (d.elevation_m - c.elevation_m) / (2.0 * step * lon_scale))

    def is_valid(self, latitude: float, longitude: float) -> bool:
        return self.get_elevation(latitude, longitude).valid

    def get_quality(self, latitude: float, longitude: float) -> str:
        return self.get_elevation(latitude, longitude).quality

    def intersects(self, latitude_range: tuple[float, float], longitude_range: tuple[float, float]) -> bool:
        return True


def resample_dem(
    provider: DEMProvider,
    latitudes: Sequence[float] | np.ndarray,
    longitudes: Sequence[float] | np.ndarray,
) -> DEMGrid:
    """Query a provider on a geographic grid without filling invalid cells.

    The result is deliberately regular and provider-agnostic.  Consumers must
    use ``valid_mask`` before using an elevation or gradient derived from it.
    """

    lat_grid, lon_grid = np.broadcast_arrays(np.asarray(latitudes, dtype=np.float64), np.asarray(longitudes, dtype=np.float64))
    elevations = np.full(lat_grid.shape, np.nan, dtype=np.float64)
    valid = np.zeros(lat_grid.shape, dtype=bool)
    quality = np.empty(lat_grid.shape, dtype=object)
    quality.fill("unavailable")
    for index in np.ndindex(lat_grid.shape):
        query = provider.get_elevation(float(lat_grid[index]), float(lon_grid[index]))
        quality[index] = query.quality
        if query.valid and query.elevation_m is not None and math.isfinite(query.elevation_m):
            elevations[index] = float(query.elevation_m)
            valid[index] = True
    return DEMGrid(lat_grid, lon_grid, elevations, valid, quality)


def validate_dem_provider(
    provider: DEMProvider,
    points: Iterable[tuple[float, float]],
) -> DEMValidation:
    checked = valid = gradients = 0
    for latitude, longitude in points:
        checked += 1
        query = provider.get_elevation(latitude, longitude)
        if query.valid and query.elevation_m is not None and math.isfinite(query.elevation_m):
            valid += 1
        if provider.get_gradient(latitude, longitude) is not None:
            gradients += 1
    return DEMValidation(
        provider=provider.name,
        checked=checked,
        valid=valid,
        invalid=checked - valid,
        gradients_available=gradients,
        status="verified" if checked > 0 and valid == checked else ("partial" if valid else "unavailable"),
    )


class UnavailableDEMProvider(DEMProvider):
    name = "unavailable"

    def __init__(self, reason: str = "no_dem_available") -> None:
        self.reason = reason

    def get_elevation(self, latitude: float, longitude: float) -> DEMQuery:
        return DEMQuery(None, False, "unavailable", self.reason, self.name)

    def get_gradient(self, latitude: float, longitude: float) -> tuple[float, float] | None:
        return None


class KaguyaDTMProvider(DEMProvider):
    name = "SELENE_Kaguya_TCOrtho_DTM_v3.0"

    def __init__(self, data_path: str | Path, label_path: str | Path) -> None:
        self.data_path = Path(data_path)
        self.label_path = Path(label_path)
        self.label = parse_pds3_label(self.label_path)
        self.projection = PolarStereographic.from_label(self.label)
        self.dummy = int(float(self.label["DUMMY"]))
        self.valid_minimum = int(float(self.label["VALID_MINIMUM"]))
        self.valid_maximum = int(float(self.label["VALID_MAXIMUM"]))
        expected = self.projection.lines * self.projection.samples * 2
        if self.data_path.stat().st_size != expected:
            raise ValueError(f"DTM size does not match label: {self.data_path}")
        self.array = np.memmap(self.data_path, dtype=">i2", mode="r", shape=(self.projection.lines, self.projection.samples))

    def _pixel(self, latitude: float, longitude: float) -> tuple[float, float] | None:
        try:
            x, y = self.projection.latlon_to_pixel(latitude, longitude)
        except ValueError:
            return None
        return (x, y) if self.projection.contains_pixel(x, y) else None

    def _nearest(self, x: float, y: float) -> DEMQuery:
        ix, iy = int(round(x)), int(round(y))
        if not self.projection.contains_pixel(ix, iy):
            return DEMQuery(None, False, "out_of_coverage", "pixel_out_of_bounds", self.name)
        value = int(self.array[iy, ix])
        valid = self.valid_minimum <= value <= self.valid_maximum and value != self.dummy
        return DEMQuery(float(value) if valid else None, valid, "label_valid" if valid else "nodata", "valid_dtm_value" if valid else "pds_dummy_or_invalid", self.name)

    def get_elevation(self, latitude: float, longitude: float) -> DEMQuery:
        pixel = self._pixel(latitude, longitude)
        if pixel is None:
            return DEMQuery(None, False, "out_of_coverage", "coordinate_out_of_scene", self.name)
        return self._nearest(*pixel)

    def get_gradient(self, latitude: float, longitude: float) -> tuple[float, float] | None:
        pixel = self._pixel(latitude, longitude)
        if pixel is None:
            return None
        x, y = pixel
        ix, iy = int(round(x)), int(round(y))
        if ix < 1 or ix >= self.projection.samples - 1 or iy < 1 or iy >= self.projection.lines - 1:
            return None
        values = self.array[iy - 1 : iy + 2, ix - 1 : ix + 2].astype(np.float64)
        valid = (values >= self.valid_minimum) & (values <= self.valid_maximum) & (values != self.dummy)
        if not bool(valid.all()):
            return None
        # Kaguya map pixels are 10 m in both projected axes. The returned
        # order is (dElevation/dNorth, dElevation/dEast), in m/m.
        return (float((values[2, 1] - values[0, 1]) / 20.0), float((values[1, 2] - values[1, 0]) / 20.0))

    def intersects(self, latitude_range: tuple[float, float], longitude_range: tuple[float, float]) -> bool:
        latitudes = [self.projection.pixel_to_latlon(x, y)[0] for x, y in ((0, 0), (self.projection.samples - 1, 0), (0, self.projection.lines - 1), (self.projection.samples - 1, self.projection.lines - 1))]
        return max(latitudes) >= min(latitude_range) and min(latitudes) <= max(latitude_range)


class GeoTiffDEMProvider(DEMProvider):
    """Reader for the existing small official LROC WMS GeoTIFF extracts.

    Pillow is used because the project intentionally does not require GDAL.
    The extracts contain absolute lunar radius in metres, so the provider
    converts values to elevation using the Kaguya mean reference radius.
    """

    name = "LROC_GLD100_WMS_numeric"

    def __init__(self, path: str | Path, reference_radius_m: float = 1_737_400.0) -> None:
        from PIL import Image

        self.path = Path(path)
        self.image = Image.open(self.path)
        self.array = np.asarray(self.image, dtype=np.float64)
        self.tags = self.image.tag_v2
        tie = tuple(float(v) for v in self.tags[33922])
        scale = tuple(float(v) for v in self.tags[33550])
        self.origin_lon = tie[3]
        self.origin_lat = tie[4]
        self.lon_step = scale[0]
        self.lat_step = -abs(scale[1])
        self.reference_radius_m = float(reference_radius_m)
        self.nodata = None

    def _pixel(self, latitude: float, longitude: float) -> tuple[float, float]:
        x = (unwrap_longitude(longitude, self.origin_lon) - self.origin_lon) / self.lon_step
        y = (float(latitude) - self.origin_lat) / self.lat_step
        return x, y

    def get_elevation(self, latitude: float, longitude: float) -> DEMQuery:
        x, y = self._pixel(latitude, longitude)
        if x < 0 or y < 0 or x >= self.array.shape[1] or y >= self.array.shape[0]:
            return DEMQuery(None, False, "out_of_coverage", "coordinate_out_of_extract", self.name)
        value = float(self.array[int(round(y)), int(round(x))])
        valid = math.isfinite(value) and value > 0.0
        elevation = value - self.reference_radius_m if valid else None
        return DEMQuery(elevation, valid, "finite_absolute_radius" if valid else "nodata", "converted_absolute_radius" if valid else "nonfinite_or_nonpositive", self.name)

    def get_gradient(self, latitude: float, longitude: float) -> tuple[float, float] | None:
        x, y = self._pixel(latitude, longitude)
        ix, iy = int(round(x)), int(round(y))
        if ix < 1 or iy < 1 or ix >= self.array.shape[1] - 1 or iy >= self.array.shape[0] - 1:
            return None
        values = self.array[iy - 1 : iy + 2, ix - 1 : ix + 2]
        if not np.isfinite(values).all():
            return None
        lat_m = 1737400.0 * math.pi / 180.0 * abs(self.lat_step)
        lon_m = 1737400.0 * math.pi / 180.0 * math.cos(math.radians(latitude)) * self.lon_step
        return (float((values[2, 1] - values[0, 1]) / (2.0 * lat_m)), float((values[1, 2] - values[1, 0]) / (2.0 * lon_m)))


def discover_dem_providers(root: str | Path) -> list[DEMProvider]:
    root = Path(root)
    providers: list[DEMProvider] = []
    with (root / "data" / "kaguya_manifest.csv").open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            label_path = root / row["dtm_label_path"]
            data_path = root / row["dtm_path"]
            if data_path.exists() and label_path.exists():
                providers.append(KaguyaDTMProvider(data_path, label_path))
    for path in sorted((root / "data" / "dem").glob("*.tif")):
        try:
            providers.append(GeoTiffDEMProvider(path))
        except Exception:
            continue
    return providers
