import geopandas as gpd
import h3pandas


def tile_shapefile(path_to_shp, resolution, subset=None):
    """Breaks a shapefile into hexagonal tiles

    Uses the H3 Global Grid
    
    Parameters
    ----------
    path_to_shp : string
        Path to shapefile
    resolution : int
        The size of the hexagons. Range from 1 (large) to 15 (small)
    subset : list
        Names of the polygons within shapefile that should be included (default is None, all returned)
    """
    
    shp = gpd.read_file(path_to_shp)
    shp = shp.to_crs(4326)
    if subset != None:
        for col in subset:
            if col == "index":
                shp = shp.loc[subset[col]]
            else:
                shp = shp.loc[shp[col].isin(subset[col])]
            if len(shp.index) == 0:
                raise RuntimeError("Subsetting conditions resulted in empty dataframe.")
    shp_h3 = shp.h3.polyfill_resample(resolution).sort_index().reset_index()
    shp_h3["deme"] = shp_h3.index
    
    return shp_h3