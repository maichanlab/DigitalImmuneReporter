import pandas as pd


def count_cell_in_tissue(cell_table: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of cells of each cell type within each tissue type.

    Args:
        cell_table: DataFrame with columns 'x', 'y', 'cell_label', 'tissue_label'.

    Returns:
        pd.DataFrame with columns 'cell_label', 'tissue_label', 'count'.
    """
    if {'x', 'y', 'cell_label', 'tissue_label'} - set(cell_table.columns):
        raise ValueError("cell_table must contain 'x', 'y', 'cell_label', and 'tissue_label' columns.")
    cell_count = cell_table.groupby(['cell_label', 'tissue_label']).size().reset_index(name="count")
    return cell_count


def count_cell_density(cell_count: pd.DataFrame, tissue_area: dict) -> pd.DataFrame:
    """
    Compute the density of each cell type within each tissue type.

    Args:
        cell_count: DataFrame with columns 'cell_label', 'tissue_label', 'count'
            (typically produced by `count_cell_in_tissue`).
        tissue_area: Mapping from tissue type to its area, e.g. {'tumor': 15342.0, ...}.

    Returns:
        pd.DataFrame with columns 'cell_label', 'tissue_label', 'count', 'tissue_area',
        'density' (count / tissue_area), and 'density_label'.
    """
    if set(cell_count.columns) - {'count', 'cell_label', 'tissue_label'}:
        raise ValueError("cell_count must contain 'count', 'cell_label' and 'tissue_label' columns.")
    if set(cell_count['tissue_label'].unique()) - set(tissue_area.keys()):
        raise ValueError("tissue_area must contain all unique tissue labels in cell_count.")
    cell_count["tissue_area"] = cell_count["tissue_label"].map(tissue_area)
    cell_count["density"] = cell_count["count"] / cell_count["tissue_area"]
    cell_count["density_label"] = cell_count[['cell_label', 'tissue_label']].apply(
        lambda x: '{}_density_in_{}_region'.format(x.iloc[0], x.iloc[1]), axis=1
    )
    return cell_count
