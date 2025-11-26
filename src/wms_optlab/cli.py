import click
import pandas as pd
from .data.adapters_erp import load_generic_erp_data
from .slotting.heuristics import assign_by_abc_popularity
from .layout.geometry import manhattan_distance
from .integration.wms_export_templates import generate_wms_slotting_import_file

@click.group()
def main():
    """WMS-OptLab: A toolkit for warehouse optimization."""
    pass

@main.command()
@click.option('--skus', 'skus_path', required=True, type=click.Path(exists=True), help='Path to SKU master CSV.')
@click.option('--locations', 'locations_path', required=True, type=click.Path(exists=True), help='Path to locations master CSV.')
@click.option('--orders', 'orders_path', required=True, type=click.Path(exists=True), help='Path to orders CSV.')
@click.option('--output', 'output_path', required=True, type=click.Path(), help='Path to save the optimized slotting plan CSV.')
def optimize_slotting(skus_path, locations_path, orders_path, output_path):
    """
    Performs slotting optimization using a popularity-based heuristic.
    """
    click.echo("Loading data...")
    skus_df = pd.read_csv(skus_path)
    locations_df = pd.read_csv(locations_path)
    orders_df = pd.read_csv(orders_path)

    warehouse = load_generic_erp_data(skus_df, locations_df, orders_df)
    click.echo(f"Loaded {len(warehouse.skus)} SKUs, {len(warehouse.locations)} locations, and {len(warehouse.orders)} orders.")

    click.echo("Running ABC popularity-based slotting...")
    # Assuming depot is at (0,0,0) for simplicity in the CLI
    depot_location = (0, 0, 0)
    
    slotting_plan = assign_by_abc_popularity(
        warehouse.skus,
        warehouse.locations,
        warehouse.orders,
        distance_metric=manhattan_distance,
        depot_location=depot_location
    )

    click.echo(f"Generated slotting plan for {len(slotting_plan)} SKUs.")

    # Save the result to a CSV file suitable for WMS import
    output_df = generate_wms_slotting_import_file(slotting_plan)
    output_df.to_csv(output_path, index=False)
    click.echo(f"Slotting plan saved to {output_path}")

if __name__ == '__main__':
    main()
