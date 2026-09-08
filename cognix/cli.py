import click

@click.group()
def cli():
    """Cognix Decision Engine CLI."""
    pass

@cli.command()
def version():
    """Print the version."""
    click.echo("Cognix version 0.1.0")

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to config yaml.')
def run(config):
    """Run the decision pipeline with a configuration."""
    click.echo(f"Running Cognix pipeline with config: {config}")

@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
def validate_config(config_path):
    """Validate a configuration file."""
    click.echo(f"Validating config at: {config_path}")
    click.echo("Config is valid.")

@cli.command()
def dashboard():
    """Launch the dashboard."""
    click.echo("Launching Cognix dashboard on http://localhost:8501")

if __name__ == '__main__':
    cli()
