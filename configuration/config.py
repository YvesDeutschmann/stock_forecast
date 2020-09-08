import yaml
config = yaml.safe_load(open('configuration/config.yaml'))

finnhub_key = config['finnhub_key']
symbols = config['symbols']