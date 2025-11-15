"""Minimal GOK:AI core entry point - M_Boga"""
import yaml, os

def load_config(path=None):
    path = path or os.path.join(os.path.dirname(__file__), 'config.yml')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run(config_path=None):
    cfg = load_config(config_path)
    print('M_Boga GOK core starting with config:')
    for k,v in cfg.items():
        print(f'  {k}: {v}')
    # placeholder for core loop
    print('Pretend we initialized neural modules and quantum pipes...')

if __name__ == '__main__':
    run()
