import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs' / 'datasets.json'
DRIVE_PATTERN = re.compile(r'/file/d/([^/]+)')


def test_dataset_download_config():
    value = json.loads(CONFIG.read_text())
    expected = {'ptbxl_original', 'ptbxl_noisy', 'ptbxl_denoised'}
    assert set(value['datasets']) == expected
    for name, item in value['datasets'].items():
        assert DRIVE_PATTERN.search(item['url']).group(1) == item['drive_id']
        assert Path(item['archive_name']).name == item['archive_name']
        assert item['format'] in {'zip', 'tar'}
        assert item['role']
    assert isinstance(value['feature_archives'], list)
