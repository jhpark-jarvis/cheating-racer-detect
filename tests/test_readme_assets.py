"""Public README figures must exist, retain provenance and match selected bytes."""

import hashlib
import json
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


class Figures(unittest.TestCase):
    def test_assets_are_named_bounded_pngs_with_provenance(self):
        ledger = json.loads((ROOT/'assets/figures/provenance.json').read_text(encoding='utf-8'))
        self.assertEqual(ledger['schema'], 'readme-figures-v1')
        self.assertEqual(len(ledger['figures']), 5)
        self.assertEqual({f['file'] for f in ledger['figures']}, {'concept-scene.png', 'pipeline.png', 'observed.png', 'prediction.png', 'ambiguity.png'})
        for record in ledger['figures']:
            data = (ROOT/'assets/figures'/record['file']).read_bytes()
            self.assertEqual(data[:8], b'\x89PNG\r\n\x1a\n')
            self.assertLess(len(data), 3_000_000)
            self.assertEqual(hashlib.sha256(data).hexdigest(), record['sha256'])
            self.assertTrue(record['scope'])
            if record['kind'].startswith('ai-generated'):
                self.assertEqual(record['generator'], 'built-in image_gen')
                self.assertIn('Constraints:', record['prompt'])
            else:
                self.assertEqual(record['kind'], 'actual-synthetic-render')
                self.assertEqual(record['renderer'], 'cheating_racer_detect/review/overlay.py')

    def test_readme_embeds_all_figures_and_local_links_resolve(self):
        readme = (ROOT/'README.md').read_text(encoding='utf-8')
        images = re.findall(r'!\[[^\]]*\]\((assets/figures/[^)]+)\)', readme)
        self.assertEqual(len(images), 8)
        for target in re.findall(r'\]\(([^)]+)\)', readme):
            if target.startswith(('http:', 'https:', '#')):
                continue
            self.assertNotIn('..', target.split('/'))
            self.assertTrue((ROOT/target.split('#')[0]).is_file(), target)

    def test_readme_keeps_concepts_and_evidence_distinct(self):
        readme = (ROOT/'README.md').read_text(encoding='utf-8')
        for marker in ('AI 생성', '합성', '실영상', '미검증', '미정', '원본', 'VFR'):
            self.assertIn(marker, readme)
        self.assertNotIn('cheating-racer-detect-dev-docs', readme)
        self.assertNotIn('AGENTS.md', readme)

    def test_original_vector_diagram_is_safe_and_has_verified_provenance(self):
        ledger = json.loads((ROOT/'assets/figures/provenance.json').read_text(encoding='utf-8'))
        self.assertEqual(len(ledger['diagrams']), 3)
        record = ledger['diagrams'][0]
        self.assertEqual(record['file'], 'input-replay.svg')
        data = (ROOT/'assets/figures'/record['file']).read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), record['sha256'])
        self.assertLess(len(data), 20_000)
        svg = ET.fromstring(data)
        self.assertEqual(svg.tag, '{http://www.w3.org/2000/svg}svg')
        self.assertEqual(svg.attrib['viewBox'], '0 0 1200 370')
        for item in svg.iter():
            self.assertNotIn(item.tag.split('}')[-1], ('script', 'image', 'foreignObject', 'a'))
            self.assertFalse(any(key.startswith('on') or key.endswith('href') for key in item.attrib))
        labels = ' '.join(svg.itertext())
        for word in ('SHA256', 'unknown', 'not a violation', 'not accuracy'):
            self.assertIn(word, labels)
        for source in record['source']:
            self.assertTrue((ROOT/source).is_file())

    def test_new_diagrams_are_original_safe_and_reproducible(self):
        ledger = json.loads((ROOT/'assets/figures/provenance.json').read_text(encoding='utf-8'))
        records = {r['file']: r for r in ledger['diagrams'][1:]}
        self.assertEqual(set(records), {'assessment-coverage.svg', 'research-context.svg'})
        for name, record in records.items():
            data = (ROOT/'assets/figures'/name).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), record['sha256'])
            self.assertLess(len(data), 20_000)
            svg = ET.fromstring(data)
            for item in svg.iter():
                self.assertNotIn(item.tag.split('}')[-1], ('script', 'image', 'foreignObject', 'a'))
                self.assertFalse(any(k.startswith('on') or k.endswith('href') for k in item.attrib))
            for source in record['source']:
                self.assertTrue((ROOT/source).is_file())
            labels = ' '.join(svg.itertext())
            if name == 'assessment-coverage.svg':
                for text in ('3/4', '2/4', '1/4', 'MISSED EVENT', 'unknown', 'not measured'):
                    self.assertIn(text, labels)
            else:
                self.assertEqual(record['references'], ['https://arxiv.org/abs/2107.08430', 'https://arxiv.org/abs/1905.01333'])
                for text in ('YOLOX', 'DeepSignals', 'not implemented', 'unknown'):
                    self.assertIn(text, labels)


if __name__ == '__main__':
    unittest.main()
