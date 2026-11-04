"""Render the original 2026-11-04 synthetic study summary with no ML dependencies."""
import argparse
from pathlib import Path

# Case, foreground fraction, no-mask error, oracle error, oracle selected features.
# px/pair errors compare the fitted transform to the known background transform.
# Actual local YOLOX-s found zero vehicle boxes in each of these 12 texture images;
# its union mask therefore excludes nothing and has the same error as no-mask.
RESULTS = (
    ('small', 0.11953125, 6.00230874488517, 0.000018402489233673048, 180),
    ('medium', 0.33177083333333335, 6.000939431124725, 0.000048608995625334496, 180),
    ('dominant', 0.6140625, 5.999678071578443, 0.00004715176286984738, 180),
    ('weak_background', 0.6140625, 6.000284594722111, None, 0),
    ('moving_background', 0.6140625, 4.123031139278468, 0.029586753743272083, 180),
    ('near_full', 0.8765625, 6.001585884102299, None, 0),
)


def render():
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 650" role="img" aria-labelledby="title desc">',
             '<title id="title">Moving foreground: mask provenance changes the background fit</title>',
             '<desc id="desc">Six own synthetic image pairs. No mask and actual detector masks select foreground motion. Oracle exclusion improves four background fits; two lack usable background features. Not driving accuracy.</desc>',
             '<rect width="1080" height="650" fill="#f8fafc"/>',
             '<g font-family="Arial, sans-serif" fill="#14263d">',
             '<text x="36" y="42" font-size="25" font-weight="bold">Background motion needs evidence beyond a fitted transform</text>',
             '<text x="36" y="70" font-size="15">Own 320 x 192 moving textures | OpenCV 4.11.0 | 6 cases x 6 mask variants</text>',
             '<text x="36" y="96" font-size="15">Shown: maximum error at 3 background control points (px/pair; smaller is closer to oracle).</text>']
    headers = [(36, 'Case / foreground union'), (330, 'No exclusion'), (555, 'Oracle exclusion'), (790, 'Actual YOLOX mask')]
    for x, label in headers:
        parts.append(f'<text x="{x}" y="136" font-size="16" font-weight="bold">{label}</text>')
    for i, (case, fraction, none, oracle, _features) in enumerate(RESULTS):
        y = 151+i*61
        parts.append(f'<rect x="24" y="{y}" width="1032" height="56" rx="6" fill="{"#ffffff" if i%2==0 else "#edf2f7"}"/>')
        parts.append(f'<text x="36" y="{y+23}" font-size="16">{case.replace("_", " ")}</text>')
        parts.append(f'<text x="36" y="{y+43}" font-size="13" fill="#52647a">{100*fraction:.1f}% foreground union</text>')
        for x, error, color in ((330, none, '#b45309'), (555, oracle, '#137d6b'), (790, none, '#b45309')):
            if error is None:
                parts.append(f'<text x="{x}" y="{y+31}" font-size="16" fill="#52647a">unknown (no features)</text>')
            else:
                label = '&lt;0.0001' if error < .0001 else f'{error:.4f}'
                parts.append(f'<rect x="{x}" y="{y+33}" width="{error/6.1*190:.4f}" height="6" fill="{color}"/>')
                parts.append(f'<text x="{x}" y="{y+25}" font-size="16" fill="{color}">{label}</text>')
    parts.extend(['<text x="36" y="548" font-size="16" font-weight="bold">Actual detector: 0 vehicle boxes / 12 texture images. Empty exclusion is not a valid background mask.</text>',
                  '<text x="36" y="577" font-size="15">Oracle = known source/target rectangle union + 12px. Auto = actual box union + 12px.</text>',
                  '<text x="36" y="601" font-size="15">Inset, shifted and total exclusion variants test mask failures; oracle is not automatic performance.</text>',
                  '<text x="36" y="625" font-size="15">All ego / visibility states remain unknown. Not car accuracy, 3D pose or physical lane identity.</text>',
                  '</g></svg>\n'])
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('xb') as stream:
        stream.write(render().encode('utf-8'))


if __name__ == '__main__':
    main()
