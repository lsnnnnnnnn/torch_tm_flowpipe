# Flowstar Width-Growth Diagnostics

Source run: `flowstar_style_o6_candidate8_output6_cutoff`.
Accepted PyTorch segments compared: `95` through t=`2.400737667399793`.
First >2x final width ratio: `0.75`.
First >5x final width ratio: `1.2966796875`.
First >10x final width ratio: `1.681823283433914`.
First >20x final width ratio: `2.1242673315395533`.
First >50x final width ratio: `2.3460652287784387`.
Dominant dimension at first >10x crossing: `y`.
Do width jumps occur after step rejections? yes.
Is the local oracle failure explainable by already-wide reset boxes? true.
Most likely missing Flow* mechanism: Flow*-style symbolic remainder queue plus normalized insertion/composition.

## Threshold Crossings

| threshold | first t |
| ---: | ---: |
| 2x | 0.75 |
| 5x | 1.2966796875 |
| 10x | 1.681823283433914 |
| 20x | 2.1242673315395533 |
| 50x | 2.3460652287784387 |
