"""Write csv-stats' fitted theta for some sampled scenarios, for
check_lmm_optimum.R to evaluate with lme4's own deviance function.

Run from the csv-stats folder:
    python tests/reference/write_ours_check.py RIS_006 RIS_025
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # tests/: test_lmm_scenarios, _lmm_oracle

import test_lmm_scenarios as T  # noqa: E402

out = {}
for case in sys.argv[1:]:
    f = T.our_fit(case)
    out[case] = {"theta": [float(t) for t in f.theta], "names": list(T.model(case).theta_names),
                 "criterion": float(f.criterion), "path": f.optimizer_path}
    print(f"{case}: criterion {f.criterion:.10f} ({f.optimizer_path})")
path = HERE / "scenarios" / "ours_check.json"
path.write_text(json.dumps(out, indent=1))
print(f"wrote {path}")
