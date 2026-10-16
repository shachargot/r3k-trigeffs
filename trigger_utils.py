"""Small shared helpers (no ROOT dependency, so they can be unit-tested anywhere)."""
import copy
import glob
import math
import re
from pathlib import Path


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, key):
        # Never fabricate dunder attributes (copy/pickle probe for them)
        if key.startswith('__') and key.endswith('__'):
            raise AttributeError(key)
        return self.get(key, False)

    def __setattr__(self, key, value):
        self[key] = DotDict(value) if isinstance(value, dict) else value

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            raise AttributeError(f'No attribute named {key}')

    def __deepcopy__(self, memo):
        return DotDict(copy.deepcopy(dict(self), memo))

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, DotDict) else v for k, v in self.items()}


def get_input_files(inputs, n_files=None):
    """Expand files / directories / wildcard patterns (absolute or relative) into sorted .root paths."""
    def _process(p):
        p = str(p)
        if any(ch in p for ch in '*?['):
            return sorted(glob.glob(p, recursive=True))
        pp = Path(p)
        if pp.is_dir():
            return sorted(str(f) for f in pp.glob('*.root'))
        if pp.is_file() and pp.suffix == '.root':
            return [str(pp)]
        return []

    paths = [inputs] if isinstance(inputs, (str, Path)) else inputs
    files = sorted(set(f for path in paths for f in _process(path)))
    return files[:n_files] if n_files and n_files <= len(files) else files


# ---------------------------------------------------------------- geometry
def phi_mpi_pi(x):
    """Wrap an angle difference into (-pi, pi]."""
    while x > math.pi:
        x -= 2 * math.pi
    while x <= -math.pi:
        x += 2 * math.pi
    return x


def delta_r(eta1, phi1, eta2, phi2):
    return math.hypot(eta1 - eta2, phi_mpi_pi(phi1 - phi2))


# ---------------------------------------------------------------- L1 acceptance
def max_dr_for(seed_name):
    """L1 seeds carry their DeltaR cut in the name, e.g. DoubleEG6_er1p2_dR_Max0p8 -> 0.8."""
    m = re.search(r'dR_Max(\d+p\d+)', seed_name or '')
    return float(m.group(1).replace('p', '.')) if m else 999.


def build_max_dr(trigger_map):
    """{path: max DeltaR(e1,e2) allowed by that path's L1 seed}. 'trigger_OR' (value 0/None)
    gets the loosest cut among the real paths, i.e. the acceptance of the whole menu."""
    out = {k: max_dr_for(v[0]) for k, v in trigger_map.items() if v}
    loosest = max(out.values()) if out else 999.
    for k, v in trigger_map.items():
        if not v:
            out[k] = loosest
    return out


# ---------------------------------------------------------------- HLT object matching
def match_trigger_objects(eta, phi, trig_objs, bit, dr_max=0.1, pdg_id=11):
    """Indices of TrigObj entries of the given id that passed filter `bit` and lie within dr_max."""
    hits = set()
    for i, t in enumerate(trig_objs):
        if abs(t.id) != pdg_id:
            continue
        if not (int(t.filterBits) >> bit) & 1:
            continue
        if delta_r(eta, phi, t.eta, t.phi) < dr_max:
            hits.add(i)
    return hits


def both_matched(m0, m1):
    """Both electrons matched, and to two *different* trigger objects."""
    return bool(m0) and bool(m1) and len(m0 | m1) >= 2
