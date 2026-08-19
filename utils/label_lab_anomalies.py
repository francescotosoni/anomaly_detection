"""
Labelling of the anomalies recorded in the laboratory.

The lab anomaly files have NO ground truth (no `actual_digital_input_bits` as in
the academic dataset): labels must be derived from the physical signal.

Due regimi, distinti per natura dell'anomalia:

  * DISTRIBUTED (`weight_*`): the extra payload is present for the whole duration
    the recording (verified: median |F| 34-49 N against 10-12 N in the normal
    runs, from the first instant to the last) -> label = 1 over the whole file.

  * A EVENTI (`push_*`): spinte manuali brevi e localizzate (|F| mediana
    invariata, ma picchi 75-125 N contro max 60-67 N dei normali) -> label = 1
    only where the force exceeds the normal baseline of the same task.

Labels derive from the force measured at the TCP, NOT from the model: evaluation
stays independent of the model under test (no circularity). They are also
"tight" (they cover the event only, not the tail up to end of file): exactly the
correction that paper_comparison.md flagged as the most impactful.

Usage:
  python utils/label_lab_anomalies.py                        # task1 + task2
  python utils/label_lab_anomalies.py --task 1 --plot        # with a visual check
  python utils/label_lab_anomalies.py --threshold-quantile 0.9999
"""

import os
import sys
import json
import glob
import argparse

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.lab_io import read_telemetry, force_magnitude, SAMPLING_HZ

# --- Detection parameter defaults (documented in the JSON produced) ---
DEFAULTS = dict(
    threshold_quantile=0.999,  # quantile della |F| normale: soglia per i push (picchi netti)
    min_duration_s=0.10,       # eventi più brevi = rumore, scartati
    merge_gap_s=1.00,          # buchi più corti di così: stesso evento
    dilate_pre_s=0.25,         # margine prima (la spinta inizia prima del picco)
    dilate_post_s=0.50,        # margine dopo (risposta dinamica/oscillazione)
    # --- variante "stretta" (weight): morfologia diversa, parametri dedicati ---
    # It must EXCLUDE the normal-force phases, hence minimal merge/dilation:
    # with merge_gap=1s the dips below threshold (short but frequent) would be
    # closed again and coverage would rise to 76-96%, defeating the variant.
    tight_quantile=0.990,
    tight_min_duration_s=0.04,
    tight_merge_gap_s=0.10,
    tight_dilate_pre_s=0.00,
    tight_dilate_post_s=0.05,
)

# Every anomalous file gets TWO label sets, for different uses:
#
#   'events'       PRIMARY label following the anomaly regime.
#                  - weight: the whole file (the Graabaek et al. convention for
#                    change_weight: the anomaly is the task configuration).
#                  - push:   localised events only.
#                  Use for EVENT-BASED metrics (industrial deployment).
#
#   'events_tight' TIGHT label: only where the signal genuinely leaves the normal
#                  envelope. For pushes it coincides with 'events'; for weights it
#                  marks only the phases where the payload actually loads the TCP
#                  (30-66% of samples show a force indistinguishable from normal
#                  when the object is released).
#                  Use for PER-TIMESTEP metrics (ROC-AUC), where persistent labels
#                  persistenti introdurrebbero l'artefatto descritto in
#                  paper_comparison.md.
#
# Different thresholds for different morphologies: pushes are sharp peaks (75-125 N
# against a normal max of 60-67 N) and tolerate a strict threshold (q99.9); an altered
# payload is a moderate, sustained elevation, for which the boundary of the normal
# envelope (q99) is the appropriate criterion. Both derive from the normal baseline of
# the SAME task, so there is no leakage between tasks.

# Prefissi -> regime di anomalia
DISTRIBUTED_PREFIXES = ('weight',)
EVENT_PREFIXES = ('push',)


def detect_events(mask, merge_gap, min_len):
    """
    Group the marked samples into inclusive [start, end] events.

    `merge_gap` is the maximum distance (in samples) between two detections for
    them to belong to the same event. CONTIGUOUS samples are 1 apart, so the
    effective threshold cannot go below 1: otherwise merge_gap=0 would split even
    a continuous sequence into single-sample groups (then dropped by min_len).
    """
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    gap = max(1, int(merge_gap))
    groups = [[int(idx[0]), int(idx[0])]]
    for i in idx[1:]:
        if i - groups[-1][1] <= gap:
            groups[-1][1] = int(i)
        else:
            groups.append([int(i), int(i)])
    return [g for g in groups if g[1] - g[0] + 1 >= min_len]


def dilate_events(events, pre, post, n_samples):
    """Widen every event by (pre, post) samples, with clipping and re-merge."""
    out = []
    for a, b in events:
        out.append([max(0, a - pre), min(n_samples - 1, b + post)])
    # Dilation can make adjacent events overlap: merge them again
    merged = []
    for ev in sorted(out):
        if merged and ev[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], ev[1])
        else:
            merged.append(ev)
    return merged


def events_to_mask(events, n_samples):
    """Convert a list of events into a per-timestep boolean array."""
    mask = np.zeros(n_samples, dtype=bool)
    for a, b in events:
        mask[a:b + 1] = True
    return mask


def classify(basename):
    """Determine the anomaly regime from the file name."""
    stem = basename.lower()
    if stem.startswith(DISTRIBUTED_PREFIXES):
        return 'distributed'
    if stem.startswith(EVENT_PREFIXES):
        return 'event'
    return None


def normal_baseline(task_dir):
    """|F| statistics over the normal runs of the task (local reference)."""
    files = sorted(glob.glob(os.path.join(task_dir, 'run*.csv')))
    if not files:
        raise RuntimeError(f"no normal run (run*.csv) found in {task_dir}")
    forces = [force_magnitude(read_telemetry(f)) for f in files]
    allf = np.concatenate(forces)
    return allf, dict(n_files=len(files), n_samples=int(len(allf)),
                      median=float(np.median(allf)), q99=float(np.quantile(allf, 0.99)),
                      q999=float(np.quantile(allf, 0.999)), max=float(allf.max()))


def label_task(task, base='data/lab', params=None, plot=False):
    """Label every anomalous file of a task. Returns the results dict."""
    p = dict(DEFAULTS)
    if params:
        p.update(params)
    task_dir = os.path.join(base, f'task{task}')

    allf, stats = normal_baseline(task_dir)
    threshold = float(np.quantile(allf, p['threshold_quantile']))
    threshold_tight = float(np.quantile(allf, p['tight_quantile']))

    print("=" * 74)
    print(f"  TASK {task} — labeling anomalie")
    print("=" * 74)
    print(f"  Baseline normale: {stats['n_files']} run, {stats['n_samples']} campioni")
    print(f"    |F| mediana={stats['median']:.1f}  q99={stats['q99']:.1f}  "
          f"q99.9={stats['q999']:.1f}  max={stats['max']:.1f} N")
    print(f"  Push threshold  (q{p['threshold_quantile']}): {threshold:.1f} N")
    print(f"  Tight threshold (q{p['tight_quantile']}): {threshold_tight:.1f} N  "
          f"[for the 'tight' variant of the weight files]")
    print()

    merge_gap = int(p['merge_gap_s'] * SAMPLING_HZ)
    min_len = int(p['min_duration_s'] * SAMPLING_HZ)
    pre = int(p['dilate_pre_s'] * SAMPLING_HZ)
    post = int(p['dilate_post_s'] * SAMPLING_HZ)
    t_merge = int(p['tight_merge_gap_s'] * SAMPLING_HZ)
    t_min = int(p['tight_min_duration_s'] * SAMPLING_HZ)
    t_pre = int(p['tight_dilate_pre_s'] * SAMPLING_HZ)
    t_post = int(p['tight_dilate_post_s'] * SAMPLING_HZ)

    out_files = {}
    plot_data = []

    for path in sorted(glob.glob(os.path.join(task_dir, '*.csv'))):
        name = os.path.basename(path)
        kind = classify(name)
        if kind is None:
            continue  # run normali e file non riconosciuti

        F = force_magnitude(read_telemetry(path))
        n = len(F)

        if kind == 'distributed':
            # PRIMARY: the whole recording (the altered payload is always fitted)
            events = [[0, n - 1]]
            # TIGHT: only where the payload actually loads the TCP
            raw_t = detect_events(F > threshold_tight, t_merge, t_min)
            events_tight = dilate_events(raw_t, t_pre, t_post, n)
            note = "payload present for the whole recording"
        else:
            raw = detect_events(F > threshold, merge_gap, min_len)
            events = dilate_events(raw, pre, post, n)
            events_tight = events  # for pushes the primary label is already tight
            note = f"{len(raw)} eventi grezzi -> {len(events)} dopo dilatazione/merge"

        mask = events_to_mask(events, n)
        frac = float(mask.mean())
        frac_tight = float(events_to_mask(events_tight, n).mean())
        durations = [(b - a + 1) / SAMPLING_HZ for a, b in events]

        # Events touching the start/end of the file are TRUNCATED: recording was
        # started/stopped mid-anomaly. Evaluation may exclude them (the signal is
        # partial) without counting them as model false positives.
        truncated = [i for i, (a, b) in enumerate(events)
                     if (a <= 1 or b >= n - 2) and kind == 'event']

        out_files[name] = dict(
            type=kind, n_samples=n,
            events=events, events_tight=events_tight,
            anomalous_fraction=frac, anomalous_fraction_tight=frac_tight,
            event_durations_s=[round(d, 3) for d in durations],
            truncated_events=truncated,
        )
        print(f"  {name:14s} [{kind:11s}] {len(events):2d} eventi  "
              f"{100 * frac:5.1f}% anomalo  durate={[f'{d:.2f}s' for d in durations]}")
        print(f"    -> {note}")
        if kind == 'distributed':
            print(f"    -> tight variant: {len(events_tight)} segments, "
                  f"{100 * frac_tight:.1f}% of the file (the rest has normal-looking force)")
        if truncated:
            print(f"    [!] events truncated at start/end of recording: indices {truncated}")
        plot_data.append((name, F, mask, events_to_mask(events_tight, n) if kind == 'distributed' else None))

    result = dict(
        task=task, sampling_hz=SAMPLING_HZ,
        threshold_N=threshold, threshold_tight_N=threshold_tight,
        params=p, normal_baseline=stats, files=out_files,
    )

    out_json = os.path.join(base, f'labels_task{task}.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  [OK] Label salvate in {out_json}")

    if plot and plot_data:
        _plot(task, plot_data, threshold, stats, base)

    return result


def _plot(task, plot_data, threshold, stats, base):
    """Visual check: |F| over time with the labelled regions highlighted."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(plot_data), 1, figsize=(14, 2.4 * len(plot_data)), squeeze=False)
    for ax, (name, F, mask, mask_tight) in zip(axes[:, 0], plot_data):
        t = np.arange(len(F)) / SAMPLING_HZ
        top = F.max() * 1.05
        ax.fill_between(t, 0, top, where=mask, color='#ffcc00', alpha=0.30,
                        label='label primaria')
        if mask_tight is not None:
            ax.fill_between(t, 0, top, where=mask_tight, color='#ff6600', alpha=0.35,
                            label='label stretta')
        ax.plot(t, F, lw=0.6, color='#3366cc', label='|F| TCP')
        ax.axhline(threshold, color='#cc3333', ls='--', lw=1,
                   label=f'soglia {threshold:.0f} N')
        ax.axhline(stats['median'], color='#888', ls=':', lw=1,
                   label=f"mediana normale {stats['median']:.0f} N")
        ax.set_title(f'Task {task} — {name}', fontsize=10)
        ax.set_ylabel('|F| (N)')
        ax.set_xlim(0, t[-1])
        ax.set_ylim(0, top)
        ax.legend(fontsize=7, loc='upper right', ncol=5)
    axes[-1, 0].set_xlabel('tempo (s)')
    fig.tight_layout()
    out_png = os.path.join(base, f'labels_task{task}.png')
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"  [OK] Visual check saved to {out_png}")


def load_labels(task, base='data/lab', variant='primary'):
    """
    Load the labels of a task (for evaluation).

    variant='primary' -> EVENT-BASED metrics (weight = the whole recording)
    variant='tight'   -> PER-TIMESTEP metrics (weight = out-of-range force only)

    Returns (dict filename -> info with a per-timestep boolean 'mask', metadata).
    """
    if variant not in ('primary', 'tight'):
        raise ValueError("variant must be 'primary' or 'tight'")
    path = os.path.join(base, f'labels_task{task}.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run first: python utils/label_lab_anomalies.py --task {task}")
    with open(path) as f:
        data = json.load(f)
    key = 'events' if variant == 'primary' else 'events_tight'
    out = {}
    for name, info in data['files'].items():
        out[name] = dict(info)
        out[name]['mask'] = events_to_mask(info[key], info['n_samples'])
        out[name]['variant'] = variant
    return out, data


def main():
    ap = argparse.ArgumentParser(description="Labelling of laboratory anomalies")
    ap.add_argument('--task', type=int, nargs='+', default=[1, 2], help="Tasks to label")
    ap.add_argument('--base', default='data/lab', help="Base folder of the lab data")
    ap.add_argument('--plot', action='store_true', help="Save a PNG for visual checking")
    ap.add_argument('--threshold-quantile', type=float, default=DEFAULTS['threshold_quantile'])
    ap.add_argument('--min-duration-s', type=float, default=DEFAULTS['min_duration_s'])
    ap.add_argument('--merge-gap-s', type=float, default=DEFAULTS['merge_gap_s'])
    ap.add_argument('--dilate-pre-s', type=float, default=DEFAULTS['dilate_pre_s'])
    ap.add_argument('--dilate-post-s', type=float, default=DEFAULTS['dilate_post_s'])
    ap.add_argument('--tight-quantile', type=float, default=DEFAULTS['tight_quantile'])
    args = ap.parse_args()

    params = dict(
        threshold_quantile=args.threshold_quantile,
        min_duration_s=args.min_duration_s,
        merge_gap_s=args.merge_gap_s,
        dilate_pre_s=args.dilate_pre_s,
        dilate_post_s=args.dilate_post_s,
        tight_quantile=args.tight_quantile,
    )

    totals = {}
    for t in args.task:
        res = label_task(t, base=args.base, params=params, plot=args.plot)
        n_ev = sum(len(v['events']) for v in res['files'].values() if v['type'] == 'event')
        n_dist = sum(1 for v in res['files'].values() if v['type'] == 'distributed')
        totals[t] = (n_ev, n_dist)
        print()

    print("=" * 74)
    print("  SUMMARY - size of the evaluation set")
    print("=" * 74)
    for t, (n_ev, n_dist) in totals.items():
        print(f"   Task {t}: {n_ev} localised push events | "
              f"{n_dist} altered-payload runs (distributed anomaly)")


if __name__ == '__main__':
    main()
