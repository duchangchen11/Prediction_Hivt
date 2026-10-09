"""Fixed-scene case coverage and five figure families; no outcome selection."""
from pathlib import Path
import sys, json

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / s) for s in ('00_manifest', '06_planning_interface', '07_evaluation_preflight')]
from stage13a_common import *
from stage13a_artifacts import load_observation, load_evaluation_only
from stage13a_metrics import collision_proxy, headings, interpolate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import shapely

COLORS = ['#286AAB', '#C97617', '#704E98']


def polygons(ax, shape, color, alpha, outline=False):
    if shape.is_empty:
        return
    if shape.geom_type == 'Polygon':
        xy = np.asarray(shape.exterior.coords)
        ax.add_patch(Polygon(xy, facecolor=color, edgecolor='#ADB5B2' if outline else 'none',
                             alpha=alpha, linewidth=.35, zorder=0))
        for hole in shape.interiors:
            ax.add_patch(Polygon(np.asarray(hole.coords), facecolor='white', edgecolor='none', zorder=.1))
    elif hasattr(shape, 'geoms'):
        for part in shape.geoms:
            polygons(ax, part, color, alpha, outline)


def static_map(ax, obs, bounds):
    patch = shapely.box(*bounds)
    polygons(ax, obs.map_geometry['drivable'].intersection(patch), '#CDD8D1', .7, True)
    for shape in obs.map_geometry['walkway_parts']:
        polygons(ax, shape.intersection(patch), '#EEDCC0', .6)
    for shape in obs.map_geometry['crossing_parts']:
        polygons(ax, shape.intersection(patch), '#B9CDDB', .5)
    for lane in obs.map_geometry['MapLaneGeometry']:
        ax.plot(lane[:, 0], lane[:, 1], color='#8D9994', linewidth=.45, alpha=.65, zorder=1)


def ego(ax, obs, labels=None, plan=None):
    h = obs.ego_history
    ax.plot(h[:, 0], h[:, 1], '-o', color='#202D32', linewidth=2, markersize=2, zorder=5)
    ax.scatter(0, 0, color='#202D32', marker='>', s=55, zorder=6)
    if labels is not None:
        p = labels.ego_future_gt.copy()
        p[~labels.ego_future_mask] = np.nan
        ax.plot(np.r_[0, p[:, 0]], np.r_[0, p[:, 1]], color='#1C8356', linewidth=2.2, zorder=5)
        if not labels.ego_future_mask.any():
            ax.text(.04, .94, 'Future GT unavailable', transform=ax.transAxes, va='top', fontsize=8)
    if plan is not None:
        ax.plot(np.r_[0, plan[:, 0]], np.r_[0, plan[:, 1]], '--', color='#C2324D', linewidth=2, zorder=5)


def forecasts(ax, obs, display, history=False):
    for i in display:
        color = COLORS[obs.other_agent_type[i]]
        if history:
            h = obs.other_agent_history[i].copy()
            h[~obs.other_agent_history_mask[i]] = np.nan
            ax.plot(h[:, 0], h[:, 1], ':', color=color, linewidth=1.1, zorder=3)
        x, y = obs.other_agent_current_position[i]
        ax.scatter(x, y, c=color, s=8, zorder=4)
        if not obs.prediction_valid_mask[i]:
            ax.scatter(x, y, marker='x', c='#BD342C', s=25, zorder=5)
            continue
        pred = obs.other_agent_predictions[i]
        for p in pred:
            ax.plot(np.r_[x, p[:, 0]], np.r_[y, p[:, 1]], color=color, alpha=.16, linewidth=.65, zorder=2)
        top = pred[obs.other_agent_probabilities[i].argmax()]
        ax.plot(np.r_[x, top[:, 0]], np.r_[y, top[:, 1]], color=color, alpha=.85, linewidth=1.2, zorder=3)


def rectangle(ax, center, heading, size, color):
    h = np.asarray(size) / 2
    xy = np.array([[h[0], h[1]], [h[0], -h[1]], [-h[0], -h[1]], [-h[0], h[1]]])
    c, s = np.cos(heading), np.sin(heading)
    xy = xy @ np.array([[c, s], [-s, c]]) + center
    ax.add_patch(Polygon(xy, facecolor=color, edgecolor=color, alpha=.24, linewidth=.8, zorder=7))


def proxy_panel(ax, obs, labels, plan, display, bounds):
    static_map(ax, obs, bounds)
    ego(ax, obs, plan=plan)
    forecasts(ax, obs, display)
    angles = headings(obs.other_agent_predictions, obs.other_agent_current_position, obs.other_agent_current_heading)
    valid = np.broadcast_to(obs.prediction_valid_mask[:, None, None], obs.other_agent_predictions.shape[:3])
    pc = collision_proxy(obs, plan, obs.other_agent_predictions, angles, valid, obs.other_agent_probabilities)
    gc = collision_proxy(obs, plan, np.nan_to_num(labels.agent_future_gt[:, None]),
                         np.nan_to_num(labels.agent_future_heading_gt[:, None]), labels.agent_future_mask[:, None])
    for i in display:
        p = labels.agent_future_gt[i].copy()
        p[~labels.agent_future_mask[i]] = np.nan
        ax.plot(np.r_[obs.other_agent_current_position[i, 0], p[:, 0]],
                np.r_[obs.other_agent_current_position[i, 1], p[:, 1]], ':', color='#1C8356', linewidth=1, alpha=.8)
    # Illustrate the first interpolated overlap from either proxy, with physical
    # footprints rather than center-point circles. All actors enter the proxy.
    source = pc if pc['AnyFootprintOverlap'] else gc
    if source['AnyFootprintOverlap']:
        i, k, t = np.argwhere(source['OverlapTimeline'])[0]
        if source is pc:
            path, angle, mask = obs.other_agent_predictions, angles, valid
        else:
            path, angle, mask = np.nan_to_num(labels.agent_future_gt[:, None]), np.nan_to_num(labels.agent_future_heading_gt[:, None]), labels.agent_future_mask[:, None]
        ap, ah, _, _ = interpolate(path, angle, obs.future_times, mask,
            np.broadcast_to(obs.other_agent_current_position[:, None], (*path.shape[:2], 2)),
            np.broadcast_to(obs.other_agent_current_heading[:, None], path.shape[:2]))
        ea = headings(plan[None, None], obs.ego_history[-1][None], obs.ego_history_heading[-1:])[0, 0]
        ep, eh, _, _ = interpolate(plan, ea, obs.future_times, np.ones(12, bool), obs.ego_history[-1], obs.ego_history_heading[-1])
        rectangle(ax, ep[t], eh[t], [4.8, 2.], '#C2324D')
        rectangle(ax, ap[i, k, t], ah[i, k, t], obs.other_agent_sizes_length_width[i], '#A55410')
    gt_text = str(gc['AnyFootprintOverlap']) if labels.agent_future_mask.any() else 'UNKNOWN'
    ax.text(.03, .98, f"All6 predicted proxy: {pc['AnyFootprintOverlap']}\nGT proxy (EvaluationOnly): {gt_text}\nGT pair coverage: {gc['ValidInterpolatedPairFraction']:.1%}",
            transform=ax.transAxes, va='top', fontsize=7.5, bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))


def make_figure(row, tags):
    obs, plan, identity = load_observation(row)
    labels = load_evaluation_only(row)  # plotting/evaluation only; no inference
    # DISPLAY ONLY: nearest current actors, independent of their future or error.
    display = np.argsort(np.linalg.norm(obs.other_agent_current_position, axis=-1), kind='stable')[:12]
    # GT affects only figure bounds here, after inference and case selection.
    # Ensure a turning ego's EvaluationOnly trajectory is not clipped.
    points = np.vstack([obs.ego_history, plan, obs.other_agent_current_position[display],
                        labels.ego_future_gt[labels.ego_future_mask]])
    span = np.maximum(np.ptp(points, axis=0) + 20, [70, 70])
    center = (points.max(0) + points.min(0)) / 2
    bounds = [center[0]-span[0]/2, center[1]-span[1]/2, center[0]+span[0]/2, center[1]+span[1]/2]
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.6), constrained_layout=True)
    titles = ['1. Ego history + future GT\nGT = EvaluationOnly',
              '2. Histories + frozen All6 forecasts\nTop1 emphasized; all modes retained',
              '3. Static map + forecasts\nRecorded ego future = EvaluationOnly',
              '4. Static map + forecasts\nCV ego from observed history',
              '5. Footprint overlap proxies\nGT = EvaluationOnly; unknown masks kept']
    ego(axes[0], obs, labels)
    forecasts(axes[1], obs, display, history=True); ego(axes[1], obs)
    static_map(axes[2], obs, bounds); forecasts(axes[2], obs, display); ego(axes[2], obs, labels)
    static_map(axes[3], obs, bounds); forecasts(axes[3], obs, display); ego(axes[3], obs, plan=plan)
    proxy_panel(axes[4], obs, labels, plan, display, bounds)
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=8.5)
        ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3]); ax.set_aspect('equal')
        ax.set_xlabel('Forward x (m)', fontsize=8); ax.set_ylabel('Left y (m)', fontsize=8)
        ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)
    legend = [Line2D([0], [0], color='#202D32', label='Ego history'),
              Line2D([0], [0], color='#1C8356', label='GT: EvaluationOnly'),
              Line2D([0], [0], color='#C2324D', linestyle='--', label='CV reference')]
    legend += [Line2D([0], [0], color=c, label=n) for c, n in zip(COLORS, TYPES)]
    legend.append(Line2D([0], [0], color='#BD342C', marker='x', linestyle='none', label='Insufficient history / unknown risk'))
    fig.legend(handles=legend, loc='outside lower center', ncol=7, fontsize=8, frameon=False)
    fig.suptitle(f"Sample {row.SampleOrdinal:03d} | {row.MapRegion} | {row.Role} | {', '.join(tags)}\n"
                 f"scene {row.SceneToken} | t0 sample {row.SampleToken} | Fold {row.Fold} | "
                 f"V/P/B={row.VehicleActors}/{row.PedestrianActors}/{row.BicycleActors}; "
                 f"valid={row.PredictionEligible}/{row.CurrentActors}; display nearest {len(display)}, proxy uses all actors",
                 fontsize=9)
    stem = f'stage13a_case_{row.SampleOrdinal:03d}'
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(ROOT/'09_figures'/f'{stem}.{extension}', dpi=150)
    plt.close(fig)
    return dict(SampleOrdinal=int(row.SampleOrdinal), SceneToken=row.SceneToken, SampleToken=row.SampleToken,
        Tags='|'.join(tags), Role=row.Role, MapRegion=row.MapRegion, VehicleActors=int(row.VehicleActors),
        PedestrianActors=int(row.PedestrianActors), BicycleActors=int(row.BicycleActors),
        CurrentActors=int(row.CurrentActors), PredictionEligible=int(row.PredictionEligible),
        GTComplete=bool(row.CompleteEgoEvaluation), PotentialPredictionConflict=bool(row.PotentialPredictionConflict),
        DisplayOnlySubsetInstanceTokens='|'.join(obs.instance_tokens[i] for i in display),
        DisplaySelection='nearest 12 t0 actors; no future/GT/error selection; risk uses all current actors',
        FiveFigureFamilies=True, SameAxesAndEgoFrame=True, GTLabel='EvaluationOnly',
        PNG=f'09_figures/{stem}.png', PDF=f'09_figures/{stem}.pdf', SVG=f'09_figures/{stem}.svg')


def main():
    rows = pd.read_csv(ROOT/'06_planning_interface/stage13a_planning_samples_manifest.csv')
    assert len(rows)==48 and read_json(ROOT/'06_planning_interface/stage13a_small_preflight.json')['Status']=='PASS'
    tags = {}; categorized = {}
    conditions = [('vehicle-rich', rows.VehicleActors>=10), ('pedestrian-rich', rows.PedestrianActors>=5),
                  ('potential-predicted-conflict', rows.PotentialPredictionConflict)]
    for name, condition in conditions:
        selected = rows[condition & rows.CompleteEgoEvaluation].sort_values('SampleOrdinal').drop_duplicates('SceneToken').head(3)
        categorized[name] = selected.SceneToken.tolist()
        for ordinal in selected.SampleOrdinal:
            tags.setdefault(int(ordinal), []).append(name)
    # Every fixed scene is represented, including those without proxy conflicts.
    represented = set(rows.loc[rows.SampleOrdinal.isin(tags), 'SceneToken'])
    for row in rows[rows.Role=='first-evaluable'].itertuples():
        if row.SceneToken not in represented:
            tags.setdefault(row.SampleOrdinal, []).append('ordinary fixed-scene validation')
    missing = rows[rows.Role=='inference-only-missing-future'].groupby('MapRegion', sort=True).head(1)
    for ordinal in missing.SampleOrdinal:
        tags.setdefault(int(ordinal), []).append('missing future GT; inference available, evaluation unavailable')
    cases = [make_figure(rows[rows.SampleOrdinal==i].iloc[0], tags[i]) for i in sorted(tags)]
    dump('08_cases/stage13a_case_manifest.csv', cases)
    atomic_json(ROOT/'08_cases/stage13a_case_audit.json', dict(Status='PASS',Cases=len(cases),
        FixedScenesRepresented=len({r['SceneToken'] for r in cases}),
        VehicleRichScenes=len(categorized['vehicle-rich']), PedestrianRichScenes=len(categorized['pedestrian-rich']),
        PotentialConflictScenes=len(categorized['potential-predicted-conflict']), MissingFutureCases=len(missing),
        SelectionSource='registered fixed12scene pool; observable density or prediction-CV proxy, never GT planning success',
        FiveRequiredFigureFamilies=True, SameAxesPerCase=True, AllGTMarkedEvaluationOnly=True,
        FigureDisplaySubsetOnly=True, ProxyIncludesAllActors=True,
        SourceTimestampMissing='all fixed scenes lack original ego_pose.timestamp in currently accessible raw JSON',
        Dimensions='class priors explicitly used when current raw annotation size is unavailable'))
    print('STAGE13A_CASES_PASS', len(cases), flush=True)


if __name__=='__main__':
    main()
