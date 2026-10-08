"""Fixed-quota coverage, preservation, retention and graph-size accounting."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage8a0c_common import *

def mean(x):return float(np.mean(x))
def presence(x):return np.stack([(x==t).any(-1) for t in range(6)],axis=-1)
def main():
    data={}
    for split in ('train','val'):
        with np.load(ROOT/'04_coverage'/f'stage8a0c_{split}_accounting.npz') as z:data[split]={k:z[k] for k in z.files}
    data['combined']={k:np.concatenate([data[s][k] for s in ('train','val')]) for k in data['train']}
    coverage=[];retention=[];comparison=[];graph=[];gates=[]
    for split,d in data.items():
        for population,pmask in [('current_valid',np.ones(len(d['actor_type']),dtype=bool)),('full_horizon_ranking_targets',d['full'])]:
            for name,gm in groups(d['actor_type'],d['motion']).items():
                mask=gm&pmask;n=int(mask.sum())
                if not n:continue
                selected=d['selected_counts'][mask];uncapped=d['uncapped_counts'][mask]
                authorized=QUOTAS[d['actor_type'][mask]]>0
                relevant=uncapped*authorized[:,None,:];q=selected.sum(-1)
                assert (q<=QUOTAS[d['actor_type'][mask]].sum(-1)[:,None]).all()
                assert (q.sum(-1)<=np.array([48,36,42])[d['actor_type'][mask]]).all()
                assert np.array_equal(selected>0,(relevant>0))
                prefix={'Split':split,'Population':population,'Group':name}
                coverage.append({**prefix,'Actors':n,'Candidates':n*6,'MapNonEmptyRate':mean(q>0),'ZeroMapRate':mean(q==0),
                    'MeanEntitiesPerMode':mean(q),'MedianEntities':float(np.median(q)),'P95Entities':float(np.percentile(q,95)),
                    'MaxEntities':int(q.max()),'RouteCenterlineCoverage':mean(uncapped[...,:2].sum(-1)>0),
                    'SemanticSpecificCoverage':mean(selected[...,[4,5]].sum(-1)>0),
                    'MeanDistinctSemanticTypes':mean((selected>0).sum(-1))})
                for tid,etype in enumerate(TYPES):
                    values=relevant[...,tid];chosen=selected[...,tid];total=int(values.sum());selected_total=int(chosen.sum())
                    present=int((values>0).sum());preserved=int(((values>0)&(chosen>0)).sum())
                    assert present==preserved
                    retention.append({'Split':split,'Population':population,'ActorGroup':name,'EntityType':etype,
                        'AuthorizedPrimaryType':bool(authorized[:,tid].any()),'UncappedEntities':total,'SelectedEntities':selected_total,
                        'RetentionRate':selected_total/total if total else None,'CandidatesWithType':present,
                        'CandidatesPreservedType':preserved,'PresencePreservationRate':preserved/present if present else None})
                total=int(relevant.sum());chosen=int(selected.sum());present=int((relevant.sum(-1)>0).sum());preserved=int((q>0).sum())
                retention.append({'Split':split,'Population':population,'ActorGroup':name,'EntityType':'total_primary',
                    'AuthorizedPrimaryType':True,'UncappedEntities':total,'SelectedEntities':chosen,'RetentionRate':chosen/total if total else None,
                    'CandidatesWithType':present,'CandidatesPreservedType':preserved,'PresencePreservationRate':preserved/present if present else None})
                for selector,key in [('GlobalTop8','global_top8_types'),('TypeAwareQuota','entity_types'),('GlobalTop8AllSix_secondary_diagnostic','all_six_top8_types')]:
                    typ=d[key][mask];pr=presence(typ)
                    comparison.append({**prefix,'Selector':selector,'Candidates':n*6,'MeanSelectedEntities':mean((typ>=0).sum(-1)),
                        'MeanDistinctEntityTypes':mean(pr.sum(-1)),**{field:mean(pr[...,tid]) for tid,field in enumerate([
                            'LanePresence','ConnectorPresence','DrivablePresence','CarparkPresence','CrosswalkPresence','WalkwayPresence'])}})
                global_presence=presence(d['global_top8_types'][mask]);quota_presence=selected>0
                assert np.all(quota_presence.sum(-1)>=global_presence.sum(-1))
                map_edges=q.sum(-1);mode_edges=d['neighbor_count'][mask].astype(np.int32)*36
                assert (mode_edges>=0).all() and (mode_edges<=288).all()
                graph.append({**prefix,'Targets':n,'MeanModeMapEdges':mean(map_edges),'P95ModeMapEdges':float(np.percentile(map_edges,95)),
                    'MaxModeMapEdges':int(map_edges.max()),'MeanModeModeEdges':mean(mode_edges),
                    'P95ModeModeEdges':float(np.percentile(mode_edges,95)),'MaxModeModeEdges':int(mode_edges.max()),
                    'ZeroMapTargetRate':mean(map_edges==0),'AnyZeroMapModeTargetRate':mean((q==0).any(-1)),
                    'MeanEntitiesPerMode':mean(q),'MedianEntitiesPerMode':float(np.median(q)),
                    'P95EntitiesPerMode':float(np.percentile(q,95)),'MaxEntitiesPerMode':int(q.max())})
    write_csv(ROOT/'06_tables/stage8a0c_sparse_coverage.csv',coverage)
    write_csv(ROOT/'06_tables/stage8a0c_entity_retention.csv',retention)
    write_csv(ROOT/'06_tables/stage8a0c_selector_comparison.csv',comparison)
    write_csv(ROOT/'06_tables/stage8a0c_graph_statistics.csv',graph)
    for split in ('train','val'):
        by={r['Group']:r for r in coverage if r['Split']==split and r['Population']=='full_horizon_ranking_targets'}
        for group,metric,threshold in [('MovingVehicle','RouteCenterlineCoverage',.95),('Vehicle','MapNonEmptyRate',.80),
            ('ParkedVehicle','MapNonEmptyRate',.70),('Pedestrian','MapNonEmptyRate',.85),('Pedestrian','SemanticSpecificCoverage',.70),('Bicycle','MapNonEmptyRate',.80)]:
            value=by[group][metric];gates.append({'Split':split,'Group':group,'Metric':metric,'Value':value,'Threshold':threshold,'PASS':value>=threshold})
    write_csv(ROOT/'06_tables/stage8a0c_sparse_readiness_gates.csv',gates)
    atomic_json(ROOT/'04_coverage/stage8a0c_coverage_audit.json',{'SparseMapSelector':'PASS','SemanticPresencePreservation':'PASS',
        'SparseCoverage':'PASS' if all(r['PASS'] for r in gates) else 'FAIL','gates':gates,
        'all_authorized_type_presence_preserved':True,'uncapped_coverage_equal_0B':True,
        'Vehicle_max_map_edges':48,'Pedestrian_max_map_edges':36,'Bicycle_max_map_edges':42,'max_interaction_edges':288,
        'GlobalTop8_comparison_pool':'same actor-primary relevant entities; all-six comparison is separate descriptive context',
        'zero_map_actors_retained':True,'GT_used_only_for_offline_population_mask':True,'prediction_metrics_computed':False,
        'thresholds_are_input_support_engineering_gates':True,'training_executed':False})
    print('COVERAGE',all(r['PASS'] for r in gates),'PRESENCE100%',flush=True)

if __name__=='__main__':main()
