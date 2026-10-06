"""Exact edge- and target-weighted moments of paired formal observations."""
from stage4fd_common import *
from scipy.stats import spearmanr

def metrics(record,weighting,layer=None,head=None):
    assert len(record) and (record['degree']>0).all()
    values=record['values']
    if layer is not None:values=values[:,layer:layer+1]
    if head is not None:values=values[:,:,head:head+1]
    # Float64 moments from actual float32 observations. Head/layer counts
    # are identical for all included targets; weights apply at target level.
    weights=record['degree'].astype(np.float64) if weighting=='EdgeWeighted' else np.ones(len(record))
    mean=(values.mean(axis=(1,2))*weights[:,None]).sum(axis=0)/weights.sum()
    d=dict(zip(METRICS,mean));gate=float(np.dot(record['gate'],weights)/weights.sum());eps=1e-8
    an=d['A_raw']/(d['A_base']+eps);fn=d['F_effective']/(d['F_base']+eps)
    retention=d['F_effective']/(d['A_raw']+eps)
    return {'TargetCount':len(record),'EdgeCount':int(record['degree'].sum()),'GateMean':gate,
        'A_MeanAbsBase':d['A_base'],'A_MeanAbsRawBias':d['A_raw'],'A_NormalizedInteraction':an,
        'F_MeanAbsBase':d['F_base'],'F_MeanAbsRawBias':d['F_raw'],'F_MeanAbsEffectiveBias':d['F_effective'],'F_NormalizedInteraction':fn,
        'RawAmplification':d['F_raw']/(d['A_raw']+eps),'EffectiveRetention':retention,'CompensationIndex':retention/(gate+eps),
        'ExpectedRetentionWithoutCompensation':gate,'NormalizedInteractionRetention':fn/(an+eps),
        'A_AttentionL1Shift':d['A_shift'],'F_AttentionL1Shift':d['F_shift'],'AttentionShiftRetention':d['F_shift']/(d['A_shift']+eps),
        'A_TopNeighborSwitchRate':d['A_switch'],'F_TopNeighborSwitchRate':d['F_switch'],'DeltaSwitchRate':d['F_switch']-d['A_switch'],
        'A_EntropyBase':d['A_Hbase'],'A_EntropyFinal':d['A_Hfinal'],'F_EntropyBase':d['F_Hbase'],'F_EntropyFinal':d['F_Hfinal'],
        'A_DeltaEntropy':d['A_Hfinal']-d['A_Hbase'],'F_DeltaEntropy':d['F_Hfinal']-d['F_Hbase']}

def pair_tables(records):
    raw=read_json(DIAG/'stage4fd_pair_aggregates.json');counts=np.array(raw['counts']);a=np.array(raw['A_sums']);f=np.array(raw['F_sums']);rows=[];heads=[]
    for pair,label in enumerate(PAIRS):
        assert counts[pair]>0
        for li in (None,0,1,2):
            av=a[pair] if li is None else a[pair,li:li+1];fv=f[pair] if li is None else f[pair,li:li+1]
            av=av.mean(axis=(0,1))/counts[pair];fv=fv.mean(axis=(0,1))/counts[pair]
            retention=fv[2]/(av[1]+1e-8)
            rows.append({'Pair':label,'Layer':'All' if li is None else li+1,'Weighting':'EdgeWeighted','EdgeCount':int(counts[pair]),
                'EdgeHeadObservationCount':int(counts[pair])*(24 if li is None else 8),'GateMean':fv[3],
                'A_Raw':av[1],'F_Raw':fv[1],'F_Effective':fv[2],'RawAmplification':fv[1]/(av[1]+1e-8),
                'EffectiveRetention':retention,'CompensationIndex':retention/(fv[3]+1e-8)})
        for li in range(3):
            for h in range(8):
                av=a[pair,li,h]/counts[pair];fv=f[pair,li,h]/counts[pair];ret=fv[2]/(av[1]+1e-8)
                heads.append({'Pair':label,'Layer':li+1,'Head':h+1,'EdgeCount':int(counts[pair]),'GateMean':fv[3],
                    'A_Raw':av[1],'F_Raw':fv[1],'F_Effective':fv[2],'RawAmplification':fv[1]/(av[1]+1e-8),'EffectiveRetention':ret,'CompensationIndex':ret/(fv[3]+1e-8)})
    overall=metrics(records[(records['population']==1)&(records['degree']>0)],'EdgeWeighted')
    # Independent stream sums across directed pairs reconcile target reductions.
    grand_a=a.sum(axis=(0,1,2))/(counts.sum()*24);grand_f=f.sum(axis=(0,1,2))/(counts.sum()*24)
    checks={'A_raw':abs(grand_a[1]-overall['A_MeanAbsRawBias']),'F_raw':abs(grand_f[1]-overall['F_MeanAbsRawBias']),
        'F_effective':abs(grand_f[2]-overall['F_MeanAbsEffectiveBias']),'gate':abs(grand_f[3]-overall['GateMean'])}
    assert max(checks.values())<1e-8
    write_csv(ROOT/'06_tables/stage4fd_effective_strength_by_pair.csv',rows)
    write_csv(ROOT/'06_tables/stage4fd_pair_head_statistics.csv',heads)
    return checks

def targets(records):
    fields=['id','scene_token','sample_token','target_instance_token','type','motion_group','horizon','GT_endpoint_displacement_m','HasIncomingEdges','IncomingEdgeCount','gate',
        'A_mean_abs_base','A_mean_abs_raw','F_mean_abs_base','F_mean_abs_raw','F_mean_abs_effective','A_attention_shift','F_attention_shift',
        'A_top_neighbor_switch_rate','F_top_neighbor_switch_rate','A_entropy_base','A_entropy_final','F_entropy_base','F_entropy_final']
    means=records['values'].mean(axis=(1,2));ids=records['id'];rows=0
    with IDENTITY.open() as source,TARGETS.open('w',newline='') as dest:
        writer=csv.DictWriter(dest,fields,lineterminator='\n');writer.writeheader()
        for i,row in enumerate(csv.DictReader(source)):
            assert int(row['id'])==int(ids[i]);r=records[i];defined=r['degree']>0
            out={k:row[k] for k in ('id','scene_token','sample_token','target_instance_token','type','motion_group','horizon')}
            out.update(GT_endpoint_displacement_m=float(r['displacement']) if np.isfinite(r['displacement']) else '',HasIncomingEdges=int(defined),IncomingEdgeCount=int(r['degree']),gate=float(r['gate']))
            out.update(dict(zip(fields[11:],[float(x) if defined else '' for x in means[i]])));writer.writerow(out);rows+=1
    assert rows==len(records)
    return {'relative_path':str(TARGETS.relative_to(ROOT)),'sha256':sha256(TARGETS),'row_count':rows,'schema':fields,'Git':'local_only','undefined_no_neighbor_metrics':'blank; excluded from group means'}

def main():
    verify();capture=read_json(DIAG/'stage4fd_capture_audit.json');assert capture['status']=='PASS'
    assert sha256(RAW)==capture['raw_target_head_schema']['sha256'] and sha256(IDENTITY)==capture['raw_identity_sha256']
    records=np.memmap(RAW,dtype=DTYPE,mode='r');assert len(records)==capture['all_current_valid_targets']
    masks=group_masks(records);pooled=[];layers=[];heads=[]
    for group,mask in masks.items():
        selected=records[mask]
        assert len(selected),group
        for weighting in ('EdgeWeighted','TargetWeighted'):
            pooled.append({'Group':group,'Weighting':weighting,**metrics(selected,weighting)})
            for li in range(3):
                layers.append({'Group':group,'Weighting':weighting,'Layer':li+1,**metrics(selected,weighting,li)})
                for h in range(8):heads.append({'Group':group,'Weighting':weighting,'Layer':li+1,'Head':h+1,**metrics(selected,weighting,li,h)})
    assert len(heads)==len(GROUPS)*2*24
    write_csv(ROOT/'06_tables/stage4fd_effective_strength_by_group.csv',pooled)
    write_csv(ROOT/'06_tables/stage4fd_layer_statistics.csv',layers)
    write_csv(ROOT/'06_tables/stage4fd_layer_head_statistics.csv',heads)
    checks=pair_tables(records);local=targets(records);correlations=[]
    for group in ('Overall','Vehicle','Pedestrian','Pedestrian <5m','Pedestrian 5-10m'):
        selected=records[masks[group]];means=selected['values'].mean(axis=(1,2))
        for name in ('F_raw','F_effective'):
            values=means[:,METRICS.index(name)];rho=spearmanr(selected['gate'],values).statistic
            correlations.append({'Group':group,'Measurement':name,'TargetCount':len(selected),'Spearman_rho':float(rho) if np.isfinite(rho) else None,
                'Interpretation':'descriptive target-level rank association; overlapping windows; no causal or significance claim'})
    write_csv(ROOT/'06_tables/stage4fd_gate_bias_correlations.csv',correlations)
    # Cross-check gates against the frozen final inference, with no new forward.
    gate_rows={actor_key(r):r for r in csv.DictReader((ROOT/'04_evaluation/stage4f_actor_gates.csv').open()) if r['horizon']!='context_only'}
    max_gate_diff=0.;matched=0
    with TARGETS.open() as f:
        for row in csv.DictReader(f):
            if row['horizon']=='context_only':continue
            key=(row['scene_token'],row['sample_token'],row['target_instance_token'],row['horizon']);old=gate_rows[key]
            max_gate_diff=max(max_gate_diff,abs(float(row['gate'])-float(old['necessity_gate'])));matched+=1
    assert matched==85027 and max_gate_diff<1e-6
    atomic_json(DIAG/'stage4fd_statistics_audit.json',{'status':'PASS','capture_sha256':sha256(DIAG/'stage4fd_capture_audit.json'),
        'raw_target_head_sha256':sha256(RAW),'target_table':local,'pair_vs_target_stream_absolute_differences':checks,
        'frozen_gate_actor_records_matched':matched,'frozen_gate_max_absolute_difference':max_gate_diff,
        'Weightings':['EdgeWeighted','TargetWeighted'],'primary_population':'full-horizon targets with >=1 actual incoming edge',
        'group_rows':len(pooled),'layer_rows':len(layers),'layer_head_rows':len(heads),'pair_layers':36,'pair_heads':216,
        'groups':{g:{'TargetCount':int(m.sum()),'EdgeCount':int(records['degree'][m].sum())} for g,m in masks.items()},
        'metric_definition':'ratios of means with eps1e-8; CompensationIndex=EffectiveRetention/(corresponding weighted gate mean+eps)',
        'attention_full_neighborhood':True,'attention_EdgeWeighted_degree_weighted_and_TargetWeighted_standard_mean':True,
        'GT_only_offline_grouping':True,'no_statistical_significance_threshold':True})
    verify();print('STAGE4FD_STATISTICS_PASS',len(pooled),len(heads),flush=True)

if __name__=='__main__':main()
