import asyncio
import time
import logging
import vector; vector.register_awkward()
import awkward as ak
import cabinetry
from coffea import processor
from coffea.processor import servicex
from coffea.nanoevents import transforms
from coffea.nanoevents.methods import base, vector
from coffea.nanoevents.schemas.base import BaseSchema, zip_forms
from func_adl import ObjectStream
import hist
import json
import matplotlib.pyplot as plt
import numpy as np
import uproot
import utils

logging.getLogger("cabinetry").setLevel(logging.INFO)

PIPELINE = "coffea"
USE_DASK = True
SERVICEX_IGNORE_CACHE = True
AF = "coffea_casa"

processor_base = processor.ProcessorABC if (PIPELINE != "servicex_processor") else servicex.Analysis
def flat_variation(ones):
    return (1.0 + np.array([0.001, -0.001], dtype=np.float32)) * ones[:, None]
def btag_weight_variation(i_jet, jet_pt):
    return 1 + np.array([0.1, -0.1]) * (ak.singletons(jet_pt[:, i_jet]) / 50).to_numpy()
def jet_pt_resolution(pt):
    counts = ak.num(pt)
    pt_flat = ak.flatten(pt)
    resolution_variation = np.random.normal(np.ones_like(pt_flat), 0.05)
    return ak.unflatten(resolution_variation, counts)

class TtbarAnalysis(processor_base):
    def __init__(self):
        num_bins = 25
        bin_low = 50
        bin_high = 550
        name = "observable"
        label = "observable [GeV]"
        self.hist = (
            hist.Hist.new.Reg(num_bins, bin_low, bin_high, name=name, label=label)
            .StrCat(["4j1b", "4j2b"], name="region", label="Region")
            .StrCat([], name="process", label="Process", growth=True)
            .StrCat([], name="variation", label="Systematic variation", growth=True)
            .Weight()
        )
    def process(self, events):
        histogram = self.hist.copy()
        process = events.metadata["process"]
        variation = events.metadata["variation"]
        x_sec = events.metadata["xsec"]
        nevts_total = events.metadata["nevts"]
        lumi = 3378 # /pb
        if process != "data":
            xsec_weight = x_sec * lumi / nevts_total
        else:
            xsec_weight = 1
        if process == "wjets":
            events.add_systematic("scale_var", "UpDownSystematic", "weight", flat_variation)
        events["pt_nominal"] = 1.0
        events["pt_scale_up"] = 1.03
        events["pt_res_up"] = jet_pt_resolution(events.jet.pt)
        pt_variations = ["pt_nominal", "pt_scale_up", "pt_res_up"] if variation == "nominal" else ["pt_nominal"]
        for pt_var in pt_variations:
            selected_electrons = events.electron[events.electron.pt > 25]
            selected_muons = events.muon[events.muon.pt > 25]
            jet_filter = events.jet.pt * events[pt_var] > 25
            selected_jets = events.jet[jet_filter]
            event_filters = (ak.count(selected_electrons.pt, axis=1) & ak.count(selected_muons.pt, axis=1) == 1)
            pt_var_modifier = events[pt_var] if "res" not in pt_var else events[pt_var][jet_filter]
            event_filters = event_filters & (ak.count(selected_jets.pt * pt_var_modifier, axis=1) >= 4)
            B_TAG_THRESHOLD = 0.5
            event_filters = event_filters & (ak.sum(selected_jets.btag >= B_TAG_THRESHOLD, axis=1) >= 1)
            selected_events = events[event_filters]
            selected_electrons = selected_electrons[event_filters]
            selected_muons = selected_muons[event_filters]
            selected_jets = selected_jets[event_filters]
            for region in ["4j1b", "4j2b"]:
                if region == "4j1b":
                    region_filter = ak.sum(selected_jets.btag >= B_TAG_THRESHOLD, axis=1) == 1
                    selected_jets_region = selected_jets[region_filter]
                    pt_var_modifier = events[event_filters][region_filter][pt_var] if "res" not in pt_var else events[pt_var][jet_filter][event_filters][region_filter]
                    observable = ak.sum(selected_jets_region.pt * pt_var_modifier, axis=-1)
                elif region == "4j2b":
                    region_filter = ak.sum(selected_jets.btag > B_TAG_THRESHOLD, axis=1) >= 2
                    selected_jets_region = selected_jets[region_filter]
                    if PIPELINE == "servicex_processor":
                        selected_jets_region = ak.zip(
                            {
                                "pt": selected_jets_region.pt, "eta": selected_jets_region.eta, "phi": selected_jets_region.phi,
                                "mass": selected_jets_region.mass, "btag": selected_jets_region.btag,
                            },
                            with_name="Momentum4D",
                        )
                    trijet = ak.combinations(selected_jets_region, 3, fields=["j1", "j2", "j3"])
                    trijet["p4"] = trijet.j1 + trijet.j2 + trijet.j3
                    trijet["max_btag"] = np.maximum(trijet.j1.btag, np.maximum(trijet.j2.btag, trijet.j3.btag))
                    trijet = trijet[trijet.max_btag > B_TAG_THRESHOLD]
                    trijet_mass = trijet["p4"][ak.argmax(trijet.p4.pt, axis=1, keepdims=True)].mass
                    observable = ak.flatten(trijet_mass)
                if pt_var == "pt_nominal":
                    histogram.fill(
                            observable=observable, region=region, process=process, variation=variation, weight=xsec_weight
                        )
                    if variation == "nominal":
                        for weight_name in events.systematics.fields:
                            for direction in ["up", "down"]:
                                weight_variation = events.systematics[weight_name][direction][f"weight_{weight_name}"][event_filters][region_filter]
                                histogram.fill(
                                    observable=observable, region=region, process=process, variation=f"{weight_name}_{direction}", weight=xsec_weight*weight_variation
                                )
                        for i_var, weight_name in enumerate([f"btag_var_{i}" for i in range(4)]):
                            for i_dir, direction in enumerate(["up", "down"]):
                                if len(observable):
                                    weight_variation = btag_weight_variation(i_var, selected_jets_region.pt)[:, 1-i_dir]
                                else:
                                    weight_variation = 1
                                histogram.fill(
                                    observable=observable, region=region, process=process, variation=f"{weight_name}_{direction}", weight=xsec_weight*weight_variation
                                )
                elif variation == "nominal":
                    histogram.fill(
                            observable=observable, region=region, process=process, variation=pt_var, weight=xsec_weight
                        )
        output = {"nevents": {events.metadata["dataset"]: len(events)}, "hist": histogram}
        return output
    def postprocess(self, accumulator):
        return accumulator
    
class AGCSchema(BaseSchema):
    def __init__(self, base_form):
        super().__init__(base_form)
        self._form["contents"] = self._build_collections(self._form["contents"])

    def _build_collections(self, branch_forms):
        names = set([k.split('_')[0] for k in branch_forms.keys() if not (k.startswith('number'))])
        names = [k for k in names if not (k.startswith('n') | k.startswith('met') | k.startswith('GenPart') | k.startswith('PV'))]
        output = {}
        for name in names:
            offsets = transforms.counts2offsets_form(branch_forms['number' + name])
            content = {k[len(name)+1:]: branch_forms[k] for k in branch_forms if (k.startswith(name + "_") & (k[len(name)+1:] != 'e'))}
            content['energy'] = branch_forms[name+'_e']
            output[name] = zip_forms(content, name, 'PtEtaPhiELorentzVector', offsets=offsets)
        output['met'] = zip_forms({k[len('met')+1:]: branch_forms[k] for k in branch_forms if k.startswith('met_')}, 'met')
        output['PV'] = zip_forms({k[len('PV')+1:]: branch_forms[k] for k in branch_forms if (k.startswith('PV_') & ('npvs' not in k))}, 'PV', offsets=transforms.counts2offsets_form(branch_forms['nPV_x']))
        return output
    @property
    def behavior(self):
        behavior = {}
        behavior.update(base.behavior)
        behavior.update(vector.behavior)
        return behavior
    
print(f"processes in fileset: {list(fileset.keys())}")
print(f"\nexample of information in fileset:\n{{\n  'files': [{fileset['ttbar__nominal']['files'][0]}, ...],")
print(f"  'metadata': {fileset['ttbar__nominal']['metadata']}\n}}")

def get_query(source: ObjectStream) -> ObjectStream:
    """Query for event / column selection: no filter, select relevant lepton and jet columns
    """
    return source.Select(lambda e: {
                                    "electron_pt": e.electron_pt,
                                    "muon_pt": e.muon_pt,
                                    "jet_pt": e.jet_pt,
                                    "jet_eta": e.jet_eta,
                                    "jet_phi": e.jet_phi,
                                    "jet_mass": e.jet_mass,
                                    "jet_btag": e.jet_btag,
                                   }
                        )

if PIPELINE == "servicex_databinder":
    from servicex_databinder import DataBinder
    t0 = time.time()
    query_string = """Where(
        lambda event: event.electron_pt.Where(lambda pT: pT > 25).Count() + event.muon_pt.Where(lambda pT: pT > 25).Count() == 1
        ).Where(lambda event: event.jet_pt.Where(lambda pT: pT > 25).Count() >= 4
        ).Where(lambda event: event.jet_btag.Where(lambda btag: btag > 0.5).Count() >= 1
        ).Select(
             lambda e: {"electron_pt": e.electron_pt, "muon_pt": e.muon_pt,
                        "jet_pt": e.jet_pt, "jet_eta": e.jet_eta, "jet_phi": e.jet_phi, "jet_mass": e.jet_mass, "jet_btag": e.jet_btag}
    )"""
    sample_names = ["ttbar__nominal", "ttbar__scaledown", "ttbar__scaleup", "ttbar__ME_var", "ttbar__PS_var",
                    "single_top_s_chan__nominal", "single_top_t_chan__nominal", "single_top_tW__nominal", "wjets__nominal"]
    sample_names = ["single_top_s_chan__nominal"]  # for quick tests: small dataset with only 50 files
    sample_list = []

    for sample_name in sample_names:
        sample_list.append({"Name": sample_name, "RucioDID": f"user.ivukotic:user.ivukotic.{sample_name}", "Tree": "events", "FuncADL": query_string})
    databinder_config = {
                            "General": {
                                           "ServiceXBackendName": "uproot",
                                            "OutputDirectory": "outputs_databinder",
                                            "OutputFormat": "root",
                                            "IgnoreServiceXCache": SERVICEX_IGNORE_CACHE
                            },
                            "Sample": sample_list
                        }

    sx_db = DataBinder(databinder_config)
    out = sx_db.deliver()
    print(f"execution took {time.time() - t0:.2f} seconds")
    

    

class Suite:
    timeout = 1200.00
    params = ([10, 100, 500, 1000, -1])

    def setup(self, n):
        N_FILES_MAX_PER_SAMPLE = n
        fileset = utils.construct_fileset(N_FILES_MAX_PER_SAMPLE, use_xcache=False)
        class Q1Processor(processor.ProcessorABC):
            t0 = time.time()
            if PIPELINE == "coffea":
                if USE_DASK:
                    executor = processor.DaskExecutor(client=utils.get_client(AF))
                else:
                    executor = processor.IterativeExecutor()
                from coffea.nanoevents.schemas.schema import auto_schema
                schema = AGCSchema if PIPELINE == "coffea" else auto_schema
                run = processor.Runner(executor=executor, schema=schema, savemetrics=True, metadata_cache={})
                all_histograms, metrics = run(fileset, "events", processor_instance=TtbarAnalysis())
                all_histograms = all_histograms["hist"]
            elif PIPELINE == "servicex_processor":
                async def produce_all_the_histograms(fileset):
                    return await utils.produce_all_histograms(fileset, use_dask=False)
                all_histograms = asyncio.run(produce_all_the_histograms(fileset))
            elif PIPELINE == "servicex_databinder":
                # needs a slightly different schema, not currently implemented
                raise NotImplementedError("further processing of this method is not currently implemented")
        tic = time.monotonic()
        output, metrics = run(processor.Runner(executor=executor,
                                               schema=schemas.NanoAODSchema,
                                               savemetrics=True,
                                               chunksize=2 ** 19,
                                              ))
        workers = len(client.scheduler_info()['workers'])
        print('workers = ', workers, ' cores = ', 2*workers)
        toc = time.monotonic()
        walltime = toc - tic
        ave_num_threads = metrics['processtime']/(toc-tic)
        metrics['walltime']=walltime
        metrics['ave_num_threads']=ave_num_threads
        with open('output.pickle', 'wb') as fd:
            pickle.dump(metrics, fd, protocol=pickle.HIGHEST_PROTOCOL)
        return fd

    def TrackWalltime(self,n):
        with open('output.pickle', 'rb') as fd:
                 run_data = pickle.load(fd)
        return run_data['walltime']
    TrackWalltime.param_names = ['Walltime']