#!/usr/bin/env python
# coding: utf-8
#To be run via asv run

import asyncio
import json

import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import uproot

def get_client(af="coffea_casa"):
    if af == "coffea_casa":
        from dask.distributed import Client

        client = Client("tls://localhost:8786")

    elif af == "EAF":
        from lpcdaskgateway import LPCGateway

        gateway = LPCGateway()
        cluster = gateway.new_cluster()
        cluster.scale(10)
        print("Please allow up to 60 seconds for HTCondor worker jobs to start")
        print(f"Cluster dashboard: {str(cluster.dashboard_link)}")

        client = cluster.get_client()

    elif af == "local":
        from dask.distributed import Client

        client = Client()

    else:
        raise NotImplementedError(f"unknown analysis facility: {af}")

    return client


def set_style():
    mpl.style.use("ggplot")
    plt.rcParams["axes.facecolor"] = "none"
    plt.rcParams["axes.edgecolor"] = "222222"
    plt.rcParams["axes.labelcolor"] = "222222"
    plt.rcParams["xtick.color"] = "222222"
    plt.rcParams["ytick.color"] = "222222"
    plt.rcParams["font.size"] = 12
    plt.rcParams['text.color'] = "222222"


def construct_fileset(n_files_max_per_sample, use_xcache=False):
    # using https://atlas-groupdata.web.cern.ch/atlas-groupdata/dev/AnalysisTop/TopDataPreparation/XSection-MC15-13TeV.data
    # for reference
    # x-secs are in pb
    xsec_info = {
        "ttbar": 396.87 + 332.97, # nonallhad + allhad, keep same x-sec for all
        "single_top_s_chan": 2.0268 + 1.2676,
        "single_top_t_chan": (36.993 + 22.175)/0.252,  # scale from lepton filter to inclusive
        "single_top_tW": 37.936 + 37.906,
        "wjets": 61457 * 0.252,  # e/mu+nu final states
        "data": None
    }

    # list of files
    with open("/home/cms-jovyan/US_CMS_Summer_2022/ntuples.json") as f:
        file_info = json.load(f)

    # process into "fileset" summarizing all info
    fileset = {}
    for process in file_info.keys():
        if process == "data":
            continue  # skip data

        for variation in file_info[process].keys():
            file_list = file_info[process][variation]["files"]
            if n_files_max_per_sample != -1:
                file_list = file_list[:n_files_max_per_sample]  # use partial set of samples

            file_paths = [f["path"] for f in file_list]
            if use_xcache:
                file_paths = [f.replace("https://xrootd-local.unl.edu:1094", "root://red-xcache1.unl.edu") for f in file_paths]
            nevts_total = sum([f["nevts"] for f in file_list])
            metadata = {"process": process, "variation": variation, "nevts": nevts_total, "xsec": xsec_info[process]}
            fileset.update({f"{process}__{variation}": {"files": file_paths, "metadata": metadata}})

    return fileset


def save_histograms(all_histograms, fileset, filename):
    nominal_samples = [sample for sample in fileset.keys() if "nominal" in sample]

    all_histograms += 1e-6  # add minimal event count to all bins to avoid crashes when processing a small number of samples

    pseudo_data = (all_histograms[:, :, "ttbar", "ME_var"] + all_histograms[:, :, "ttbar", "PS_var"]) / 2  + all_histograms[:, :, "wjets", "nominal"]

    with uproot.recreate(filename) as f:
        for region in ["4j1b", "4j2b"]:
            f[f"{region}_pseudodata"] = pseudo_data[120j::hist.rebin(2), region]
            for sample in nominal_samples:
                sample_name = sample.split("__")[0]
                f[f"{region}_{sample_name}"] = all_histograms[120j::hist.rebin(2), region, sample_name, "nominal"]

                # b-tagging variations
                for i in range(4):
                    for direction in ["up", "down"]:
                        variation_name = f"btag_var_{i}_{direction}"
                        f[f"{region}_{sample_name}_{variation_name}"] = all_histograms[120j::hist.rebin(2), region, sample_name, variation_name]

                # jet energy scale variations
                for variation_name in ["pt_scale_up", "pt_res_up"]:
                    f[f"{region}_{sample_name}_{variation_name}"] = all_histograms[120j::hist.rebin(2), region, sample_name, variation_name]

            f[f"{region}_ttbar_ME_var"] = all_histograms[120j::hist.rebin(2), region, "ttbar", "ME_var"]
            f[f"{region}_ttbar_PS_var"] = all_histograms[120j::hist.rebin(2), region, "ttbar", "PS_var"]
            for process in ["ttbar", "wjets"]:
                f[f"{region}_{process}_scaledown"] = all_histograms[120j::hist.rebin(2), region, process, "scaledown"]
                f[f"{region}_{process}_scaleup"] = all_histograms[120j::hist.rebin(2), region, process, "scaleup"]


# ### Imports: setting up our environment

# In[2]:


import asyncio
import time
import logging

import vector; vector.register_awkward()

import awkward as ak
import cabinetry
from coffea import processor
from coffea.nanoevents import transforms
from coffea.nanoevents.methods import base, vector
from coffea.nanoevents.schemas.base import BaseSchema, zip_forms
import hist
import json
import matplotlib.pyplot as plt
import numpy as np
import uproot

logging.getLogger("cabinetry").setLevel(logging.INFO)


# ### Configuration: number of files and data delivery path
# 
# The number of files per sample set here determines the size of the dataset we are processing.
# There are 9 samples being used here, all part of the 2015 CMS Open Data release.
# They are pre-converted from miniAOD files into ntuple format, similar to nanoAODs.
# More details about the inputs can be found [here](https://github.com/iris-hep/analysis-grand-challenge/tree/main/datasets/cms-open-data-2015).
# 
# The table below summarizes the amount of data processed depending on the `N_FILES_MAX_PER_SAMPLE` setting.
# 
# | setting | number of files | total size |
# | --- | --- | --- |
# | `10` | 90 | 15.6 GB |
# | `100` | 850 | 150 GB |
# | `500` | 3545| 649 GB |
# | `1000` | 5864 | 1.05 TB |
# | `-1` | 22635 | 3.44 TB |
# 
# The input files are all in the 100–200 MB range.
# 
# Some files are also rucio-accessible (with ATLAS credentials):
# 
# | dataset | number of files | total size |
# | --- | --- | --- |
# | `user.ivukotic:user.ivukotic.ttbar__nominal` | 7066 | 1.46 TB |
# | `user.ivukotic:user.ivukotic.ttbar__scaledown` | 902 | 209 GB |
# | `user.ivukotic:user.ivukotic.ttbar__scaleup` | 917 | 191 GB |
# | `user.ivukotic:user.ivukotic.ttbar__ME_var` | 438 | 103 GB |
# | `user.ivukotic:user.ivukotic.ttbar__PS_var` | 443 | 100 GB |
# | `user.ivukotic:user.ivukotic.single_top_s_chan__nominal` | 114 | 11 GB |
# | `user.ivukotic:user.ivukotic.single_top_t_chan__nominal` | 2506 | 392 GB |
# | `user.ivukotic:user.ivukotic.single_top_tW__nominal` | 50 | 9 GB |
# | `user.ivukotic:user.ivukotic.wjets__nominal` | 10199 | 1.13 TB |
# | total | 22635 | 3.61 TB |
# 
# The difference in total file size is presumably due to the different storages, which report slightly different sizes.
# 
# When setting the `PIPELINE` variable below to `"servicex_databinder"`, the `N_FILES_MAX_PER_SAMPLE` variable is ignored and all files are processed.

# In[3]:


### GLOBAL CONFIGURATION

# input files per process, set to e.g. 10 (smaller number = faster)
N_FILES_MAX_PER_SAMPLE = 10

# pipeline to use:
# - "coffea" for pure coffea setup
PIPELINE = "coffea"

# enabled Dask (not supported yet with ServiceX processors in coffea)
USE_DASK = True

# ServiceX behavior: ignore cache with repeated queries
SERVICEX_IGNORE_CACHE = True

# analysis facility: set to "coffea_casa" for coffea-casa environments, "EAF" for FNAL, "local" for local setups
AF = "coffea_casa"


# ### Defining our `coffea` Processor
# 
# The processor includes a lot of the physics analysis details:
# - event filtering and the calculation of observables,
# - event weighting,
# - calculating systematic uncertainties at the event and object level,
# - filling all the information into histograms that get aggregated and ultimately returned to us by `coffea`.

# In[4]:


processor_base = processor.ProcessorABC

# functions creating systematic variations
def flat_variation(ones):
    # 0.1% weight variations
    return (1.0 + np.array([0.001, -0.001], dtype=np.float32)) * ones[:, None]


def btag_weight_variation(i_jet, jet_pt):
    # weight variation depending on i-th jet pT (10% as default value, multiplied by i-th jet pT / 50 GeV)
    return 1 + np.array([0.1, -0.1]) * (ak.singletons(jet_pt[:, i_jet]) / 50).to_numpy()


def jet_pt_resolution(pt):
    # normal distribution with 5% variations, shape matches jets
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

        process = events.metadata["process"]  # "ttbar" etc.
        variation = events.metadata["variation"]  # "nominal" etc.

        # normalization for MC
        x_sec = events.metadata["xsec"]
        nevts_total = events.metadata["nevts"]
        lumi = 3378 # /pb
        if process != "data":
            xsec_weight = x_sec * lumi / nevts_total
        else:
            xsec_weight = 1

        #### systematics
        # example of a simple flat weight variation, using the coffea nanoevents systematics feature
        if process == "wjets":
            events.add_systematic("scale_var", "UpDownSystematic", "weight", flat_variation)

        # jet energy scale / resolution systematics
        # need to adjust schema to instead use coffea add_systematic feature, especially for ServiceX
        # cannot attach pT variations to events.jet, so attach to events directly
        # and subsequently scale pT by these scale factors
        events["pt_nominal"] = 1.0
        events["pt_scale_up"] = 1.03
        events["pt_res_up"] = jet_pt_resolution(events.jet.pt)

        pt_variations = ["pt_nominal", "pt_scale_up", "pt_res_up"] if variation == "nominal" else ["pt_nominal"]
        for pt_var in pt_variations:

            ### event selection
            # very very loosely based on https://arxiv.org/abs/2006.13076

            # pT > 25 GeV for leptons & jets
            selected_electrons = events.electron[events.electron.pt > 25]
            selected_muons = events.muon[events.muon.pt > 25]
            jet_filter = events.jet.pt * events[pt_var] > 25  # pT > 25 GeV for jets (scaled by systematic variations)
            selected_jets = events.jet[jet_filter]

            # single lepton requirement
            event_filters = (ak.count(selected_electrons.pt, axis=1) & ak.count(selected_muons.pt, axis=1) == 1)
            # at least four jets
            pt_var_modifier = events[pt_var] if "res" not in pt_var else events[pt_var][jet_filter]
            event_filters = event_filters & (ak.count(selected_jets.pt * pt_var_modifier, axis=1) >= 4)
            # at least one b-tagged jet ("tag" means score above threshold)
            B_TAG_THRESHOLD = 0.5
            event_filters = event_filters & (ak.sum(selected_jets.btag >= B_TAG_THRESHOLD, axis=1) >= 1)

            # apply event filters
            selected_events = events[event_filters]
            selected_electrons = selected_electrons[event_filters]
            selected_muons = selected_muons[event_filters]
            selected_jets = selected_jets[event_filters]

            for region in ["4j1b", "4j2b"]:
                # further filtering: 4j1b CR with single b-tag, 4j2b SR with two or more tags
                if region == "4j1b":
                    region_filter = ak.sum(selected_jets.btag >= B_TAG_THRESHOLD, axis=1) == 1
                    selected_jets_region = selected_jets[region_filter]
                    # use HT (scalar sum of jet pT) as observable
                    pt_var_modifier = events[event_filters][region_filter][pt_var] if "res" not in pt_var else events[pt_var][jet_filter][event_filters][region_filter]
                    observable = ak.sum(selected_jets_region.pt * pt_var_modifier, axis=-1)

                elif region == "4j2b":
                    region_filter = ak.sum(selected_jets.btag > B_TAG_THRESHOLD, axis=1) >= 2
                    selected_jets_region = selected_jets[region_filter]

                    if PIPELINE == "servicex_processor":
                        # wrap into a four-vector object to allow addition
                        selected_jets_region = ak.zip(
                            {
                                "pt": selected_jets_region.pt, "eta": selected_jets_region.eta, "phi": selected_jets_region.phi,
                                "mass": selected_jets_region.mass, "btag": selected_jets_region.btag,
                            },
                            with_name="Momentum4D",
                        )

                    # reconstruct hadronic top as bjj system with largest pT
                    # the jet energy scale / resolution effect is not propagated to this observable at the moment
                    trijet = ak.combinations(selected_jets_region, 3, fields=["j1", "j2", "j3"])  # trijet candidates
                    trijet["p4"] = trijet.j1 + trijet.j2 + trijet.j3  # calculate four-momentum of tri-jet system
                    trijet["max_btag"] = np.maximum(trijet.j1.btag, np.maximum(trijet.j2.btag, trijet.j3.btag))
                    trijet = trijet[trijet.max_btag > B_TAG_THRESHOLD]  # require at least one-btag in trijet candidates
                    # pick trijet candidate with largest pT and calculate mass of system
                    trijet_mass = trijet["p4"][ak.argmax(trijet.p4.pt, axis=1, keepdims=True)].mass
                    observable = ak.flatten(trijet_mass)

                ### histogram filling
                if pt_var == "pt_nominal":
                    # nominal pT, but including 2-point systematics
                    histogram.fill(
                            observable=observable, region=region, process=process, variation=variation, weight=xsec_weight
                        )

                    if variation == "nominal":
                        # also fill weight-based variations for all nominal samples
                        for weight_name in events.systematics.fields:
                            for direction in ["up", "down"]:
                                # extract the weight variations and apply all event & region filters
                                weight_variation = events.systematics[weight_name][direction][f"weight_{weight_name}"][event_filters][region_filter]
                                # fill histograms
                                histogram.fill(
                                    observable=observable, region=region, process=process, variation=f"{weight_name}_{direction}", weight=xsec_weight*weight_variation
                                )

                        # calculate additional systematics: b-tagging variations
                        for i_var, weight_name in enumerate([f"btag_var_{i}" for i in range(4)]):
                            for i_dir, direction in enumerate(["up", "down"]):
                                # create systematic variations that depend on object properties (here: jet pT)
                                if len(observable):
                                    weight_variation = btag_weight_variation(i_var, selected_jets_region.pt)[:, 1-i_dir]
                                else:
                                    weight_variation = 1 # no events selected
                                histogram.fill(
                                    observable=observable, region=region, process=process, variation=f"{weight_name}_{direction}", weight=xsec_weight*weight_variation
                                )

                elif variation == "nominal":
                    # pT variations for nominal samples
                    histogram.fill(
                            observable=observable, region=region, process=process, variation=pt_var, weight=xsec_weight
                        )

        output = {"nevents": {events.metadata["dataset"]: len(events)}, "hist": histogram}

        return output

    def postprocess(self, accumulator):
        return accumulator


# ### AGC `coffea` schema
# 
# When using `coffea`, we can benefit from the schema functionality to group columns into convenient objects.
# This schema is taken from [mat-adamec/agc_coffea](https://github.com/mat-adamec/agc_coffea).

# In[5]:


class AGCSchema(BaseSchema):
    def __init__(self, base_form):
        super().__init__(base_form)
        self._form["contents"] = self._build_collections(self._form["contents"])

    def _build_collections(self, branch_forms):
        names = set([k.split('_')[0] for k in branch_forms.keys() if not (k.startswith('number'))])
        # Remove n(names) from consideration. It's safe to just remove names that start with n, as nothing else begins with n in our fields.
        # Also remove GenPart, PV and MET because they deviate from the pattern of having a 'number' field.
        names = [k for k in names if not (k.startswith('n') | k.startswith('met') | k.startswith('GenPart') | k.startswith('PV'))]
        output = {}
        for name in names:
            offsets = transforms.counts2offsets_form(branch_forms['number' + name])
            content = {k[len(name)+1:]: branch_forms[k] for k in branch_forms if (k.startswith(name + "_") & (k[len(name)+1:] != 'e'))}
            # Add energy separately so its treated correctly by the p4 vector.
            content['energy'] = branch_forms[name+'_e']
            # Check for LorentzVector
            output[name] = zip_forms(content, name, 'PtEtaPhiELorentzVector', offsets=offsets)

        # Handle GenPart, PV, MET. Note that all the nPV_*'s should be the same. We just use one.
        output['met'] = zip_forms({k[len('met')+1:]: branch_forms[k] for k in branch_forms if k.startswith('met_')}, 'met')
        #output['GenPart'] = zip_forms({k[len('GenPart')+1:]: branch_forms[k] for k in branch_forms if k.startswith('GenPart_')}, 'GenPart', offsets=transforms.counts2offsets_form(branch_forms['numGenPart']))
        output['PV'] = zip_forms({k[len('PV')+1:]: branch_forms[k] for k in branch_forms if (k.startswith('PV_') & ('npvs' not in k))}, 'PV', offsets=transforms.counts2offsets_form(branch_forms['nPV_x']))
        return output

    @property
    def behavior(self):
        behavior = {}
        behavior.update(base.behavior)
        behavior.update(vector.behavior)
        return behavior


# ### "Fileset" construction and metadata
# 
# Here, we gather all the required information about the files we want to process: paths to the files and asociated metadata.

# In[6]:







# ### Execute the data delivery pipeline
# 
# What happens here depends on the configuration setting for `PIPELINE`:
# - when set to `servicex_processor`, ServiceX will feed columns to `coffea` processors, which will asynchronously process them and accumulate the output histograms,
# - when set to `coffea`, processing will happen with pure `coffea`,
# - if `PIPELINE` was set to `servicex_databinder`, the input data has already been pre-processed and will be processed further with `coffea`.

# In[7]:
class Suite:
    timeout = 1200.00
    params = ([10,100])
    def setup(self,n):
#        fileset = construct_fileset(N_FILES_MAX_PER_SAMPLE, use_xcache=False)
        fileset = construct_fileset(n, use_xcache=False)
        print(f"processes in fileset: {list(fileset.keys())}")
        print(f"\nexample of information in fileset:\n{{\n  'files': [{fileset['ttbar__nominal']['files'][0]}, ...],")
        print(f"  'metadata': {fileset['ttbar__nominal']['metadata']}\n}}")
        t0 = time.time()

#        if PIPELINE == "coffea":
#            if USE_DASK:
#                executor = processor.DaskExecutor(client=get_client(AF),status=False)
#            else:
        executor = processor.IterativeExecutor()

        from coffea.nanoevents.schemas.schema import auto_schema
        schema = AGCSchema if PIPELINE == "coffea" else auto_schema
        run = processor.Runner(executor=executor, schema=schema, savemetrics=True, metadata_cache={})
        metrics = run(fileset, "events", processor_instance=TtbarAnalysis())
#       all_histograms = all_histograms["hist"]    
        walltime = time.time() - t0
        ave_num_threads = metrics['processtime']/(walltime)
        metrics['walltime']=walltime
        metrics['ave_num_threads']=ave_num_threads
        metrics['chunksize'] = n
        
        set_style()

        all_histograms[120j::hist.rebin(2), "4j1b", :, "nominal"].stack("process")[::-1].plot(stack=True, histtype="fill", linewidth=1, edgecolor="grey")
        plt.legend(frameon=False)
        plt.title(">= 4 jets, 1 b-tag")
        plt.xlabel("HT [GeV]");


        # In[9]:


        all_histograms[:, "4j2b", :, "nominal"].stack("process")[::-1].plot(stack=True, histtype="fill", linewidth=1,edgecolor="grey")
        plt.legend(frameon=False)
        plt.title(">= 4 jets, >= 2 b-tags")
        plt.xlabel("$m_{bjj}$ [Gev]");


        # Our top reconstruction approach ($bjj$ system with largest $p_T$) has worked!
        # 
        # Let's also have a look at some systematic variations:
        # - b-tagging, which we implemented as jet-kinematic dependent event weights,
        # - jet energy variations, which vary jet kinematics, resulting in acceptance effects and observable changes.
        # 
        # We are making of [UHI](https://uhi.readthedocs.io/) here to re-bin.

        # In[10]:


        # b-tagging variations
        all_histograms[120j::hist.rebin(2), "4j1b", "ttbar", "nominal"].plot(label="nominal", linewidth=2)
        all_histograms[120j::hist.rebin(2), "4j1b", "ttbar", "btag_var_0_down"].plot(label="NP 1", linewidth=2)
        all_histograms[120j::hist.rebin(2), "4j1b", "ttbar", "btag_var_1_down"].plot(label="NP 2", linewidth=2)
        all_histograms[120j::hist.rebin(2), "4j1b", "ttbar", "btag_var_2_down"].plot(label="NP 3", linewidth=2)
        all_histograms[120j::hist.rebin(2), "4j1b", "ttbar", "btag_var_3_down"].plot(label="NP 4", linewidth=2)
        plt.legend(frameon=False)
        plt.xlabel("HT [GeV]")
        plt.title("b-tagging variations");


        # In[11]:


        # jet energy scale variations
        all_histograms[:, "4j2b", "ttbar", "nominal"].plot(label="nominal", linewidth=2)
        all_histograms[:, "4j2b", "ttbar", "pt_scale_up"].plot(label="scale up", linewidth=2)
        all_histograms[:, "4j2b", "ttbar", "pt_res_up"].plot(label="resolution up", linewidth=2)
        plt.legend(frameon=False)
        plt.xlabel("$m_{bjj}$ [Gev]")
        plt.title("Jet energy variations");


        # ### Save histograms to disk
        # 
        # We'll save everything to disk for subsequent usage.
        # This also builds pseudo-data by combining events from the various simulation setups we have processed.

        # In[12]:


        save_histograms(all_histograms, fileset, "histograms.root")


        # ### Statistical inference
        # 
        # A statistical model has been defined in `config.yml` (below: inlined for easier usage as a standalone script), ready to be used with our output.
        # We will use `cabinetry` to combine all histograms into a `pyhf` workspace and fit the resulting statistical model to the pseudodata we built.

        # In[22]:


        config = {'General': {'Measurement': 'CMS_ttbar',
          'POI': 'ttbar_norm',
          'HistogramFolder': 'histograms/',
          'InputPath': 'histograms.root:{RegionPath}_{SamplePath}{VariationPath}',
          'VariationPath': ''},
         'Regions': [{'Name': '4j1b CR', 'RegionPath': '4j1b'},
          {'Name': '4j2b SR', 'RegionPath': '4j2b'}],
         'Samples': [{'Name': 'Pseudodata', 'SamplePath': 'pseudodata', 'Data': True},
          {'Name': 'ttbar', 'SamplePath': 'ttbar'},
          {'Name': 'W+jets', 'SamplePath': 'wjets'},
          {'Name': 'single top, s-channel', 'SamplePath': 'single_top_s_chan'},
          {'Name': 'single top, t-channel', 'SamplePath': 'single_top_s_chan'},
          {'Name': 'tW', 'SamplePath': 'single_top_tW'}],
         'Systematics': [{'Name': 'ME variation',
           'Type': 'NormPlusShape',
           'Up': {'VariationPath': '_ME_var'},
           'Down': {'Symmetrize': True},
           'Samples': 'ttbar'},
          {'Name': 'PS variation',
           'Type': 'NormPlusShape',
           'Up': {'VariationPath': '_PS_var'},
           'Down': {'Symmetrize': True},
           'Samples': 'ttbar'},
          {'Name': 'ttbar scale variations',
           'Type': 'NormPlusShape',
           'Up': {'VariationPath': '_scaleup'},
           'Down': {'VariationPath': '_scaledown'},
           'Samples': 'ttbar'},
          {'Name': 'Jet energy scale',
           'Type': 'NormPlusShape',
           'Up': {'VariationPath': '_pt_scale_up'},
           'Down': {'Symmetrize': True}},
          {'Name': 'Jet energy resolution',
           'Type': 'NormPlusShape',
           'Up': {'VariationPath': '_pt_res_up'},
           'Down': {'Symmetrize': True}},
          {'Name': 'b-tag NP 1',
           'Type': 'NormPlusShape',
           'Up': {'VariationPath': '_btag_var_0_up'},
           'Down': {'VariationPath': '_btag_var_0_down'}},
          {'Name': 'b-tag NP 2',
           'Type': 'NormPlusShape',
           'Up': {'VariationPath': '_btag_var_1_up'},
           'Down': {'VariationPath': '_btag_var_1_down'}},
          {'Name': 'b-tag NP 3',
           'Type': 'NormPlusShape',
           'Up': {'VariationPath': '_btag_var_2_up'},
           'Down': {'VariationPath': '_btag_var_2_down'}},
          {'Name': 'b-tag NP 4',
           'Type': 'NormPlusShape',
           'Up': {'VariationPath': '_btag_var_3_up'},
           'Down': {'VariationPath': '_btag_var_3_down'}},
          {'Name': 'W+jets scale variations',
           'Type': 'NormPlusShape',
           'Up': {'VariationPath': '_scaleup'},
           'Down': {'VariationPath': '_scaledown'},
           'Samples': 'W+jets'},
          {'Name': 'Luminosity',
           'Type': 'Normalization',
           'Up': {'Normalization': 0.03},
           'Down': {'Normalization': 0.03}}],
         'NormFactors': [{'Name': 'ttbar_norm',
           'Samples': 'ttbar',
           'Nominal': 1.0,
           'Bounds': [0, 10]}]}
        cabinetry.templates.collect(config)
        cabinetry.templates.postprocess(config)  # optional post-processing (e.g. smoothing)
        ws = cabinetry.workspace.build(config)
        cabinetry.workspace.save(ws, "workspace.json")


        # Let's try out what we built: the next cell will perform a maximum likelihood fit of our statistical model to the pseudodata we built.

        # In[23]:


        model, data = cabinetry.model_utils.model_and_data(ws)
        fit_results = cabinetry.fit.fit(model, data)

        cabinetry.visualize.pulls(
            fit_results, exclude="ttbar_norm", close_figure=True, save_figure=False
        )


        # For this pseudodata, what is the resulting ttbar cross-section divided by the Standard Model prediction?

        # In[16]:


        poi_index = model.config.poi_index
        print(f"\nfit result for ttbar_norm: {fit_results.bestfit[poi_index]:.3f} +/- {fit_results.uncertainty[poi_index]:.3f}")


        # Let's also visualize the model before and after the fit, in both the regions we are using.
        # The binning here corresponds to the binning used for the fit.

        # In[17]:


        model_prediction = cabinetry.model_utils.prediction(model)
        figs = cabinetry.visualize.data_mc(model_prediction, data, close_figure=True)
        figs[0]["figure"]


        # In[18]:


        figs[1]["figure"]


        # We can see very good post-fit agreement.

        # In[19]:


        model_prediction_postfit = cabinetry.model_utils.prediction(model, fit_results=fit_results)
        figs = cabinetry.visualize.data_mc(model_prediction_postfit, data, close_figure=True)
        figs[0]["figure"]
        
        with open('output.pickle', 'wb') as fd:
            pickle.dump(metrics, fd, protocol=pickle.HIGHEST_PROTOCOL)
        return fd        

    def TrackThreadcount(self,n):
        with open('output.pickle', 'rb') as fd:
            run_data = pickle.load(fd)
        return run_data['ave_num_threads']
    TrackThreadcount.param_names = ['Average Number of Threads']
                

# ### Inspecting the produced histograms
# 
# Let's have a look at the data we obtained.
# We built histograms in two phase space regions, for multiple physics processes and systematic variations.

# In[8]:




# In[20]:


#figs[1]["figure"]