# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import time, os

import awkward as ak
import hist
import matplotlib.pyplot as plt
import numpy as np
from coffea import processor
from coffea.nanoevents import schemas

fileset = {'SingleMu' : ["root://eospublic.cern.ch//eos/root-eos/benchmark/Run2012B_SingleMu.root"]}
class TimeSuite:
    timeout = 1200.00
    def TimeQ1(self):
        class Q1Processor(processor.ProcessorABC):
            def process(self, events):
                return (
                    hist.Hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]")
                    .Double()
                    .fill(events.MET.pt)
                )
            def postprocess(self, accumulator):
                return accumulator
        if os.environ.get("LABEXTENTION_FACTORY_MODULE") == "coffea_casa":
            from dask.distributed import Client
            client = Client("tls://localhost:8786")
            executor = processor.DaskExecutor(client=client)
        else:
            executor = processor.IterativeExecutor()
        run = processor.Runner(executor=executor,
                       schema=schemas.NanoAODSchema,
                       savemetrics=True,
                       chunksize=2**19,
                      )
        run(fileset, "Events", processor_instance=Q1Processor())

    def TimeQ2(self):
        class Q2Processor(processor.ProcessorABC):
            def process(self, events):
                return (
                    hist.Hist.new.Reg(100, 0, 200, name="ptj", label="Jet $p_{T}$ [GeV]")
                    .Double()
                    .fill(ak.flatten(events.Jet.pt))
                )
            def postprocess(self, accumulator):
                return accumulator
        if os.environ.get("LABEXTENTION_FACTORY_MODULE") == "coffea_casa":
            from dask.distributed import Client
            client = Client("tls://localhost:8786")
            executor = processor.DaskExecutor(client=client)
        else:
            executor = processor.IterativeExecutor()
        run = processor.Runner(executor=executor,
                       schema=schemas.NanoAODSchema,
                       savemetrics=True,
                       chunksize=2**19,
                      )
        run(fileset, "Events", processor_instance=Q2Processor())

    def TimeQ3(self):
        class Q3Processor(processor.ProcessorABC):
            def process(self, events):
                return (
                    hist.Hist.new.Reg(100, 0, 200, name="ptj", label="Jet $p_{T}$ [GeV]")
                    .Double()
                    .fill(ak.flatten(events.Jet[abs(events.Jet.eta) < 1].pt))
                 )
            def postprocess(self, accumulator):
                return accumulator
        if os.environ.get("LABEXTENTION_FACTORY_MODULE") == "coffea_casa":
            from dask.distributed import Client
            client = Client("tls://localhost:8786")
            executor = processor.DaskExecutor(client=client)
        else:
            executor = processor.IterativeExecutor()

        run = processor.Runner(executor=executor,
                            schema=schemas.NanoAODSchema,
                            savemetrics=True,
                            chunksize=2**19,
                            )
        run(fileset, "Events", processor_instance=Q3Processor())

    def TimeQ4(self):
        class Q4Processor(processor.ProcessorABC):
            def process(self, events):
                has2jets = ak.sum(events.Jet.pt > 40, axis=1) >= 2
                return (
                    hist.Hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]")
                    .Double()
                    .fill(events[has2jets].MET.pt)
                )
            def postprocess(self, accumulator):
                return accumulator
        if os.environ.get("LABEXTENTION_FACTORY_MODULE") == "coffea_casa":
            from dask.distributed import Client
            client = Client("tls://localhost:8786")
            executor = processor.DaskExecutor(client=client)
        else:
            executor = processor.IterativeExecutor()
        run = processor.Runner(executor=executor,
                            schema=schemas.NanoAODSchema,
                            savemetrics=True,
                            chunksize=2**19,
                            )
        run(fileset, "Events", processor_instance=Q4Processor())

    def TimeQ5(self):
        class Q5Processor(processor.ProcessorABC):
            def process(self, events):
                mupair = ak.combinations(events.Muon, 2)
                with np.errstate(invalid="ignore"):
                    pairmass = (mupair.slot0 + mupair.slot1).mass
                goodevent = ak.any(
                    (pairmass > 60)
                    & (pairmass < 120)
                    & (mupair.slot0.charge == -mupair.slot1.charge),
                    axis=1,
                )
                return (
                    hist.Hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]")
                    .Double()
                    .fill(events[goodevent].MET.pt)
                )
            def postprocess(self, accumulator):
                return accumulator
        if os.environ.get("LABEXTENTION_FACTORY_MODULE") == "coffea_casa":
            from dask.distributed import Client
            client = Client("tls://localhost:8786")
            executor = processor.DaskExecutor(client=client)
        else:
            executor = processor.IterativeExecutor()
        run = processor.Runner(executor=executor,
                            schema=schemas.NanoAODSchema,
                            savemetrics=True,
                            chunksize=2**19,
                            )
        run(fileset, "Events", processor_instance=Q5Processor())

    def TimeQ6(self):
        class Q6Processor(processor.ProcessorABC):
            def process(self, events):
                jets = ak.zip(
                    {k: getattr(events.Jet, k) for k in ["x", "y", "z", "t", "btag"]},
                    with_name="LorentzVector",
                    behavior=events.Jet.behavior,
                )
                trijet = ak.combinations(jets, 3, fields=["j1", "j2", "j3"])
                trijet["p4"] = trijet.j1 + trijet.j2 + trijet.j3
                trijet = ak.flatten(
                    trijet[ak.singletons(ak.argmin(abs(trijet.p4.mass - 172.5), axis=1))]
                )
                maxBtag = np.maximum(
                    trijet.j1.btag,
                    np.maximum(
                        trijet.j2.btag,
                        trijet.j3.btag,
                    ),
                )
                return {
                    "trijetpt": hist.Hist.new.Reg(
                        100, 0, 200, name="pt3j", label="Trijet $p_{T}$ [GeV]"
                    )
                    .Double()
                    .fill(trijet.p4.pt),
                    "maxbtag": hist.Hist.new.Reg(
                        100, 0, 1, name="btag", label="Max jet b-tag score"
                    )
                    .Double()
                    .fill(maxBtag),
                }
            def postprocess(self, accumulator):
                return accumulator
        if os.environ.get("LABEXTENTION_FACTORY_MODULE") == "coffea_casa":
            from dask.distributed import Client
            client = Client("tls://localhost:8786")
            executor = processor.DaskExecutor(client=client)
        else:
            executor = processor.IterativeExecutor()
        run = processor.Runner(executor=executor,
                            schema=schemas.NanoAODSchema,
                            savemetrics=True,
                            chunksize=2**19,
                            )
        run(fileset, "Events", processor_instance=Q6Processor())

    def TimeQ7(self):
        class Q7Processor(processor.ProcessorABC):
            def process(self, events):
                cleanjets = events.Jet[
                    ak.all(
                        events.Jet.metric_table(events.Muon[events.Muon.pt > 10]) >= 0.4, axis=2
                    )
                    & ak.all(
                        events.Jet.metric_table(events.Electron[events.Electron.pt > 10]) >= 0.4,
                        axis=2,
                    )
                    & (events.Jet.pt > 30)
                ]
                return (
                    hist.Hist.new.Reg(
                        100, 0, 200, name="sumjetpt", label="Jet $\sum p_{T}$ [GeV]"
                    )
                    .Double()
                    .fill(ak.sum(cleanjets.pt, axis=1))
                )
            def postprocess(self, accumulator):
                return accumulator
        if os.environ.get("LABEXTENTION_FACTORY_MODULE") == "coffea_casa":
            from dask.distributed import Client
            client = Client("tls://localhost:8786")
            executor = processor.DaskExecutor(client=client)
        else:
            executor = processor.IterativeExecutor()
        run = processor.Runner(executor=executor,
                            schema=schemas.NanoAODSchema,
                            savemetrics=True,
                            chunksize=2**19,
                            )
        run(fileset, "Events", processor_instance=Q7Processor())

    def TimeQ8(self):
        class Q8Processor(processor.ProcessorABC):
            def process(self, events):
                events["Electron", "pdgId"] = -11 * events.Electron.charge
                events["Muon", "pdgId"] = -13 * events.Muon.charge
                events["leptons"] = ak.concatenate(
                    [events.Electron, events.Muon],
                    axis=1,
                )
                events = events[ak.num(events.leptons) >= 3]
                pair = ak.argcombinations(events.leptons, 2, fields=["l1", "l2"])
                pair = pair[(events.leptons[pair.l1].pdgId == -events.leptons[pair.l2].pdgId)]
                with np.errstate(invalid="ignore"):
                    pair = pair[
                        ak.singletons(
                            ak.argmin(
                                abs(
                                    (events.leptons[pair.l1] + events.leptons[pair.l2]).mass
                                    - 91.2
                                ),
                                axis=1,
                            )
                        )
                    ]
                events = events[ak.num(pair) > 0]
                pair = pair[ak.num(pair) > 0][:, 0]
                l3 = ak.local_index(events.leptons)
                l3 = l3[(l3 != pair.l1) & (l3 != pair.l2)]
                l3 = l3[ak.argmax(events.leptons[l3].pt, axis=1, keepdims=True)]
                l3 = events.leptons[l3][:, 0]
                mt = np.sqrt(2 * l3.pt * events.MET.pt * (1 - np.cos(events.MET.delta_phi(l3))))
                return (
                    hist.Hist.new.Reg(
                        100, 0, 200, name="mt", label="$\ell$-MET transverse mass [GeV]"
                    )
                    .Double()
                    .fill(mt)
                )
            def postprocess(self, accumulator):
                return accumulator
        if os.environ.get("LABEXTENTION_FACTORY_MODULE") == "coffea_casa":
            from dask.distributed import Client
            client = Client("tls://localhost:8786")
            executor = processor.DaskExecutor(client=client)
        else:
            executor = processor.IterativeExecutor()

        run = processor.Runner(executor=executor,
                            schema=schemas.NanoAODSchema,
                            savemetrics=True,
                            chunksize=2**19,
                            )
        run(fileset, "Events", processor_instance=Q8Processor())

    
    def setup(self):
        self.d = {}
        for x in range(500):
            self.d[x] = None

    def time_keys(self):
        for key in self.d.keys():
            pass

    def time_values(self):
        for value in self.d.values():
            pass

    def time_range(self):
        d = self.d
        for key in range(500):
            x = d[key]


class MemSuite:
    def mem_list(self):
        return [0] * 256
