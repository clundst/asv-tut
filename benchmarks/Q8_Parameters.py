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
    def TimeQ8(self, n):
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
                            chunksize=n,
                            )
        run(fileset, "Events", processor_instance=Q8Processor())
    TimeQ8.params = [2 ** 17, 2 ** 18, 2 ** 19]