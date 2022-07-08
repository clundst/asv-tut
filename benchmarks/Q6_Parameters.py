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
    def TimeQ6(self, n):
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
                            chunksize=n,
                            )
        run(fileset, "Events", processor_instance=Q6Processor())
    TimeQ6.params = [2 ** 17, 2 ** 18, 2 ** 19]