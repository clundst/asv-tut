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
    def TimeQ7(self, n):
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
                            chunksize=n,
                            )
        run(fileset, "Events", processor_instance=Q7Processor())
    TimeQ7.params = [2 ** 17, 2 ** 18, 2 ** 19]

