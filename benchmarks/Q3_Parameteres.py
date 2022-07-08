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
    def TimeQ3(self, n):
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
                            chunksize=n,
                            )
        run(fileset, "Events", processor_instance=Q3Processor())
    TimeQ3.params = [2 ** 17, 2 ** 18, 2 ** 19]