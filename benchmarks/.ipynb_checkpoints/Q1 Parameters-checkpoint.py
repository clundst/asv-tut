#def time_range(n):
#    for i in range(n):
#        pass
#time_range.params = [10, 100, 1000, 100000]

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
    def TimeQ1(self, n):
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
                       chunksize=n,
                      )
        run(fileset, "Events", processor_instance=Q1Processor())
    TimeQ1.params = [2 ** 17, 2 ** 18, 2 ** 19]