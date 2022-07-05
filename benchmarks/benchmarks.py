# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import time, os

import awkward as ak
import hist
import matplotlib.pyplot as plt
import numpy as np
from coffea import processor
from coffea.nanoevents import schemas

class TimeSuite:
    fileset = {'SingleMu' : ["root://eospublic.cern.ch//eos/root-eos/benchmark/Run2012B_SingleMu.root"]}
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
        tic = time.monotonic()
        output, metrics = run(fileset, "Events", processor_instance=Q1Processor())
        toc = time.monotonic()
        metrics["walltime"] = toc - tic
        metrics
    
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
