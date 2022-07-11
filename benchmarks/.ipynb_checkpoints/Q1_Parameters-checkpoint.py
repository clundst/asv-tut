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
    def TimeQ1(self, n, ncores):
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
            #from dask.distributed import Client
            #client = Client("tls://localhost:8786")
            executor = processor.FuturesExecutor(workers=ncores, status=False)
        else:
            #executor = processor.IterativeExecutor()
            executor = processor.FuturesExecutor(workers=ncores, status=False)
        run = processor.Runner(executor=executor,
                       schema=schemas.NanoAODSchema,
                       savemetrics=True,
                       chunksize=n
                      )
        run(fileset, "Events", processor_instance=Q1Processor())
    TimeQ1.params = ([2 ** 17, 2 ** 18, 2 ** 19], [1, 2])
    
    
    
    
    
    
    
    
#    def time_ranges(n, func_name):
 #   f = {'range': range, 'arange': numpy.arange}[func_name]
  #  for i in f(n):
   #     pass
   # time_ranges.params = ([10, 1000], [2, 4])