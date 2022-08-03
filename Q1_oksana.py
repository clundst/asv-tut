import time, os
import awkward as ak
import hist
import matplotlib.pyplot as plt
import numpy as np
from coffea import processor
from coffea.nanoevents import NanoAODSchema
import pickle

fileset = {'SingleMu' : ["root://eospublic.cern.ch//eos/root-eos/benchmark/Run2012B_SingleMu.root"]}

class Q1Processor(processor.ProcessorABC):
    def process(self, events):
        return (hist.Hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]")
                    .Double()
                    .fill(events.MET.pt)
                )
    def postprocess(self, accumulator):
        return accumulator


class Q1Suite:
    #timeout = 1200.00
    #params = ([2 ** 17, 2 ** 18, 2 ** 19])
    def setup(self):
        if os.environ.get("LABEXTENTION_FACTORY_MODULE") == "coffea_casa":
            from dask.distributed import Client
            client = Client("tls://localhost:8786")
            self.executor = processor.DaskExecutor(client=client, status=False)
            #executor = processor.FuturesExecutor()
        else:
            self.executor = processor.IterativeExecutor()
            #self.executor = processor.FuturesExecutor()
        self.run = processor.Runner(executor=self.executor,
                            schema=NanoAODSchema,
                            savemetrics=True,
                            #chunksize=n,
                            )

    def track_objects(self):
        tic = time.monotonic()
        #outputs = self.run.preprocess(fileset, "Events")
        output, metrics = self.run(fileset, "Events", processor_instance=Q1Processor())
        toc = time.monotonic()
        
        walltime = toc - tic
        ave_num_threads = metrics['processtime']/(toc-tic)
        metrics['walltime']=walltime
        metrics['ave_num_threads']=ave_num_threads
        metrics['chunksize'] = n
        return metrics['walltime']

    track_objects.unit = "walltime"
