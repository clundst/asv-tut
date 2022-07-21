import time, os

import awkward as ak
import hist
import matplotlib.pyplot as plt
import numpy as np
from coffea import processor
from coffea.nanoevents import schemas

fileset = {'SingleMu' : ["root://eospublic.cern.ch//eos/root-eos/benchmark/Run2012B_SingleMu.root"]}
class Suite:
    timeout = 1200.00
    def TrackQ4(self, n):
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
            executor = processor.DaskExecutor(client=client, status=False)
        else:
            executor = processor.IterativeExecutor()
        run = processor.Runner(executor=executor,
                            schema=schemas.NanoAODSchema,
                            savemetrics=True,
                            chunksize=n,
                            )
        tic = time.monotonic()
        output, metrics = run(fileset, "Events", processor_instance=Q4Processor())
        workers = len(client.scheduler_info()['workers'])
        print('workers = ', workers, ' cores = ', 2*workers)
        toc = time.monotonic()
        walltime = toc - tic
        return walltime/(2*workers)
    
    TrackQ4.params = [2 ** 17, 2 ** 18, 2 ** 19]
    TrackQ4.param_names = ['chunksize']