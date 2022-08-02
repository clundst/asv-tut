import time, os
import awkward as ak
import hist
import matplotlib.pyplot as plt
import numpy as np
from coffea import processor
from coffea.nanoevents import schemas
import pickle

fileset = {'SingleMu' : ["root://eospublic.cern.ch//eos/root-eos/benchmark/Run2012B_SingleMu.root"]}

class Suite:
    def setup_cache(self):
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
            executor = processor.DaskExecutor(client=client, status=False)
            #executor = processor.FuturesExecutor(workers=ncores, status=False)
        else:
            executor = processor.IterativeExecutor()
            #executor = processor.FuturesExecutor(workers=ncores, status=False)
        run = processor.Runner(executor=executor,
                            schema=schemas.NanoAODSchema,
                            savemetrics=True,
                            chunksize=2**19,
                            )
        tic = time.monotonic()
        output, metrics = run(fileset, "Events", processor_instance=Q1Processor())
        workers = len(client.scheduler_info()['workers'])
        print('workers = ', workers, ' cores = ', 2*workers)
        toc = time.monotonic()
        walltime = toc - tic
        ave_num_threads = metrics['processtime']/(toc-tic)
        metrics['walltime']=walltime
        metrics['ave_num_threads']=ave_num_threads
        metrics['chunksize'] = 2**19
        with open('output.pickle', 'wb') as fd:
            pickle.dump(metrics, fd, protocol=pickle.HIGHEST_PROTOCOL)
        return metrics
    def TrackWalltime(self, metrics):
        with open('output.pickle', 'rb') as fd:
                 run_data = pickle.load(fd)
        return run_data['walltime']
    TrackWalltime.param_names = ['Walltime']
    def TrackThreadcount(self, fd):
        with open('output.pickle', 'rb') as fd:
            run_data = pickle.load(fd)
        return run_data['ave_num_threads']
    TrackThreadcount.param_names = ['Average Number of Threads']
    def TrackBytes(self, fd):
        with open('output.pickle', 'rb') as fd:
            run_data = pickle.load(fd)
        return run_data['bytesread']/run_data['walltime']
    TrackBytes.param_names = ['Bytes per Second']
    def TrackChunksize(self, fd):
        with open('output.pickle', 'rb') as fd:
            run_data = pickle.load(fd)
        return run_data['chunksize']/run_data['walltime']
    TrackChunksize.param_names = ['Chunksize per Second']
    def TrackBytesPerThread(self, fd):
        with open('output.pickle', 'rb') as fd:
            run_data = pickle.load(fd)
        return (run_data['bytesread']/run_data['walltime'])/run_data['ave_num_threads']
    TrackBytesPerThread.param_names = ['Bytes per Thread']
    def TrackChunksizePerThread(self, fd):
        with open('output.pickle', 'rb') as fd:
            run_data = pickle.load(fd)
        return (run_data['chunksize']/run_data['walltime'])/run_data['ave_num_threads']
    TrackChunksizePerThread.param_names = ['Chunksize per Thread']