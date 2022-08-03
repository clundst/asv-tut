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
    timeout = 1200.00
    params = ([2 ** 17, 2 ** 18, 2 ** 19])

    def setup(self,n):
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
            executor = processor.DaskExecutor(client=client, status=False)
        else:
            executor = processor.IterativeExecutor()
        run = processor.Runner(executor=executor,
                            schema=schemas.NanoAODSchema,
                            savemetrics=True,
                            chunksize=n,
                            )
        tic = time.monotonic()
        output, metrics = run(fileset, "Events", processor_instance=Q6Processor())
        workers = len(client.scheduler_info()['workers'])
        print('workers = ', workers, ' cores = ', 2*workers)
        toc = time.monotonic()
        walltime = toc - tic
        ave_num_threads = metrics['processtime']/(toc-tic)
        metrics['walltime']=walltime
        metrics['ave_num_threads']=ave_num_threads
        metrics['chunksize'] = n
        with open('output.pickle', 'wb') as fd:
            pickle.dump(metrics, fd, protocol=pickle.HIGHEST_PROTOCOL)
        return fd
 
    def TrackWalltime(self,n):
        with open('output.pickle', 'rb') as fd:
                 run_data = pickle.load(fd)
        return run_data['walltime']
    TrackWalltime.param_names = ['Walltime']

    def TrackThreadcount(self,n):
        with open('output.pickle', 'rb') as fd:
            run_data = pickle.load(fd)
        return run_data['ave_num_threads']
    TrackThreadcount.param_names = ['Average Number of Threads']

    def TrackBytes(self, n):
        with open('output.pickle', 'rb') as fd:
            run_data = pickle.load(fd)
        return run_data['bytesread']/run_data['walltime']
    TrackBytes.param_names = ['Bytes per Second']

    def TrackChunksize(self, n):
        with open('output.pickle', 'rb') as fd:
            run_data = pickle.load(fd)
        return run_data['chunksize']/run_data['walltime']
    TrackChunksize.param_names = ['Chunksize per Second']

    def TrackBytesPerThread(self, n):
        with open('output.pickle', 'rb') as fd:
            run_data = pickle.load(fd)
        return (run_data['bytesread']/run_data['walltime'])/run_data['ave_num_threads']
    TrackBytesPerThread.param_names = ['Bytes per Thread']

    def TrackChunksizePerThread(self, n):
        with open('output.pickle', 'rb') as fd:
            run_data = pickle.load(fd)
        return (run_data['chunksize']/run_data['walltime'])/run_data['ave_num_threads']
    TrackChunksizePerThread.param_names = ['Chunksize per Thread']