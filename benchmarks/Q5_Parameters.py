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
    def TimeQ5(self, n):
        class Q5Processor(processor.ProcessorABC):
            def process(self, events):
                mupair = ak.combinations(events.Muon, 2)
                with np.errstate(invalid="ignore"):
                    pairmass = (mupair.slot0 + mupair.slot1).mass
                goodevent = ak.any(
                    (pairmass > 60)
                    & (pairmass < 120)
                    & (mupair.slot0.charge == -mupair.slot1.charge),
                    axis=1,
                )
                return (
                    hist.Hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]")
                    .Double()
                    .fill(events[goodevent].MET.pt)
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
        run(fileset, "Events", processor_instance=Q5Processor())
    TimeQ5.params = [2 ** 17, 2 ** 18, 2 ** 19]