import asyncio
import time
import logging
import vector; vector.register_awkward()
import awkward as ak
#import cabinetry
from coffea import processor
from coffea.processor import servicex
from coffea.nanoevents import transforms
from coffea.nanoevents.methods import base, vector
from coffea.nanoevents.schemas.base import BaseSchema, zip_forms
from func_adl import ObjectStream
import hist
import json
import matplotlib.pyplot as plt
import numpy as np
import uproot
import utils

#logging.getLogger("cabinetry").setLevel(logging.INFO)

PIPELINE = n
USE_DASK = True
SERVICEX_IGNORE_CACHE = True
AF = "coffea_casa"

N_FILES_MAX_PER_SAMPLE = 10
fileset = utils.construct_fileset(N_FILES_MAX_PER_SAMPLE, use_xcache=False)
class Suite:
    timeout = 1200.00
    def TrackQ1(self, n):
        class Q1Processor(processor.ProcessorABC):
            t0 = time.time()
            if PIPELINE == "coffea":
                if USE_DASK:
                    executor = processor.DaskExecutor(client=utils.get_client(AF))
                else:
                    executor = processor.IterativeExecutor()
                from coffea.nanoevents.schemas.schema import auto_schema
                schema = AGCSchema if PIPELINE == "coffea" else auto_schema
                run = processor.Runner(executor=executor, schema=schema, savemetrics=True, metadata_cache={})
                all_histograms, metrics = run(fileset, "events", processor_instance=TtbarAnalysis())
                all_histograms = all_histograms["hist"]
            elif PIPELINE == "servicex_processor":
                async def produce_all_the_histograms(fileset):
                    return await utils.produce_all_histograms(fileset, use_dask=False)
                all_histograms = asyncio.run(produce_all_the_histograms(fileset))
            elif PIPELINE == "servicex_databinder":
                # needs a slightly different schema, not currently implemented
                raise NotImplementedError("further processing of this method is not currently implemented")
        run = processor.Runner(executor=executor,
                       schema=schemas.NanoAODSchema,
                       savemetrics=True,
                       chunksize=2 ** 19,
                      )
        tic = time.monotonic()
        output, metrics = run(fileset, "Events", processor_instance=Q1Processor())
        workers = len(client.scheduler_info()['workers'])
        print('workers = ', workers, ' cores = ', 2*workers)
        toc = time.monotonic()
        walltime = toc - tic
        return walltime/(2*workers)
    TrackQ1.params =  (["coffea", "servicex_processor", "servicex_databinder"])