from dask.distributed import Client
client = Client("tls://localhost:8786")
n = len(client.scheduler_info()['workers'])
print('workers = ', n, ' cores = ', 2*n)