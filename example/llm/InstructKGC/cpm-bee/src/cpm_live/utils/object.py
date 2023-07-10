import bmtrain as bmt
import pickle
import torch


def allgather_objects(obj):
    if bmt.world_size() == 1:
        return [obj]

    with torch.no_grad():
        data_bytes: bytes = pickle.dumps(obj)
        data_length: int = len(data_bytes)

        gpu_data_length = torch.tensor([data_length], device="cuda", dtype=torch.long)
        gathered_length = bmt.distributed.all_gather(gpu_data_length).view(-1).cpu()
        max_data_length = gathered_length.max().item()

        gpu_data_bytes = torch.zeros(max_data_length, dtype=torch.uint8, device="cuda")
        byte_storage = torch.ByteStorage.from_buffer(data_bytes)
        gpu_data_bytes[:data_length] = torch.ByteTensor(byte_storage)

        gathered_data = bmt.distributed.all_gather(gpu_data_bytes).cpu()

        ret = []
        for i in range(gathered_data.size(0)):
            data_bytes = gathered_data[i, : gathered_length[i].item()].numpy().tobytes()
            ret.append(pickle.loads(data_bytes))
        return ret
