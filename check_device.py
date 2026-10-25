from rag.device import choose_torch_device, nvidia_smi_summary
from shared.config import get_settings


if __name__ == "__main__":
    smi = nvidia_smi_summary()
    if smi:
        print("nvidia-smi:", smi)
    else:
        print("nvidia-smi: khong co hoac khong doc duoc")

    settings = get_settings()
    print("embedding:", choose_torch_device(settings.embedding.device, "EMBEDDING_DEVICE", settings.force_cuda))
    print("reranker:", choose_torch_device(settings.reranker.device, "RERANKER_DEVICE", settings.force_cuda))
