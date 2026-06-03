from rag.device import choose_torch_device, nvidia_smi_summary


if __name__ == "__main__":
    smi = nvidia_smi_summary()
    if smi:
        print("nvidia-smi:", smi)
    else:
        print("nvidia-smi: khong co hoac khong doc duoc")

    embedding_choice = choose_torch_device("EMBEDDING_DEVICE", default="cpu")
    reranker_choice = choose_torch_device("RERANKER_DEVICE", default="cpu")

    print("embedding:", embedding_choice)
    print("reranker:", reranker_choice)
