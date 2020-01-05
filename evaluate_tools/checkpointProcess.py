def ProcessCheckpoint(checkpoint):
    sd = {}
    quantized = False
    for k in checkpoint.keys():
        truekey=k.replace('.wrapped_module','')
        truekey=truekey.replace('module.','')
        sd[truekey]=checkpoint[k]
    return sd
