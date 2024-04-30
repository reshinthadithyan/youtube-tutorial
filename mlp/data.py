import torch



def get_data(data_shape):
    return torch.randn(data_shape)

def dummy_fn(inp):
    return inp**2

def make_synth_data(samples,feature):
    x = get_data((samples,2))
    y = dummy_fn(x)
    return x,y


if __name__ == "__main__":
    x,y = make_synth_data(10,2)
    print(x.shape)