

import webdataset as wds

from torchvision import transforms

def get_webdataset(data_dir: str) -> wds.WebDataset:
    """
    Function to create a WebDataset for loading images from a webdataset directory.
    @author: Stephen Krol

    :param data_dir: Path to the directory containing the webdataset shards (e.g., "data/laion400m/laion-400m-subset").
    :type data_dir: str

    :return: WebDataset object for loading the dataset.
    :rtype: wds.WebDataset
    """

    # TODO: maybe add some data augmentation here (random crop, horizontal flip, color jitter, etc.)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # load dataset ignore captions for now, just use images
    dataset = (
        wds.WebDataset(data_dir + "/{00000..00024}.tar", shardshuffle=True)
        .decode("pil")
        .to_tuple("jpg")
        .map_tuple(transform)  # apply transform to images, ignore captions for now
    )

    return dataset


if __name__ == "__main__":

    dataset = get_webdataset("laion-cache/incoming/round-0001")

    for i, (image,) in enumerate(dataset):
        print(f"Image shape: {image.shape}")
        if i >= 5:
            break





