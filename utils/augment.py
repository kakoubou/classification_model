import imgaug.augmenters as iaa

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Sometimes(0.5, iaa.Affine(rotate=180)),
    iaa.GaussianBlur(sigma=(0, 3.0))
])
