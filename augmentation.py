import numpy as np
import cupy as cp
from scipy.ndimage import rotate as ndi_rotate
from scipy.ndimage import shift as ndi_shift


def add_uniform_noise(x,epsilon,rng):
    if epsilon<=0:
        return x.copy()

    n=rng.uniform(
        -epsilon,epsilon,x.shape
    ).astype(x.dtype)

    return cp.clip(x+n,0,1)


def horizontal_flip(sequence,height,width):
    frames=sequence.reshape(
        -1,height,width
    )

    return frames[:,:,::-1].reshape(
        sequence.shape
    ).copy()


def rigid_transform(
    sequence,
    height,
    width,
    angle,
    dx,
    dy
):
    frames=sequence.reshape(
        -1,height,width
    )

    transformed=ndi_rotate(
        frames,
        angle=float(angle),
        axes=(1,2),
        reshape=False,
        order=1,
        mode="reflect",
        prefilter=False
    )

    transformed=ndi_shift(
        transformed,
        shift=(0,float(dy),float(dx)),
        order=1,
        mode="reflect",
        prefilter=False
    )

    return np.clip(
        transformed,0,1
    ).astype(
        np.float32
    ).reshape(
        sequence.shape
    )


def augment_dataset(
    sequences,
    height,
    width,
    seed,
    use_flip=True,
    affine_copies=2,
    max_shift=2,
    max_rotation=3.0
):
    sequences=np.asarray(
        sequences,
        dtype=np.float32
    )

    rng=np.random.default_rng(seed)

    augmented=[]
    labels=[]

    for i,sequence in enumerate(sequences):
        augmented.append(sequence.copy())
        labels.append((i,"original"))

        if use_flip:
            augmented.append(
                horizontal_flip(
                    sequence,
                    height,
                    width
                )
            )

            labels.append(
                (i,"flip")
            )

        for copy in range(affine_copies):
            angle=rng.uniform(
                -max_rotation,
                max_rotation
            )

            dx=int(
                rng.integers(
                    -max_shift,
                    max_shift+1
                )
            )

            dy=int(
                rng.integers(
                    -max_shift,
                    max_shift+1
                )
            )

            # Avoid an almost identical "augmentation".
            if (
                abs(angle)<0.25
                and dx==0
                and dy==0
            ):
                dx=1

            transformed=rigid_transform(
                sequence,
                height,
                width,
                angle,
                dx,
                dy
            )

            augmented.append(
                transformed
            )

            labels.append((
                i,
                f"affine_{copy+1}"
            ))

    return (
        np.stack(
            augmented
        ).astype(np.float32),
        labels
    )