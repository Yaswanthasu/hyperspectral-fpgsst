{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "13fef526",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "def random_flip(patch):\n",
    "    if np.random.rand() > 0.5:\n",
    "        patch = np.flip(patch, axis=1)  # vertical\n",
    "    if np.random.rand() > 0.5:\n",
    "        patch = np.flip(patch, axis=2)  # horizontal\n",
    "    return patch\n",
    "\n",
    "def spectral_jitter(patch, sigma=0.01):\n",
    "    noise = np.random.normal(0, sigma, size=patch.shape)\n",
    "    return patch + noise\n",
    "\n",
    "def band_dropout(patch, p=0.1):\n",
    "    _, h, w = patch.shape\n",
    "    bands = np.random.rand(patch.shape[0]) < p\n",
    "    patch[bands, :, :] = 0\n",
    "    return patch\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
