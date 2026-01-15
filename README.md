# Deep-Codon
## Overview
Repository for the application of Deep-Codon. Messenger RNA (mRNA)-based vaccines have been demonstrated as a promising technology for preventing viral infections due to their safety, rapid production, high potency, and ease of industrialization. There are still some challenges faced by mRNA vaccine design, one of which is how to optimize the coding sequence (CDS) of mRNA to improve its translation efficiency. Here, we proposed a codon attention (CA) mechanism to represent codon selection probabilities for each amino acid of the inputted protein, and then based on CA we developed a deep learning method called Deep-Codon for CDS optimization, which was specifically designed to learn the long-term dependencies both in the amino acid and the codon sequences.   

![Figure](Figure1.png)

## Functions
```bash
test.py_ codes for predict.  
model/attention.py_ codes for Deep-Codon attention.  
model/wordsequence.py_ codes for Deep-Codon model.  
utils/data.py_ codes for input embedding vectors and 64-possible codons as embedding vectors.  
utils/metric.py_ codes for evaluation metric.
```

## Model_weights
[model weights](https://pan.baidu.com/s/1t4pKT-D83VQ6sDRKp0K9YQ pwd=6666)

## System_Requirements
### 1.Hardware requorements
Only a standard computer with enough RAM to support the in-memory operations is required.  

### 2.Software requirements
#### OS requirements
#### The codes are tested on the following Oses:
(1)	Linux x64  
(2)	Windows 10 x64  
#### And the following x86_64 version of Python:
Python 3.X  
#### Python dependencies:
(1)	torch  
(2)	Bio   
(3)	torchvision 

## Installation_Guide
### Download the codes
```bash
git clone https://github.com/SJGLAB/Deep-Codon.git
```
### Prepare the environment
We recommend you to use Anaconda to prepare the environments.
```bash
conda create -n Deep-Codon python=3.10  
conda activate Deep-Codon  
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121    
pip install Bio
```

## Usage and Demo
### Example of Running Command
```bash
python test.py --status dev --batch_size 8 --hidden_dim 3000 --word_emb_dim 3000 --N_layer 3 --dset_dir w3000_1128.dset --load_model_dir  w3000_1128.model
```

## Thanks
We build our model based on the architecture [code](https://github.com/Devil625/Codon_Optimization)

## Contact Us
If you have questions about using Deep-Codon, please contact us.

