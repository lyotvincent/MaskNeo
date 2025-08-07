# MaskNeo
MaskNeo: Advancing Neoantigen Immunogenicity Prediction with Adaptive Feature Masking

## Requirements
- [prot_bert](https://huggingface.co/Rostlab/prot_bert)
- [hla_prot.fasta](https://github.com/ANHIG/IMGTHLA/)

The files above should be placed in the current directory.

## Setup
We use conda to manage python packages.
```shell
git clone https://github.com/lyotvincent/MaskNeo.git
cd MaskNeo
conda env create -n MaskNeo --file environment.yml
conda activate MaskNeo
```

## Usage
An example input data is available in the `example` folder. The input data should have `peptide`, `HLA` and `immunogenicity` columns in a **`tab-delimited text format`**. Run the following command to preprocess the data:
```shell
python data_preprocess.py
```
This will generate a `train.csv` and `valid.csv` file. You can set valid_ratio=1 in the script to treat all data as validation data.

Then, run the following command to train the model:
```shell
bash model_train.sh
```

Run the following command to evaluate the model:
```shell
bash model_test.sh
```
The parameters in the scripts above can be modified as needed.

## Acknowledgements

This work references the following tools:
- [NeoaPred](https://github.com/panda1103/NeoaPred)
- [BigMHC](https://github.com/KarchinLab/bigmhc)
- [IMO](https://github.com/WilliamsToTo/IMO)
- [ProtTrans](https://github.com/agemagician/ProtTrans)
- [DeepImmuno](https://github.com/frankligy/DeepImmuno)
- [DeepNeo](https://deepneo.net/)

We thank the maintainers of these repositories for their valuable contributions.

## Contact

If you have any problems, just raise an issue in this repo.